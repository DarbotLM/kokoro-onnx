# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#     "kokoro>=0.9.4",
#     "misaki[en]==0.9.4+py314.1",
#     "numpy>=2.5.3,<3",
#     "coloredlogs",
#     "onnx>=1.22.0,<2",
#     "onnx-ir>=1.0.0,<2",
#     "onnxruntime>=1.30.0,<2",
#     "psutil",
#     "onnxscript>=0.7.2,<0.8",
#     "torch>=2.14.0,<3",
# ]
#
# [tool.uv.sources]
# kokoro = { path = "../../kokoro" }
# misaki = { path = "../../misaki", editable = true }
#
# ///

"""
Export a Kokoro checkpoint to ONNX, with waveform and duration outputs.


English (v1.0):
    mkdir -p checkpoints
    wget https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json -O checkpoints/config.json
    wget https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth -O checkpoints/kokoro-v1_0.pth
    uv run scripts/export.py -c checkpoints/config.json -p checkpoints/kokoro-v1_0.pth -o kokoro-v1.0.onnx --fp16 --int8


--fp16 and --int8 write kokoro-*.fp16.onnx and kokoro-*.int8.onnx beside the
full precision file. Both keep float inputs and outputs, so they are drop in.

The duration output reports how many frames each token takes, 600 samples each,
which is what kokoro_onnx uses for timestamps and for sliding window synthesis.
config.json is embedded in the graph metadata, so the vocabulary travels with
the model and no longer has to match the copy shipped in the package.
Speed must stay a float input: exporting it from an int tensor makes the graph
truncate fractional speeds, so speed=0.9 becomes 0 and the model fails.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from kokoro import KModel
from kokoro.custom_stft import CustomSTFT
from kokoro.istftnet import TorchSTFT
from kokoro.model import KModelForONNX

OPSET = 17
SAMPLE_RATE = 24000
FRAME = 600
CONFIG_KEY = "kokoro_config"


SAMPLE_PHONEMES = "həlˈoʊ wˈɜːld, ðɪs ɪz ɐ tˈɛst."


def sample_style(voice: str, length: int) -> torch.Tensor:
    """The real style vector for `length` phonemes, so parity is measured in
    distribution; a random one sends the model far off it."""
    # Chinese voices live in the v1.1-zh repository, everything else in v1.0
    repo = (
        "hexgrad/Kokoro-82M-v1.1-zh" if voice.startswith("z") else "hexgrad/Kokoro-82M"
    )
    from huggingface_hub import hf_hub_download

    pack = torch.load(
        hf_hub_download(repo, f"voices/{voice}.pt"), weights_only=True
    )
    if not 0 < length <= len(pack):
        raise ValueError(f"Voice {voice} has no style for {length} phonemes")
    return pack[length - 1]


def sample_inputs(model: KModelForONNX, voice: str, phonemes: str = SAMPLE_PHONEMES,
                  speed: float = 1.0) -> tuple[torch.Tensor, ...]:
    """Inputs shaped like a real call, used to trace and to verify the graph."""
    vocab = model.kmodel.vocab
    ids = [vocab[p] for p in phonemes if p in vocab]
    return (
        torch.LongTensor([[0, *ids, 0]]),
        sample_style(voice, len(ids)),
        torch.FloatTensor([speed]),
    )


def modernize(checkpoint: Path) -> Path:
    """Rename new style weight norm keys to what KModel expects.

    Checkpoints trained on recent torch store weight norm as
    parametrizations.weight.original0/1, while KModel registers the old
    weight_g/weight_v pair. It loads such a file with strict=False, leaving
    those layers randomly initialized, and the model then emits static.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    renames = {
        ".parametrizations.weight.original0": ".weight_g",
        ".parametrizations.weight.original1": ".weight_v",
    }
    if not any(
        old in key for part in state.values() for key in part for old in renames
    ):
        return checkpoint

    for part in state.values():
        for key in list(part):
            new = key
            for old, replacement in renames.items():
                new = new.replace(old, replacement)
            if new != key:
                part[new] = part.pop(key)

    target = checkpoint.with_suffix(".weightnorm.pth")
    torch.save(state, target)
    print(f"rewrote weight norm keys of {checkpoint.name} into {target.name}")
    return target


def embed_config(path: Path, config: Path) -> None:
    """Carry config.json inside the graph so the vocabulary travels with it."""
    graph = onnx.load(str(path))
    entry = graph.metadata_props.add()
    entry.key = CONFIG_KEY
    entry.value = json.dumps(json.loads(config.read_text(encoding="utf-8")))
    onnx.save(graph, str(path))
    print(f"embedded {config} as metadata key {CONFIG_KEY!r}")


def export(model: KModelForONNX, path: Path, voice: str) -> None:
    torch.onnx.export(
        model,
        args=sample_inputs(model, voice),
        f=str(path),
        export_params=True,
        input_names=["input_ids", "style", "speed"],
        output_names=["waveform", "duration"],
        opset_version=OPSET,
        dynamic_axes={
            "input_ids": {1: "input_ids_len"},
            "waveform": {0: "num_samples"},
            "duration": {0: "input_ids_len"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    print(f"exported {path} ({path.stat().st_size / 1e6:.0f} MB)")


def lower_stft(model: KModelForONNX) -> TorchSTFT:
    """Lower only the transform, preserving all trained speech-model weights."""
    generator = model.kmodel.decoder.generator
    native = generator.stft
    if not isinstance(native, TorchSTFT):
        raise TypeError("Export requires the native STFT as its reference")
    lowered = CustomSTFT(
        native.filter_length, native.hop_length, native.win_length,
        center=True, pad_mode="reflect",
    ).eval()
    size = native.filter_length * 20 + 3
    samples = torch.arange(size, dtype=torch.float32)
    signals = torch.stack([
        torch.ones(size), torch.sin(2 * torch.pi * 440 * samples / SAMPLE_RATE),
        torch.cos(torch.pi * samples),
    ])
    with torch.inference_mode():
        magnitude, phase = native.transform(signals)
        torch.testing.assert_close(
            lowered.inverse(magnitude, phase), native.inverse(magnitude, phase),
            atol=2e-6, rtol=2e-5,
        )
        lowered_magnitude, lowered_phase = lowered.transform(signals)
        torch.testing.assert_close(lowered_magnitude, magnitude, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            lowered_magnitude * torch.sin(lowered_phase), magnitude * torch.sin(phase),
            atol=2e-5, rtol=2e-5,
        )
        torch.testing.assert_close(
            lowered_magnitude * torch.cos(lowered_phase), magnitude * torch.cos(phase),
            atol=2e-5, rtol=2e-5,
        )
    generator.stft = lowered
    print("validated native-equivalent real-valued STFT lowering")
    return native


def verify(model: KModelForONNX, path: Path, voice: str, strict: bool = True,
           native: TorchSTFT | None = None) -> dict:
    """Check the unmodified graph's durations and raw audio.

    Excitation is stochastic, so waveform samples need not match across runtimes.
    Full precision durations must match; quantization may change durations.
    """
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if {i.name for i in session.get_inputs()} != {"input_ids", "style", "speed"}:
        raise ValueError("Export must accept only input_ids, style and speed")
    results = []
    for speed, phonemes in [(1.0, SAMPLE_PHONEMES), (0.9, SAMPLE_PHONEMES.split(",")[0] + "."),
                             (1.2, SAMPLE_PHONEMES * 2)]:
        inputs = sample_inputs(model, voice, phonemes, speed)
        feed = dict(zip(("input_ids", "style", "speed"), (t.numpy() for t in inputs)))
        lowered = model.kmodel.decoder.generator.stft
        if native is not None:
            model.kmodel.decoder.generator.stft = native
        try:
            with torch.inference_mode():
                expected, expected_duration = model(*inputs)
        finally:
            model.kmodel.decoder.generator.stft = lowered
        waveform, duration = session.run(["waveform", "duration"], feed)
        if strict:
            np.testing.assert_array_equal(duration, expected_duration.numpy())
        if not np.isfinite(waveform).all() or waveform.ndim != 1 or not waveform.size:
            raise ValueError("Export returned empty or nonfinite waveform")
        if (duration.shape != (inputs[0].shape[1],) or not np.isfinite(duration).all()
                or np.any(duration <= 0) or not np.equal(duration, np.floor(duration)).all()
                or len(waveform) != int(np.sum(duration)) * FRAME):
            raise ValueError("Export waveform and token durations disagree")
        peak = float(np.abs(waveform).max())
        rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
        dc = float(np.mean(waveform.astype(np.float64)))
        reference_rms = float(expected.square().mean().sqrt())
        if not 0.001 < peak <= 1 or rms <= 0.001 or abs(dc) > 0.001:
            raise ValueError(f"Invalid exported audio: peak={peak}, rms={rms}, dc={dc}")
        if not 0.5 < rms / reference_rms < 2:
            raise ValueError("Export RMS diverges from the native model")
        result = {
            "speed": speed, "tokens": inputs[0].shape[1], "samples": len(waveform),
            "peak": peak, "rms": rms, "dc": dc, "reference_rms": reference_rms,
        }
        results.append(result)
        print(f"  verified {result}")
    return {"duration_parity": strict, "waveform_sample_parity": "not_applicable_stochastic_excitation",
            "cases": results}


def quantize(path: Path, fp16: bool, int8: bool) -> list[Path]:
    """Write reduced precision copies beside the full precision export."""
    written = []

    if fp16:
        import onnx_ir as ir
        from onnxruntime.transformers.float16 import convert_float_to_float16

        target = path.with_suffix(".fp16.onnx")
        # The OnnxModel wrapper also runs symbolic Cast optimizations, which
        # fail on this graph with ONNX 1.22. Convert and validate directly.
        graph = convert_float_to_float16(onnx.load(str(path)), keep_io_types=True)
        model = ir.from_proto(graph)
        # Loop bodies capture outer values, so ordering must include subgraphs.
        model.graph.sort()
        graph = ir.to_proto(model)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(target))
        written.append(target)

    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        target = path.with_suffix(".int8.onnx")
        quantize_dynamic(str(path), str(target), weight_type=QuantType.QInt8)
        written.append(target)

    for target in written:
        print(f"quantized {target} ({target.stat().st_size / 1e6:.0f} MB)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Kokoro model to ONNX")
    parser.add_argument("-c", "--config", default="checkpoints/config.json")
    parser.add_argument("-p", "--checkpoint", default="checkpoints/kokoro-v1_0.pth")
    parser.add_argument("-o", "--output", default="kokoro.onnx")
    parser.add_argument(
        "-v", "--voice", default="af_heart", help="voice used to verify"
    )
    parser.add_argument("--fp16", action="store_true", help="also write a fp16 copy")
    parser.add_argument("--int8", action="store_true", help="also write an int8 copy")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = modernize(Path(args.checkpoint))
    model = KModelForONNX(
        KModel(config=args.config, model=str(checkpoint), disable_complex=False)
    ).eval()

    native = lower_stft(model)
    with tempfile.TemporaryDirectory(prefix=".kokoro-export-", dir=output.parent) as root:
        candidate = Path(root) / output.name
        export(model, candidate, args.voice)
        embed_config(candidate, Path(args.config))
        variants = quantize(candidate, args.fp16, args.int8)
        for path in [candidate, *variants]:
            if not args.skip_verify:
                print(path.name)
                result = verify(model, path, args.voice, strict=path == candidate, native=native)
                with path.open("rb") as stream:
                    result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
                result["voice"] = args.voice
                path.with_suffix(".verification.json").write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
        for path in [candidate, *variants]:
            target = output.parent / path.name
            path.replace(target)
            report = path.with_suffix(".verification.json")
            if report.exists():
                report.replace(target.with_suffix(".verification.json"))
            else:
                target.with_suffix(".verification.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
