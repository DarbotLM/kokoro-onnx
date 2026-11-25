---
title: Building
description: Build the Python runtime and export Kokoro checkpoints to ONNX.
---

Build and publish with Python 3.14, the minimum and default interpreter.
Package builds require Hatchling 1.32+ (below 2.0).

## GitHub release artifacts

Build from an isolated archive of committed source, not an existing `dist` folder:

```powershell
uv run --no-project --python 3.14 scripts\build_release.py --repo . --output C:\releases\kokoro-onnx
```

The output directory must not already exist. The builder emits a wheel, source
distribution, SHA-256 manifest, and source-commit provenance. Pass `--javascript`
when building the sibling Kokoro repository to include its tested npm package
and production web-demo ZIP. An optional `--index-url` selects a build-dependency
mirror without changing the project lock.

A clean source build is not a claim that every platform/model is production-ready.
Retain the documented compatibility limitations in GitHub release notes.
See the [release guide](https://darbotlm.github.io/kokoro-onnx/releases/).

## Publish new version

```console
rm -rf dist
uv build
UV_PUBLISH_TOKEN="pypi token here" uv publish
```

## Format and lint

```console
uv run ruff format
uv run ruff check
```

## Compatibility tests

```console
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

Model export uses Python 3.14 and the sibling `..\kokoro` and `..\misaki`
source checkouts. Their Python 3.14 builds must be ready before resolving the
export environment:

```console
uv run scripts/export.py --help
```

The exporter requires PyTorch 2.14+, ONNX 1.22+, and ONNXScript 0.7.2+.
PyTorch and ONNX stay below their next major releases; ONNXScript stays on 0.7.x.
FP16 conversion uses ONNX IR 1.x to sort the graph and nested Loop dependencies
before ONNX validation, avoiding unsupported symbolic Cast optimizations.
It deliberately uses the legacy TorchScript exporter (`dynamo=False`) for the
model's dynamic durations and existing ONNX graph contract.

### Audio reconstruction

The exporter loads trained weights with native STFT, then lowers only the
transform to the sibling Kokoro checkout's real-valued `CustomSTFT`. Native
complex operations cannot be traced by this exporter. The lowering must preserve
reflection padding, centered Hann windows, one-sided Fourier weights and
squared-window overlap normalization. The exporter checks these semantics before
tracing; older approximate implementations fail instead of producing a bad model.

Exporting requires the corrected sibling Kokoro source. Updating the Python
runtime package alone does not repair an existing `.onnx` file. Export a fresh
full-precision graph using the original checkpoint and configuration:

```powershell
uv run scripts\export.py -c checkpoints\config.json -p checkpoints\kokoro-v1_0.pth -o kokoro-v1.0.onnx -v am_liam
```

The production graph retains normal random excitation and its three inputs:
`input_ids`, `style`, and floating-point `speed`. It contains no injected noise
inputs, noise removal, or post-synthesis cleanup filter. Full waveform samples
are not compared across runtimes because their random generators differ.

By default, the existing verifier checks duration parity, variable input lengths,
fractional speeds, finite/non-silent output, amplitude and raw DC offset.
Model replacement occurs only after all requested variants pass. A sibling
`.verification.json` records the output SHA-256 and measurements.
`--skip-verify` explicitly bypasses runtime verification and does not emit that
report; it is not appropriate for an a2swe production model.

For a2swe, update `library/assets/speech/models.json` when replacing model bytes,
then run its core `audio-render --root PROJECT --engine both --voice PROFILE`
command. Audio and comparison evidence go in the project's `qc/audio/PROFILE/`.
Its core checks the registered model hash before synthesis.
Existing rendered videos remain unchanged until regenerated.

Voice fetching uses Python 3.14 and PyTorch 2.14+ (below 3.0):

```console
uv run --python 3.14 scripts/fetch_voices.py
```

## Gradio demo

The demo requires Gradio 6.27+ (below 7.0), installed by its inline script
dependencies:

```console
uv run examples/app.py
```

Download the model and voices listed in the script before launching. Gradio 6
receives its theme in `launch()`, not `Blocks()`. Importing the example no longer
loads a model or starts a server; `create_app(kokoro)` accepts an existing model.
The `/create` API still returns audio and phonemes, with optional 50/50 voice
blending. Select "None" in the blend dropdown to return to a single voice.

## Log

Enable log with

```console
LOG_LEVEL=DEBUG python main.py
```
