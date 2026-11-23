---
title: Releases, downloads, and checksums
---

Use the fork release pages, not similarly named upstream PyPI/npm packages:

| Fork | Release page | Typical Python artifact |
| --- | --- | --- |
| Kokoro ONNX | [Downloads](https://github.com/DarbotLM/kokoro-onnx/releases) | `kokoro_onnx-0.6.1-py3-none-any.whl` |
| Kokoro | [Downloads](https://github.com/DarbotLM/kokoro/releases) | `kokoro-0.9.4-py3-none-any.whl` |
| Misaki | [Downloads](https://github.com/DarbotLM/misaki/releases) | `misaki-0.9.4+py314.1-py3-none-any.whl` |

## November 22, 2026 build

The `py314-2026.11.22` releases supersede the package bytes from September:

- [ONNX release](https://github.com/DarbotLM/kokoro-onnx/releases/tag/py314-2026.11.22):
  corrected release/export workflow, export lowering and verification, and clean
  Python 3.14 package artifacts.
- [Kokoro release](https://github.com/DarbotLM/kokoro/releases/tag/py314-2026.11.22):
  corrected real-valued STFT reconstruction, Python wheel/source distribution,
  npm package, production web-demo ZIP, checksums, and provenance.
- [Misaki release](https://github.com/DarbotLM/misaki/releases/tag/py314-2026.11.22):
  paired Python wheel/source distribution, checksums, and provenance.

The Python package versions remain `0.6.1`, `0.9.4`, and `0.9.4+py314.1`; the
release tag records a corrected immutable build. The corrected Kokoro source
must be used to export a fresh ONNX graph; installing the Python package does
not modify an older model file.

## September 17, 2026 build

The `py314-2026.09.17` releases are published:

- [ONNX release](https://github.com/DarbotLM/kokoro-onnx/releases/tag/py314-2026.09.17):
  Python wheel, source distribution, checksums, and provenance.
- [Kokoro release](https://github.com/DarbotLM/kokoro/releases/tag/py314-2026.09.17):
  Python wheel/source distribution, `kokoro-js-1.2.1.tgz`, production web-demo ZIP,
  checksums, and provenance.
- [Misaki release](https://github.com/DarbotLM/misaki/releases/tag/py314-2026.09.17):
  Python wheel, source distribution, checksums, and provenance.

These are **prereleases**, built from clean committed snapshots. Known
CustomSTFT and macOS INT8 limitations prevent a blanket production certification.
The browser build uses stable React 19.3 and Vite 8.3; the prerelease label refers
to the fork stack's remaining validation gates, not to those frontend packages.

Release tags distinguish fork builds even when the Python package version is
unchanged. Record both the source commit and asset checksum. A prerelease label
means production acceptance remains incomplete; do not infer stability from the
wheel filename.

## Verify and install

Download the wheel, source distribution if needed, and `SHA256SUMS.txt`.
Compare each downloaded artifact against the checksum manifest:

```powershell
Get-FileHash .\kokoro_onnx-0.6.1-py3-none-any.whl -Algorithm SHA256
```

Install the paired Python artifacts together in a Python 3.14 environment:

```powershell
python -m pip install ".\misaki-0.9.4+py314.1-py3-none-any.whl" ".\kokoro-0.9.4-py3-none-any.whl" ".\kokoro_onnx-0.6.1-py3-none-any.whl"
```

A `py3-none-any` wrapper wheel does not make its native dependencies universal.
Consult the [platform matrix](compatibility.md). Model weights, voice banks,
spaCy models, and dictionaries remain separate downloads.

## Clean-build policy

Build from the tagged source in an isolated directory, not a stale `dist` folder.
Publish only explicitly selected artifacts and checksums. Do not overwrite an
existing release asset without an intentional correction process. Release notes
must disclose known STFT and model/provider limitations.
