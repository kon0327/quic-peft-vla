# Changelog

## Unreleased

- Standalone repository, CPU example, numerical and artifact tests, package
  build checks, and GitHub Actions CI.
- Add the canonical QuIC paper citation and identify the extracted
  `quic-petf-mpv` adapter source.

## 0.1.0

- Extract determinant compound adapters C1–C3 from the supplied
  `quic-petf-mpv/quic_openvla/adapter.py` without changing its implementation.
- Expose explicit linear-layer injection, frozen-host adapter parameter
  selection, paired baseline evaluation, and in-memory merging.
- Save and load adapter-only safetensors with host identity, target manifest,
  tensor digest, and artifact binding checks.
- Keep runtime dependencies limited to PyTorch and safetensors.
