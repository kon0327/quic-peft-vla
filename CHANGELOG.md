# Changelog

## Unreleased

- Apply each compound rotation to all token/spatial rows with blockwise `bmm`,
  avoiding a broadcasted matrix copy for every position. The previous path
  requested 268.73 GiB for an OpenVLA vocabulary head at batch 4 and 294 tokens.
  Adapter configuration, target coverage and parameterization are unchanged.
- Apply C2 implicitly using the upper triangle of `R A R.T`, where A stores
  the input's antisymmetric pair coordinates. This avoids explicit C2 matrices
  and their repeated-index backward scatter. C2 application uses at least the
  construction precision before casting back to the host activation dtype;
  nonidentity BF16 rounding can differ from the old cast-compound-first path.
- Support plain floating-point `Conv2d` targets with output-channel compound
  rotations, frozen bias, exact identity initialization, and geometry-preserving
  dense kernel merging. Grouped convolution rotations must stay within groups
  (`block_count` divisible by `groups`); unsupported configurations fail before
  modifying the host.
- Add artifact format 2 for convolution or mixed adapters, binding the complete
  convolution geometry. Continue reading and writing the unchanged format 1
  for Linear-only adapters. Old loaders reject format 2 artifacts.
- Export `QuICConv2d` and extend injection, adapter parameter selection,
  disabling, manifests, serialization and merging to both layer types.
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
