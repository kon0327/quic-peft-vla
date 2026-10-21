# quic-peft-vla

Portable QuIC method for frozen dense PyTorch models, extracted from
`quic-petf-mpv/quic_openvla/adapter.py`. The GitHub project is named
`quic-peft-vla`. Version 0.1.0 keeps its published distribution name
`quic-petf-vla` and import namespace `quic_petf_vla` for compatibility with
existing OpenVLA checkpoints and pinned installs. It depends only on PyTorch and safetensors; OpenVLA, OVLAB and
benchmark dependencies belong to callers.

## Standalone installation

Python 3.10 or newer is required. Install the appropriate PyTorch build for your
platform, then clone and install this package:

```bash
git clone https://github.com/kon0327/quic-peft-vla.git
cd quic-peft-vla
python -m pip install .
python examples/linear_adapter.py
```

The CPU example trains six adapter parameters in a tiny frozen host, saves the
adapter, and checks identical predictions after reloading. No dataset or model
download is needed. The package has no runtime dependence on its checkout
location or on a surrounding project.

## Method API

`inject(model, {"exact.module.name": AdapterConfig(...)})` freezes the host and
creates output-side determinant C1/C2/C3 adapters. `adapter_parameters(model)`
is the optimizer parameter list. Bias is frozen and unchanged. Plain,
materialized, floating-point `torch.nn.Linear` and `torch.nn.Conv2d` layers are
accepted; specialized or quantized subclasses remain unsupported.

`QuICConv2d` rotates output channels independently at every spatial position.
It supports patch projections such as a `32 × 3 × 14 × 14` kernel, batched and
unbatched inputs, and retains the host's stride, padding, dilation and padding
mode. For grouped convolutions, `block_count` must be divisible by `groups`;
this aligns rotation blocks with output groups and preserves connectivity when
merging. The usual output divisibility and compound-size checks also apply.
Depthwise convolutions with only one output channel per group cannot fit a
nontrivial compound block and are rejected. Bias-bearing convolutions use the
same host-plus-delta calculation as linear layers to preserve exact identity
initialization even with BF16 fused bias rounding.

Compound application batches over rotation blocks and treats token/spatial
positions as matrix rows. It does not replicate a compound matrix per token.
C2 is applied implicitly: pair coordinates form an antisymmetric matrix A,
and the upper triangle of `R A R.T` gives the same determinant compound action.
This avoids forming C2 and its repeated-index backward scatter. Application
uses at least construction precision, then casts back to the activation dtype;
nonidentity BF16 rounding can differ from the previous explicit BF16 compound.
C1/C3 application and the optional `matrices()` inspection API still construct
explicit matrices; their size remains part of the measured memory budget.

`save_adapter(model, directory, host_identity, metadata=None)` writes
`adapter.json` and `adapter.safetensors`. Host identity must contain an immutable
SHA256 digest. Callers bind their preprocessing, normalization and data split in
host identity and metadata. `load_adapter(model, directory, host_identity)`
verifies the exact identity, target manifest, tensor digest, binding, shape and
dtype before changing the model. Existing artifacts cannot be overwritten.

`disabled_adapters(model)` selects the same frozen host for paired comparisons.
`merge_adapters(model)` replaces wrappers with frozen `Linear` or `Conv2d`
layers. A `QuICConv2d` also exposes `merged_conv2d()` for an individual kernel.
These operations perform an in-memory dense layer merge, not a checkpoint
export or whole-model compression.

Linear-only adapters retain format version 1 and their existing manifests.
Artifacts containing convolutions use version 2 and bind the full convolution
geometry, including kernel size, stride, padding, dilation, groups and padding
mode. The loader accepts both versions and rejects geometry mismatches before
injection. Older package versions cannot load version 2 artifacts.

## Development, citation, and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for testing and building distributions,
[CHANGELOG.md](CHANGELOG.md) for changes, and [CITATION.cff](CITATION.cff) for
software and research citation metadata.

QuIC was introduced by Snehal Raj and Brian Coyle in [*QuIC: Quantum-Inspired
Compound Adapters for Parameter Efficient Fine-Tuning* (2025)](https://arxiv.org/abs/2502.06916).
The compound-PEFT snapshot's [README citation](https://github.com/kon0327/OpenVLABenchmark/blob/main/external/compound-peft/README.md)
lists a different title and author list for this same arXiv ID. This project
uses the title and authors in the [canonical arXiv record](https://arxiv.org/abs/2502.06916).
`f1e71f686d2f4da94d11a67fd4bafbe516d007d1`; its SHA256 and extraction history
are recorded in [PROVENANCE.json](PROVENANCE.json).

Released under the [MIT license](LICENSE).
