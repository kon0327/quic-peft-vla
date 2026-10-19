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

## OpenVLA integration

To install the method and its OpenVLA integration from the OVLAB checkout root:

```bash
python -m pip install ./external/quic-peft-vla ./external/openvla-quic
```

The OpenVLA fork declares the compatible `quic-petf-vla==0.1.0` distribution;
installing both local projects supplies that dependency without requiring a
published package index entry.
OVLAB pins this repository as its `external/quic-peft-vla` submodule. OpenVLA
loading, action normalization, dataset handling, experiment configuration,
training orchestration, and benchmark rollouts belong to the host integration.
They are not required by this method package.

## Method API

`inject(model, {"exact.module.name": AdapterConfig(...)})` freezes the host and
creates output-side determinant C1/C2/C3 adapters. `adapter_parameters(model)`
is the optimizer parameter list. Bias is frozen and unchanged. Only plain,
materialized, floating-point `torch.nn.Linear` layers are accepted.

`save_adapter(model, directory, host_identity, metadata=None)` writes
`adapter.json` and `adapter.safetensors`. Host identity must contain an immutable
SHA256 digest. Callers bind their preprocessing, normalization and data split in
host identity and metadata. `load_adapter(model, directory, host_identity)`
verifies the exact identity, target manifest, tensor digest, binding, shape and
dtype before changing the model. Existing artifacts cannot be overwritten.

`disabled_adapters(model)` selects the same frozen host for paired comparisons.
`merge_adapters(model)` is an in-memory dense layer merge, not a checkpoint
export or whole-model compression.

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
