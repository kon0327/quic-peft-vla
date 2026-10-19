# Contributing

Use Python 3.10 or newer and install the appropriate PyTorch build for your
platform first. A CPU build is sufficient for the default suite.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest
python examples/linear_adapter.py
python -m build
python -m twine check dist/*
```

Tests exercise determinant values and gradients, orthogonality, identity
initialization, frozen hosts, bias placement, parameter counts, dtype handling,
adapter save/load, and rejection of tampered artifacts. The CUDA test skips on
CPU-only machines. The suite requires neither model checkpoints nor downloaded
datasets, Transformers, OVLAB, or a benchmark simulator.

Changes to mathematics, serialization, target selection, or dtype handling
should include a test for the behavior they affect. Preserve artifact checks;
never accept an identity mismatch silently. Document artifact-format changes
and compatibility consequences in CHANGELOG.md.

Before a release, update the package version, changelog, and citation metadata;
build and check both the wheel and source distribution in a clean checkout.
Create a release tag only after these checks pass. Publishing packages and
pushing Git history are separate, explicit maintainer actions.
