"""CPU training and adapter roundtrip on a tiny frozen PyTorch host."""

import copy
from pathlib import Path
import tempfile

import torch
from torch import nn

from quic_petf_vla import (
    AdapterConfig, adapter_parameters, inject, load_adapter, save_adapter,
)


def main():
    torch.manual_seed(41)
    model = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2))
    frozen_host = copy.deepcopy(model)
    inputs, targets = torch.randn(8, 4), torch.randn(8, 2)
    reference = frozen_host(inputs)
    inject(model, {"0": AdapterConfig(orders=(1, 2), block_count=2, base_size=3)})
    torch.testing.assert_close(model(inputs), reference, rtol=0, atol=0)
    parameters = adapter_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.0)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()
    # A real host supplies a content digest of its checkpoint. This synthetic
    # example identifies its reproducibly initialized dense weights explicitly.
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in frozen_host.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(bytes(tensor.detach().contiguous().view(torch.uint8).flatten().tolist()))
    identity = {"family": "example_dense_model", "checkpoint_sha256": digest.hexdigest()}
    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / "adapter"
        save_adapter(model, artifact, identity, metadata={"optimizer_steps": 5})
        restored = copy.deepcopy(frozen_host)
        load_adapter(restored, artifact, identity)
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    print(f"Trained {sum(p.numel() for p in parameters)} adapter parameters; save/load outputs match.")


if __name__ == "__main__":
    main()
