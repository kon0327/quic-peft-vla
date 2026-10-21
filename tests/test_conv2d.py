"""Convolution geometry, channel rotation, gradients, and artifact contracts."""

import copy
import tempfile
import unittest

import torch
from torch import nn
from torch.func import functional_call

from quic_petf_vla import (
    AdapterConfig, QuICConv2d, QuICLinear, adapter_manifest, adapter_parameters,
    disabled_adapters, inject, load_adapter, merge_adapters, save_adapter,
)


HOST = {"checkpoint_sha256": "b" * 64, "family": "conv-test"}


class Conv2dTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(44)

    def test_patch_projection_identity_including_bfloat16_bias_and_unbatched(self):
        for dtype in (torch.float64, torch.bfloat16):
            for bias in (False, True):
                for batched in (False, True):
                    with self.subTest(dtype=dtype, bias=bias, batched=batched):
                        layer = nn.Conv2d(3, 32, 14, stride=14, bias=bias, dtype=dtype)
                        shape = (2, 3, 28, 42) if batched else (3, 28, 42)
                        values = torch.randn(shape, dtype=dtype)
                        expected = layer(values)
                        wrapped = QuICConv2d(layer, AdapterConfig(block_count=8))
                        torch.testing.assert_close(wrapped(values), expected, atol=0, rtol=0)
                        self.assertEqual(tuple(wrapped.weight.shape), (32, 3, 14, 14))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_bfloat16_real_patch_shapes_identity_gradients_and_reload(self):
        for channels in (1024, 1152):
            with self.subTest(channels=channels):
                host = nn.Sequential(nn.Conv2d(3, channels, 14, stride=14, bias=True,
                                               device="cuda", dtype=torch.bfloat16))
                original = copy.deepcopy(host)
                values = torch.randn(4, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
                expected = original(values)
                inject(host, {"0": AdapterConfig(block_count=8)})
                torch.testing.assert_close(host(values), expected, atol=0, rtol=0)
                parameters = adapter_parameters(host)
                optimizer = torch.optim.AdamW(parameters, lr=1e-3)
                readout = torch.randn_like(expected)
                (host(values).float() * readout.float()).mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    and p.grad.abs().max() > 0 for p in parameters))
                optimizer.step()
                self.assertGreater(parameters[0].abs().max().item(), 0)
                self.assertIsNone(host[0].weight.grad)
                torch.testing.assert_close(host[0].weight, original[0].weight, atol=0, rtol=0)
                learned = host(values).detach()
                with tempfile.TemporaryDirectory() as directory:
                    save_adapter(host, directory, HOST)
                    load_adapter(original, directory, HOST)
                    torch.testing.assert_close(original(values), learned, atol=0, rtol=0)

    def test_rotated_and_merged_kernels_preserve_geometry_and_bias(self):
        geometries = (
            {"stride": (2, 1), "padding": (2, 1), "dilation": (2, 1)},
            {"padding": (2, 1), "padding_mode": "reflect", "dilation": (2, 1)},
            {"padding": 1, "padding_mode": "replicate"},
            {"padding": 1, "padding_mode": "circular"},
            {"padding": "same"},
            {"padding": "valid"},
            {"padding": (2, 1), "padding_mode": "reflect", "groups": 2},
        )
        config = AdapterConfig(block_count=2, base_size=3, construction_dtype="float64")
        for geometry in geometries:
            with self.subTest(geometry=geometry):
                base = nn.Conv2d(4, 16, (3, 2), dtype=torch.float64, **geometry)
                wrapped = QuICConv2d(base, config).eval()
                with torch.no_grad():
                    wrapped.adapter.parameter.normal_(std=0.2)
                values = torch.randn(2, 4, 8, 9, dtype=torch.float64)
                merged = wrapped.merged_conv2d()
                # Explicit rotation of the output axis of each flattened kernel.
                rotation = wrapped.adapter(torch.eye(16, dtype=torch.float64)).T
                expected_weight = (rotation @ base.weight.flatten(1)).reshape_as(base.weight)
                torch.testing.assert_close(merged.weight, expected_weight, atol=1e-14, rtol=1e-14)
                torch.testing.assert_close(wrapped(values), merged(values), atol=1e-13, rtol=1e-13)
                torch.testing.assert_close(merged.bias, base.bias, atol=0, rtol=0)
                self.assertIs(type(merged), nn.Conv2d)
                self.assertFalse(merged.training)
                self.assertFalse(any(p.requires_grad for p in merged.parameters()))
                for attr in ("stride", "padding", "dilation", "groups", "padding_mode"):
                    self.assertEqual(getattr(merged, attr), getattr(base, attr))
                zero = torch.zeros_like(values)
                expected_bias = base.bias[None, :, None, None].expand_as(wrapped(zero))
                torch.testing.assert_close(wrapped(zero), expected_bias, atol=0, rtol=0)

    def test_group_rotations_preserve_input_connectivity(self):
        wrapped = QuICConv2d(
            nn.Conv2d(4, 16, 1, groups=4, bias=False, dtype=torch.float64),
            AdapterConfig(block_count=4, base_size=2, construction_dtype="float64"),
        )
        with torch.no_grad():
            wrapped.adapter.parameter.normal_(std=0.2)
        values = torch.zeros(1, 4, 2, 3, dtype=torch.float64)
        values[:, 1] = torch.randn(2, 3, dtype=torch.float64)
        result = wrapped(values)
        self.assertGreater(result[:, 4:8].abs().max().item(), 0)
        torch.testing.assert_close(result[:, :4], torch.zeros_like(result[:, :4]), atol=0, rtol=0)
        torch.testing.assert_close(result[:, 8:], torch.zeros_like(result[:, 8:]), atol=0, rtol=0)

    def test_group_validation_is_atomic_and_specialized_layers_are_rejected(self):
        model = nn.Sequential(nn.Linear(4, 16), nn.Conv2d(8, 32, 1, groups=4))
        with self.assertRaisesRegex(ValueError, "preserve convolution groups"):
            inject(model, {"0": AdapterConfig(block_count=2), "1": AdapterConfig(block_count=2)})
        self.assertIs(type(model[0]), nn.Linear)
        self.assertIs(type(model[1]), nn.Conv2d)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

        class SpecializedConv(nn.Conv2d):
            pass

        with self.assertRaises(TypeError):
            QuICConv2d(SpecializedConv(3, 16, 1), AdapterConfig(block_count=2))

    def test_gradcheck_and_frozen_host_with_downstream_linear(self):
        model = nn.Sequential(nn.Conv2d(4, 12, 1, groups=2, dtype=torch.float64),
                              nn.Tanh(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                              nn.Linear(12, 3, dtype=torch.float64))
        original = copy.deepcopy(model)
        inject(model, {"0": AdapterConfig(block_count=2, base_size=3, construction_dtype="float64")})
        values = torch.randn(1, 4, 2, 3, dtype=torch.float64, requires_grad=True)
        entries = model[0].adapter.parameter.detach().clone().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda theta, x: functional_call(model, {"0.adapter.base.skew_entries": theta}, (x,)),
            (entries, values), eps=1e-6, atol=1e-6, rtol=1e-4,
        ))
        optimizer = torch.optim.AdamW(adapter_parameters(model), lr=1e-2)
        initial = model[0].adapter.parameter.detach().clone()
        model(values).square().mean().backward()
        self.assertGreater(model[0].adapter.parameter.grad.norm().item(), 0)
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if "adapter" not in name))
        optimizer.step()
        self.assertFalse(torch.equal(initial, model[0].adapter.parameter))
        torch.testing.assert_close(model[0].weight, original[0].weight, atol=0, rtol=0)
        torch.testing.assert_close(model[0].bias, original[0].bias, atol=0, rtol=0)
        torch.testing.assert_close(model[4].weight, original[4].weight, atol=0, rtol=0)

    def test_dtype_transfer_and_nested_disabled_context(self):
        model = nn.Sequential(nn.Conv2d(3, 16, 1))
        original = copy.deepcopy(model)
        inject(model, {"0": AdapterConfig(block_count=2, base_size=3)})
        model.to(torch.bfloat16)
        original.to(torch.bfloat16)
        self.assertEqual(model[0].adapter.parameter.dtype, torch.float32)
        with torch.no_grad():
            model[0].adapter.parameter.normal_(std=0.2)
        values = torch.randn(2, 3, 4, 5, dtype=torch.bfloat16)
        adapted = model(values)
        self.assertEqual(adapted.dtype, torch.bfloat16)
        with disabled_adapters(model):
            torch.testing.assert_close(model(values), original(values), atol=0, rtol=0)
            with disabled_adapters(model):
                self.assertFalse(model[0].adapter_enabled)
            self.assertFalse(model[0].adapter_enabled)
        self.assertTrue(model[0].adapter_enabled)
        torch.testing.assert_close(model(values), adapted, atol=0, rtol=0)


class Conv2dArtifactTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(45)
        self.original = nn.Sequential(nn.Conv2d(4, 16, (3, 2), padding=(2, 1),
                                                groups=2, padding_mode="reflect", dtype=torch.float64),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Linear(16, 16, dtype=torch.float64))
        self.model = copy.deepcopy(self.original)
        config = AdapterConfig(block_count=2, base_size=3, construction_dtype="float64")
        inject(self.model, {"0": config, "3": config})
        with torch.no_grad():
            for parameter in adapter_parameters(self.model):
                parameter.normal_(std=0.2)
        self.values = torch.randn(2, 4, 6, 7, dtype=torch.float64)

    def test_mixed_conv_linear_v2_roundtrip_and_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = save_adapter(self.model, tmp, HOST)
            self.assertEqual(report["format_version"], 2)
            conv = report["manifest"][0]
            self.assertEqual(conv["module_type"], "Conv2d")
            self.assertEqual(conv["kernel_size"], [3, 2])
            self.assertEqual(conv["groups"], 2)
            restored = copy.deepcopy(self.original)
            self.assertEqual(load_adapter(restored, tmp, HOST), adapter_manifest(self.model))
            torch.testing.assert_close(restored(self.values), self.model(self.values), atol=0, rtol=0)
            merge_adapters(restored, names=["0"])
            self.assertIs(type(restored[0]), nn.Conv2d)
            self.assertIs(type(restored[3]), QuICLinear)
            torch.testing.assert_close(restored(self.values), self.model(self.values), atol=1e-13, rtol=1e-13)
            merge_adapters(restored)
            self.assertIs(type(restored[3]), nn.Linear)
            self.assertEqual(adapter_parameters(restored), [])
            torch.testing.assert_close(restored(self.values), self.model(self.values), atol=1e-13, rtol=1e-13)

    def test_same_weight_shape_with_different_geometry_rejected_before_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter(self.model, tmp, HOST)
            changes = ({"stride": (2, 1)}, {"padding": (1, 1)}, {"dilation": (2, 1)},
                       {"padding_mode": "circular"}, {"kernel_size": (2, 3)},
                       {"in_channels": 2, "groups": 1})
            for changed in changes:
                with self.subTest(changed=changed):
                    host = copy.deepcopy(self.original)
                    args = {"in_channels": 4, "out_channels": 16, "kernel_size": (3, 2),
                            "padding": (2, 1), "groups": 2, "padding_mode": "reflect"}
                    args.update(changed)
                    host[0] = nn.Conv2d(**args, dtype=torch.float64)
                    self.assertEqual(host[0].weight.numel(), self.original[0].weight.numel())
                    with self.assertRaisesRegex(ValueError, "manifest"):
                        load_adapter(host, tmp, HOST)
                    self.assertIs(type(host[0]), nn.Conv2d)
                    self.assertTrue(all(p.requires_grad for p in host.parameters()))

    def test_linear_only_artifacts_keep_v1_manifest_and_load(self):
        host = nn.Sequential(nn.Linear(3, 16, dtype=torch.float64))
        model = copy.deepcopy(host)
        inject(model, {"0": AdapterConfig(block_count=2, base_size=3, construction_dtype="float64")})
        with tempfile.TemporaryDirectory() as tmp:
            report = save_adapter(model, tmp, HOST)
            self.assertEqual(report["format_version"], 1)
            self.assertNotIn("module_type", report["manifest"][0])
            self.assertEqual(load_adapter(host, tmp, HOST), report["manifest"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
