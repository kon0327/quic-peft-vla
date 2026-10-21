"""Compound application must not replicate rotation matrices across tokens."""
import unittest

import torch
from torch.nn import functional as F

from quic_petf_vla import AdapterConfig, CompoundOperator


class ApplicationTests(unittest.TestCase):
    def test_arbitrary_prefix_shapes_match_explicit_block_outputs_and_gradients(self):
        torch.manual_seed(192)
        for shared in (False, True):
            for orders in ((1,), (1, 2), (1, 2, 3)):
                config = AdapterConfig(orders=orders, block_count=2, base_size=4,
                                       share_across_blocks=shared, construction_dtype="float64")
                operator = CompoundOperator(32, config, device=torch.device("cpu"))
                with torch.no_grad():
                    operator.parameter.normal_(std=0.1)
                for shape in ((32,), (3, 32), (2, 3, 32), (2, 3, 4, 32)):
                    with self.subTest(shared=shared, orders=orders, shape=shape):
                        # A strided final axis also covers convolution movedim inputs.
                        values = torch.randn(*shape[:-1], 64, dtype=torch.float64)[..., ::2].requires_grad_()
                        blocks = values.reshape(*shape[:-1], 2, 16)
                        expected = []
                        for block in range(2):
                            pieces, start = [], 0
                            for matrix, size in zip(operator.matrices(), operator.compound_sizes):
                                pieces.append(F.linear(blocks[..., block, start:start + size], matrix[0 if shared else block]))
                                start += size
                            pieces.append(blocks[..., block, start:])
                            expected.append(torch.cat(pieces, dim=-1))
                        expected = torch.stack(expected, dim=-2).reshape_as(values)
                        actual = operator(values)
                        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
                        readout = torch.randn_like(actual)
                        actual_grads = torch.autograd.grad((actual * readout).sum(), (values, operator.parameter))
                        expected_grads = torch.autograd.grad((expected * readout).sum(), (values, operator.parameter))
                        for found, reference in zip(actual_grads, expected_grads):
                            torch.testing.assert_close(found, reference, atol=1e-11, rtol=1e-11)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_openvla_vocabulary_head_batch4_forward_backward_memory(self):
        # The old broadcast matmul requested 268.73 GiB for this exact shape.
        operator = CompoundOperator(32064, AdapterConfig(block_count=8), device=torch.device("cuda"))
        values = torch.randn(4, 294, 32064, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual = operator(values)
        torch.testing.assert_close(actual, values, atol=0, rtol=0)
        actual.float().mean().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        # A blockwise but explicit C2 backward still peaks at 3.814 GiB.
        self.assertLess(peak, 3 * 1024**3)
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue(torch.isfinite(operator.parameter.grad).all())
        print(f"OpenVLA QP2 vocabulary head batch4 forward/backward peak delta: {peak / 1024**3:.3f} GiB")


if __name__ == "__main__":
    unittest.main()
