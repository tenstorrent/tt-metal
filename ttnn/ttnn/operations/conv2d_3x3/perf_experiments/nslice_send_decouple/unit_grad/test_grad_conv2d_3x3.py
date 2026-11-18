# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for conv2d_3x3 at the Phase 0 cell: bfloat16 ROW_MAJOR activation in DRAM
interleaved, bfloat16 weights, fp32 DEST accumulation, tile-aligned C_in / C_out, H*W and
H_out*W_out multiples of 32, stride 1, dilation 1, padding 0 or 1.

The op contract fixes batch at 1, so there is no multi-batch case; the 64-tile-row case is the
multi-core stand-in. The deep-K case makes one kernel row of weights (48 x 8 bf16 tiles,
double-buffered) exceed L1, which forces K-blocks smaller than one kernel row.
"""

import pytest


class _LazyTorch:
    def __getattr__(self, name):
        import torch as _torch

        return getattr(_torch, name)


torch = _LazyTorch()
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.conv2d_3x3.perf_experiments.nslice_send_decouple.graduated.conv2d_3x3 import (
    conv2d_3x3,
    default_compute_kernel_config,
    prepare_conv2d_bias,
    prepare_conv2d_weights,
)

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}

# ((H, W, C_in), C_out, padding)
SHAPES = [
    ((4, 8, 32), 32, 1),  # single output tile (M = 32)
    ((16, 16, 64), 128, 1),  # multi-tile M and N
    ((24, 40, 32), 64, 1),  # non-square
    ((16, 34, 32), 32, 0),  # valid padding: W_out != W
    ((32, 64, 128), 128, 1),  # 64 output tile-rows over many cores
    ((8, 16, 512), 256, 1),  # deep K (4608): K-blocks below one kernel row
]


def _shape_id(case):
    (H, W, c_in), c_out, padding = case
    return f"H{H}_W{W}_Ci{c_in}_Co{c_out}_p{padding}"


def _out_hw(H, W, padding, stride=1, dilation=1):
    return (
        (H + 2 * padding - 2 * dilation - 1) // stride + 1,
        (W + 2 * padding - 2 * dilation - 1) // stride + 1,
    )


def torch_conv2d_3x3(x_flat, weight, bias, *, H, W, padding):
    """Reference on the flattened channels-last contract: [1, 1, H*W, C_in] -> [1, 1, H_out*W_out, C_out]."""
    c_in = x_flat.shape[-1]
    x = x_flat.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    out = torch.nn.functional.conv2d(
        x, weight.float(), None if bias is None else bias.float(), stride=1, padding=padding, dilation=1
    )
    return out.permute(0, 2, 3, 1).reshape(1, 1, -1, out.shape[1])


def _prepare_kwargs(device, padding):
    return dict(
        stride=1,
        padding=padding,
        dilation=1,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weight_dtype=ttnn.bfloat16,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )


def _make_case(case, with_bias, device):
    (H, W, c_in), c_out, padding = case
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16) if with_bias else None
    kwargs = _prepare_kwargs(device, padding)
    tt_weight = prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = prepare_conv2d_bias(torch_bias, **kwargs) if with_bias else None
    return torch_weight, torch_bias, tt_weight, tt_bias


def _make_input(H, W, c_in, device):
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_input, tt_input


def _check(tt_output, expected, *, H_out, W_out, c_out):
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.dtype == ttnn.bfloat16
    assert list(tt_output.shape) == [1, 1, H_out * W_out, c_out]
    actual = ttnn.to_torch(tt_output).float()
    assert_with_pcc(expected, actual, PCC[ttnn.bfloat16])


@pytest.mark.parametrize("with_bias", [True, False], ids=["with_bias", "no_bias"])
@pytest.mark.parametrize("case", SHAPES, ids=[_shape_id(c) for c in SHAPES])
def test_conv2d_3x3(case, with_bias, device):
    torch.manual_seed(42)
    (H, W, c_in), c_out, padding = case
    torch_weight, torch_bias, tt_weight, tt_bias = _make_case(case, with_bias, device)
    torch_input, tt_input = _make_input(H, W, c_in, device)

    tt_output = conv2d_3x3(tt_input, tt_weight, input_height=H, input_width=W, bias=tt_bias, padding=padding)

    H_out, W_out = _out_hw(H, W, padding)
    expected = torch_conv2d_3x3(torch_input, torch_weight, torch_bias, H=H, W=W, padding=padding)
    _check(tt_output, expected, H_out=H_out, W_out=W_out, c_out=c_out)


def test_conv2d_3x3_lofi(device):
    """math_fidelity is not gated: a LoFi config must be accepted and honored."""
    torch.manual_seed(42)
    case = ((16, 16, 64), 128, 1)
    (H, W, c_in), c_out, padding = case
    torch_weight, torch_bias, tt_weight, tt_bias = _make_case(case, True, device)
    torch_input, tt_input = _make_input(H, W, c_in, device)

    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True)
    tt_output = conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        compute_kernel_config=config,
    )

    H_out, W_out = _out_hw(H, W, padding)
    expected = torch_conv2d_3x3(torch_input, torch_weight, torch_bias, H=H, W=W, padding=padding)
    _check(tt_output, expected, H_out=H_out, W_out=W_out, c_out=c_out)


def test_conv2d_3x3_program_cache_hit(device):
    """A second call on freshly allocated tensors (the first call's tensors still alive, so every
    buffer address differs) must reuse the cached program and still be correct."""
    torch.manual_seed(42)
    case = ((24, 40, 32), 64, 1)
    (H, W, c_in), c_out, padding = case
    torch_weight, torch_bias, tt_weight, tt_bias = _make_case(case, True, device)
    H_out, W_out = _out_hw(H, W, padding)

    first_input, first_tt_input = _make_input(H, W, c_in, device)
    first_output = conv2d_3x3(first_tt_input, tt_weight, input_height=H, input_width=W, bias=tt_bias, padding=padding)
    expected = torch_conv2d_3x3(first_input, torch_weight, torch_bias, H=H, W=W, padding=padding)
    _check(first_output, expected, H_out=H_out, W_out=W_out, c_out=c_out)

    num_entries = device.num_program_cache_entries()
    second_input, second_tt_input = _make_input(H, W, c_in, device)
    second_output = conv2d_3x3(second_tt_input, tt_weight, input_height=H, input_width=W, bias=tt_bias, padding=padding)
    assert device.num_program_cache_entries() == num_entries
    expected = torch_conv2d_3x3(second_input, torch_weight, torch_bias, H=H, W=W, padding=padding)
    _check(second_output, expected, H_out=H_out, W_out=W_out, c_out=c_out)


def test_prepared_weight_layout(device):
    """Rows ordered (kh, kw, c) with c fastest; C_in = 32 needs no channel or per-kh-block padding,
    so K_p = 3 * roundup32(3 * 32) = 288 and the logical width stays C_out."""
    torch.manual_seed(42)
    c_in, c_out = 32, 64
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    tt_weight = prepare_conv2d_weights(torch_weight, **_prepare_kwargs(device, 1))

    assert tt_weight.layout == ttnn.TILE_LAYOUT
    assert tt_weight.dtype == ttnn.bfloat16
    assert list(tt_weight.shape) == [1, 1, 288, c_out]
    assert tt_weight.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert tt_weight.memory_config().buffer_type == ttnn.BufferType.DRAM

    expected = torch_weight.permute(2, 3, 1, 0).reshape(9 * c_in, c_out)
    actual = ttnn.to_torch(tt_weight).reshape(9 * c_in, c_out)
    assert torch.equal(actual, expected)

    tt_bias = prepare_conv2d_bias(torch.randn(c_out).to(torch.bfloat16), **_prepare_kwargs(device, 1))
    assert tt_bias.layout == ttnn.TILE_LAYOUT
    assert tt_bias.dtype == ttnn.bfloat16
    assert list(tt_bias.shape) == [1, 1, 1, c_out]


def test_default_compute_kernel_config():
    first = default_compute_kernel_config()
    second = default_compute_kernel_config()
    assert first is not second
    assert first.math_fidelity == ttnn.MathFidelity.HiFi4
    assert first.fp32_dest_acc_en is True
    assert first.math_approx_mode is False


def test_input_shape_contract_violation(device, expect_error):
    """input_height * input_width must equal the activation's row count. Contract errors are
    ValueErrors whose message starts with the op name."""
    torch.manual_seed(42)
    case = ((16, 16, 32), 32, 1)
    (H, W, c_in), _, padding = case
    _, _, tt_weight, _ = _make_case(case, False, device)
    _, tt_input = _make_input(H, W, c_in, device)
    with expect_error(ValueError, "conv2d_3x3"):
        conv2d_3x3(tt_input, tt_weight, input_height=2 * H, input_width=W, padding=padding)


def test_bias_dtype_contract_violation(device, expect_error):
    """The bias must carry the weight dtype (what prepare_conv2d_bias emits): a bf16 bias next to
    bf8_b weights is a contract error, not a mis-sized bias buffer."""
    torch.manual_seed(42)
    H, W, c_in, c_out, padding = 16, 16, 32, 32, 1
    kwargs = _prepare_kwargs(device, padding)
    tt_weight = prepare_conv2d_weights(
        torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16), **{**kwargs, "weight_dtype": ttnn.bfloat8_b}
    )
    tt_bias = prepare_conv2d_bias(torch.randn(c_out).to(torch.bfloat16), **kwargs)
    _, tt_input = _make_input(H, W, c_in, device)
    with expect_error(ValueError, "conv2d_3x3"):
        conv2d_3x3(tt_input, tt_weight, input_height=H, input_width=W, bias=tt_bias, padding=padding)


def test_weight_k_contract_violation(device, expect_error):
    """Weights prepared for C_in = 64 (K_p = 576) must be rejected for a C_in = 32 activation (K_p = 288)."""
    torch.manual_seed(42)
    H, W, padding = 16, 16, 1
    _, _, tt_weight, _ = _make_case(((H, W, 64), 32, padding), False, device)
    _, tt_input = _make_input(H, W, 32, device)
    with expect_error(ValueError, "conv2d_3x3"):
        conv2d_3x3(tt_input, tt_weight, input_height=H, input_width=W, padding=padding)
