# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3: the 16-bit-DEST SiLU body (Plan.silu_body).

- The default rule: the quartic body (silu_pack::silu_bf16_rows) on one-tile-row plans, the library
  body otherwise, and always the library body with a 32-bit DEST.
- The SiLU in isolation: taps (0, 0, 0, 1) and a zero history make the convolution exact in a 16-bit
  DEST (y = x), so the output is exactly the SFPU body's SiLU of each bf16 input. Compared with the
  correctly rounded bf16 silu(x) from float64, for both bodies on a one-row and a multi-row plan,
  including the clamp extremes (exp(-x) underflow for x > 88, overflow for x < -88).
Shapes are golden-cell shapes (conv1d-001, conv1d-020); only the operand values are chosen.
"""

from __future__ import annotations

import importlib

import pytest
import ttnn

from ttnn.operations.gen_qkv_causal_conv1d_silu import gen_qkv_causal_conv1d_silu, get_last_plan, plan_limits

desc = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu_program_descriptor"
)

LIB, QUARTIC = desc.SILU_LIBRARY, desc.SILU_BF16_QUARTIC
TAP_COUNT = 4
HISTORY_ROWS = TAP_COUNT - 1


@pytest.mark.parametrize(
    "seq_tiles, ch_tiles, fp32, expected",
    [
        (1, 80, False, QUARTIC),  # conv1d-001: one tile-row per core
        (4, 80, False, QUARTIC),  # conv1d-003: 80 cores x one tile-row
        (128, 80, False, QUARTIC),  # conv1d-004: six tile-rows per core, work stealing (Perf 2)
        (64, 64, False, QUARTIC),  # conv1d-020: work stealing
        (4, 80, True, LIB),  # 32-bit DEST: always the library body
    ],
)
def test_default_silu_body(seq_tiles, ch_tiles, fp32, expected):
    plan = desc.select_plan(
        seq_tiles,
        ch_tiles,
        11,
        10,
        desc.REGIME_RM,
        dest_tiles=desc.dest_capacity_tiles(fp32, False),
        fp32_dest_acc_en=fp32,
    )
    assert plan.silu_body == expected
    assert (plan.core_seq_tiles_max == 1 or plan.work_stealing) == (plan.silu_body == QUARTIC) or fp32


def _bf16_ordered(t):
    import torch

    bits = t.to(torch.bfloat16).view(torch.int16).to(torch.int64)
    return torch.where(bits < 0, -(bits + (1 << 15)), bits)


def _silu_operands(seq, channels):
    """bf16 inputs: a normal spread at a few scales plus the clamp extremes; taps (0, 0, 0, 1)."""
    import torch

    gen = torch.Generator().manual_seed(1234)
    x = torch.randn(1, seq, channels, generator=gen) * torch.tensor([0.5, 2.0, 6.0, 20.0]).repeat(channels // 4)
    extremes = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 87.0, 88.5, 90.0, 200.0, -87.0, -88.5, -90.0, -200.0])
    x[0, 0, : extremes.numel()] = extremes
    x = x.to(torch.bfloat16)
    history = torch.zeros(1, HISTORY_ROWS, channels, dtype=torch.bfloat16)
    taps = [torch.zeros(1, 1, channels, dtype=torch.bfloat16) for _ in range(TAP_COUNT)]
    taps[-1] = torch.ones(1, 1, channels, dtype=torch.bfloat16)
    return x, history, taps


def _run_silu(device, x, history, taps, widths):
    import torch

    to_dev = lambda t, layout: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device)
    history_tt = to_dev(history, ttnn.ROW_MAJOR_LAYOUT)
    outputs = gen_qkv_causal_conv1d_silu(
        to_dev(x, ttnn.ROW_MAJOR_LAYOUT),
        history_tt,
        *[to_dev(t, ttnn.TILE_LAYOUT) for t in taps],
        *widths,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
        actual_start=ttnn.from_torch(
            torch.tensor([0], dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        ),
        predecessor_carry=history_tt,
    )
    seq = x.shape[1]
    return torch.cat([ttnn.to_torch(o).reshape(1, seq, -1) for o in outputs], dim=-1)


@pytest.mark.parametrize(
    "shape, widths",
    [((1, 32, 2560), (512, 512, 1536)), ((1, 2048, 2048), (1024, 512, 512))],
    ids=["conv1d-001_one_row", "conv1d-020_multi_row"],
)
def test_silu_in_isolation(device, shape, widths):
    import torch

    _, seq, channels = shape
    x, history, taps = _silu_operands(seq, channels)
    xd = x.double()
    exact = (xd * torch.sigmoid(xd)).to(torch.bfloat16)  # correctly rounded bf16 silu(x)
    finite = xd >= -80.0  # below, both bodies flush exp(-x) overflow to silu = 0 (true |silu| < 1e-33)

    results = {}
    for body in (LIB, QUARTIC):
        with plan_limits(silu_body=body):
            got = _run_silu(device, x, history, taps, widths)
        assert get_last_plan().silu_body == body
        assert torch.isfinite(got.float()).all(), f"body {body}: non-finite output"
        ulp = (_bf16_ordered(got) - _bf16_ordered(exact)).abs()
        results[body] = (got, float((ulp[finite] == 0).float().mean()), int(ulp[finite].max()))
        # exp(-x) underflow: silu(x) = x; overflow: silu(x) = 0
        big = xd >= 90.0
        assert torch.equal(got[big], x[big]), f"body {body}: silu(x) != x for x >= 90"
        assert (got[xd <= -90.0].float().abs() <= 1e-30).all(), f"body {body}: silu(x) != 0 for x <= -90"

    lib_exact, lib_max = results[LIB][1], results[LIB][2]
    q_exact, q_max = results[QUARTIC][1], results[QUARTIC][2]
    print(f"correctly rounded: library {lib_exact:.4f} (max ULP {lib_max}), quartic {q_exact:.4f} (max ULP {q_max})")
    assert q_max <= 1, "quartic body is more than 1 bf16 ULP from the correctly rounded SiLU"
    assert q_exact >= 0.98
    assert q_exact >= lib_exact
