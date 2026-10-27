# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device operand-cost rows for rms_norm_ttnn.

What a weight / a bias / a residual costs on the same shape, interleaved and sharded,
so the next perf round starts from a number instead of a guess.  Run under the
profiler and read DEVICE KERNEL DURATION; no absolute time is asserted:

    scripts/run_safe_pytest.sh --profile \\
        ttnn/ttnn/bringup/rms_norm_ttnn/tests/unit/test_rms_norm_ttnn_perf.py

Off the profiler they check the output shape only.  The seed-parity cases (against
the generated ttnn.operations.rms_norm, which exists only on the branch this op was
generated on) were removed: see CHANGELOG.md.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

# The op under test is the C++ binding; the Python builder is its parity reference
# (test_rms_norm_ttnn_cpp_parity.py).
rms_norm_ttnn = ttnn.bringup.rms_norm

# (rows, hidden) — one decode profile and one prefill profile, both at the
# precision corner the feature spec's perf cases pin (bf16 / HiFi2 / 16-bit
# DEST), plus one wide decode row that forces the cross-core width split.
SHAPES = [(32, 1024), (8192, 1024), (32, 7168)]


def _config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _tensors(device, rows, hidden):
    torch.manual_seed(0)
    x = torch.randn(1, 1, rows, hidden, dtype=torch.float32).to(torch.bfloat16)
    g = torch.randn(1, 1, 1, hidden, dtype=torch.float32).to(torch.bfloat16)
    return (
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


@pytest.mark.parametrize("rows, hidden", SHAPES, ids=[f"{r}x{h}" for r, h in SHAPES])
@pytest.mark.parametrize(
    "mode",
    ["no_gamma", "gamma", "gamma_bias", "residual", "gamma_bias_residual"],
)
def test_operand_cost(device, rows, hidden, mode):
    """What each operand costs on the same shape, as a profiled row.

    Not a gate.  It exists so the operand multiplier is a MEASUREMENT the next
    perf round can act on: the residual doubles every activation DRAM crossing,
    which is what moves a shape between the RESIDENT / ROW_RESIDENT / STREAM
    regimes.  `no_gamma` is the operand-free baseline row.
    """
    x, g = _tensors(device, rows, hidden)
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": _config()}
    if "gamma" in mode:
        kwargs["weight"] = g
    if "bias" in mode:
        torch.manual_seed(4)
        kwargs["bias"] = ttnn.from_torch(
            torch.randn(1, 1, 1, hidden, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    if "residual" in mode:
        torch.manual_seed(5)
        kwargs["residual_input_tensor"] = ttnn.from_torch(
            torch.randn(1, 1, rows, hidden, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    out = rms_norm_ttnn(x, **kwargs)
    assert list(out.shape) == [1, 1, rows, hidden]


# ---------------------------------------------------------------------------
# the sharded operand cases the feature spec carries a reference for
# ---------------------------------------------------------------------------
#
# feature_spec.py's perf group pins two of these by name:
#   (1,1,7168,1024) BLOCK_SHARDED [896,128] on (8,8), gamma_bias_residual,
#       fp32_dest_acc_en=False -> 34569 ns achievable
#   (1,1,32,5120)   WIDTH_SHARDED [32,160]  on (8,4), gamma_bias_residual,
#       fp32_dest_acc_en=True  ->  6555 ns achievable
# Both are recorded here so the operand cost on the CROSS-CORE COMBINE path is a
# measured number rather than an inference from the interleaved rows.  The first
# is also the geometry D32 gives up the D25 pipeline on (a residual makes the
# hoisted pass A write cb_x_sum, whose ring cannot hold a two-block sliding
# window), so it is where that carve-out's cost would show.

from eval.sharding import shard_config  # noqa: E402

_SHARDED_PERF = [
    ((1, 1, 7168, 1024), ([896, 128], (8, 8)), ttnn.TensorMemoryLayout.BLOCK_SHARDED, False),
    ((1, 1, 32, 5120), ([32, 160], (8, 4)), ttnn.TensorMemoryLayout.WIDTH_SHARDED, True),
    # The spec's TIGHTEST sharded reference: 28619 ns at subblock_w = 1 for the
    # weight-only 64-core block shard.
    ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), ttnn.TensorMemoryLayout.BLOCK_SHARDED, False),
    ((1, 1, 32, 7168), ([32, 256], (7, 4)), ttnn.TensorMemoryLayout.WIDTH_SHARDED, False),
    ((1, 1, 32, 1024), ([32, 128], (8, 1)), ttnn.TensorMemoryLayout.WIDTH_SHARDED, False),
]


@pytest.mark.parametrize(
    "shape, shard, memory_layout, fp32_dest",
    _SHARDED_PERF,
    ids=["block_7168x1024", "width_32x5120", "block_8192x1024", "width_32x7168", "width_32x1024"],
)
@pytest.mark.parametrize("mode", ["gamma", "gamma_bias_residual"])
def test_sharded_operand_cost(device, shape, shard, memory_layout, fp32_dest, mode):
    """The combine path's operand cost, at the feature spec's own geometries."""
    torch.manual_seed(0)
    width = shape[-1]
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    mc = shard_config(shard[0], shard[1], memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    cfg = _config()
    cfg.fp32_dest_acc_en = fp32_dest
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": cfg, "memory_config": ttnn_x.memory_config()}

    def _vec(seed):
        torch.manual_seed(seed)
        return ttnn.from_torch(
            torch.randn(1, 1, 1, width, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    if "gamma" in mode:
        kwargs["weight"] = _vec(1)
    if "bias" in mode:
        kwargs["bias"] = _vec(2)
    if "residual" in mode:
        torch.manual_seed(3)
        kwargs["residual_input_tensor"] = ttnn.from_torch(
            torch.randn(shape, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn_x.memory_config(),
        )
    out = rms_norm_ttnn(ttnn_x, **kwargs)
    assert list(out.shape) == list(shape)
