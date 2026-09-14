# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness gate + timed run for every finalize variant, both DEST modes.

    source python_env/bin/activate
    scripts/run_safe_pytest.sh --run-all ttnn/ttnn/operations/rms_norm/perf_experiments/finalize_col0_sfpu/test_finalize_col0_sfpu.py
    scripts/run_safe_pytest.sh --profile --run-all ttnn/ttnn/operations/rms_norm/perf_experiments/finalize_col0_sfpu/test_finalize_col0_sfpu.py
    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/finalize_col0_sfpu/parse_zones.py --reps 32

Each case dispatches ONE program: DEST[0] is finalized once (checked here, column 0 of all 32 rows vs a
float64 reference of the exact values in the tile); DEST[1] is finalized `reps` times inside the zone.
Perf is never asserted; parse_zones.py reads the MATH-thread zone ns after a --profile run.
"""

from __future__ import annotations

import math

import pytest
import ttnn

from .bench import (
    DEST_MODES,
    INIT_MODES,
    TILE,
    VARIANTS,
    run_finalize,
    sharded_memory_config,
)

# Root conftest's `device` fixture, module-scoped (same pattern as tests/ttnn/unit_tests/operations/rms_norm).
pytestmark = pytest.mark.use_module_device

W = 7168
INV_W = 1.0 / W
EPS = 1e-6
REPS = 32

# Focus range from the task: 32 rows with sum in [1e-2, 1e6]. Values are rounded to bf16 so both DEST
# modes hold bit-identical inputs (no unpack truncation ambiguity) and the error is the finalize's alone.
RANGES = {
    "focus_1e-2_1e6": (1e-2, 1e6),
    "wide_1e-5_1e10": (1e-5, 1e10),
}

# Soft gates: bf16 output can carry at most 2^-8 relative from output RNE alone; the op's current lambda
# adds two bf16 DEST truncations of intermediates, so the gate is 2 ulp-ish. fp32: the 23-bit rsqrt is a
# few fp32 ulp. These gate correctness only -- the measured errors are the deliverable.
MAX_REL_ERR = {"dest16": 1.6e-2, "dest32": 2e-6}


def _f32(v: float) -> float:
    import torch

    return torch.tensor(v, dtype=torch.float32).double().item()


def make_input(range_key: str, dest_mode: str, seed: int = 0):
    import torch

    lo, hi = RANGES[range_key]
    g = torch.Generator().manual_seed(seed)
    col0 = torch.exp(torch.empty(TILE, dtype=torch.float64).uniform_(math.log(lo), math.log(hi), generator=g))
    col0[0] = 0.0  # an all-zero row (sum = 0 -> rstd = rsqrt(eps)) is a legal input
    col0 = col0.float().bfloat16().double()  # bf16-representable, exact in both DEST widths
    tile = torch.randn(TILE, TILE, dtype=torch.float32, generator=g) * 3.0  # garbage lanes, incl. negatives
    tile = tile.bfloat16().double()
    tile[:, 0] = col0
    dtype = torch.float32 if dest_mode == "dest32" else torch.bfloat16
    return tile.to(dtype), col0


def reference(col0_f64: torch.Tensor) -> torch.Tensor:
    # The kernel gets inv_w / eps as fp32 bit patterns; the reference uses those exact fp32 values.
    import torch

    return 1.0 / torch.sqrt(col0_f64 * _f32(INV_W) + _f32(EPS))


@pytest.mark.parametrize("dest_mode", DEST_MODES)
@pytest.mark.parametrize("init_mode", INIT_MODES)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("range_key", list(RANGES))
def test_finalize_variant(device, variant, init_mode, dest_mode, range_key):
    import torch

    if range_key != "focus_1e-2_1e6" and init_mode != "pc":
        pytest.skip("range sweep only on the per-call-init flavor (init mode does not change numerics)")
    tile, col0 = make_input(range_key, dest_mode)
    ttnn_dtype = ttnn.float32 if dest_mode == "dest32" else ttnn.bfloat16
    x = ttnn.from_torch(
        tile, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_memory_config()
    )
    # One timed dispatch per case; the `reps` loop runs on DEST[1], the checked result is DEST[0].
    out = run_finalize(x, variant=variant, init_mode=init_mode, dest_mode=dest_mode, inv_w=INV_W, eps=EPS, reps=REPS)
    got = ttnn.to_torch(out).double()[:, 0]
    ref = reference(col0)
    assert torch.isfinite(got).all(), f"{variant}/{init_mode}/{dest_mode}: non-finite in column 0: {got}"
    rel = ((got - ref).abs() / ref.abs()).max().item()
    print(f"\nRELERR variant={variant} init={init_mode} dest={dest_mode} range={range_key} max_rel_err={rel:.3e}")
    assert rel <= MAX_REL_ERR[dest_mode], f"{variant}/{init_mode}/{dest_mode}/{range_key}: max rel err {rel:.3e}"


# Domain sweep: math_approx_mode=True is a config the op accepts. Both variants then use the 10-bit sqrt
# algorithm (APPROX flows into the same _calculate_sqrt_body_ the stock rsqrt_tile uses); the candidate must
# match the baseline's accuracy class. Error is reported; the gate is the approx-class bound.
@pytest.mark.parametrize("dest_mode", DEST_MODES)
@pytest.mark.parametrize("variant", ["baseline", "col0_skip_fused"])
def test_finalize_variant_approx(device, variant, dest_mode):
    import torch

    tile, col0 = make_input("focus_1e-2_1e6", dest_mode)
    ttnn_dtype = ttnn.float32 if dest_mode == "dest32" else ttnn.bfloat16
    x = ttnn.from_torch(
        tile, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_memory_config()
    )
    out = run_finalize(
        x, variant=variant, init_mode="pc", dest_mode=dest_mode, inv_w=INV_W, eps=EPS, reps=REPS, approx=True
    )
    got = ttnn.to_torch(out).double()[:, 0]
    ref = reference(col0)
    assert torch.isfinite(got).all(), f"{variant}/{dest_mode}/approx: non-finite in column 0: {got}"
    rel = ((got - ref).abs() / ref.abs()).max().item()
    print(f"\nRELERR variant={variant} init=pc dest={dest_mode} range=focus_1e-2_1e6 approx=True max_rel_err={rel:.3e}")
    assert rel <= 1.6e-2, f"{variant}/{dest_mode}/approx: max rel err {rel:.3e}"
