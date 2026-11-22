# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Numerical-stability and data-distribution regression tests for mhc_post.

NOT driven by the registry. Runs unconditionally in the full suite (subject
to the op being callable). Tagged @pytest.mark.numerics. All at the
maxed-out precision point (float32 streams and sublayer, fp32 DEST), except
the depth test, which also runs the production bf16 streams once the op
declares them in SUPPORTED.
"""

from __future__ import annotations

import pytest
import ttnn

from ttnn.bringup.mhc_post_ttnn.tests.golden.feature_spec import _case
from ttnn.bringup.mhc_post_ttnn.tests.golden.helpers import run_mhc_post, run_mhc_post_chain
from ttnn.bringup.mhc_post_ttnn import SUPPORTED  # type: ignore

# Device is module-scoped by conftest.py (use_module_device hook).

DISTRIBUTION_SHAPES = [_case((1, 1, 64, 4096)), _case((1, 1, 640, 7168))]
_IDS = [f"T{c[1][-2]}_nC{c[1][-1]}" for c in DISTRIBUTION_SHAPES]

_MAXED = dict(dtype=ttnn.float32, sublayer_dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)


def _run(inputs, device, **extras):
    run_mhc_post(inputs, device=device, extras=extras, **_MAXED)


@pytest.mark.numerics
@pytest.mark.parametrize("inputs", DISTRIBUTION_SHAPES, ids=_IDS)
def test_small_update_on_large_residual(inputs, device):
    """F ~ 1e-3 against X ~ 1e3: the sublayer's contribution is tiny next to
    the carried residual and must not be lost in the accumulation."""
    _run(inputs, device, seed=42, f_scale=1e-3, x_scale=1e3)


@pytest.mark.numerics
@pytest.mark.parametrize("inputs", DISTRIBUTION_SHAPES, ids=_IDS)
def test_identity_comb(inputs, device):
    """comb = I: every stream carries only itself, X'_j = post_j * F + X_j."""
    _run(inputs, device, seed=42, comb_mode="identity")


@pytest.mark.numerics
@pytest.mark.parametrize("inputs", DISTRIBUTION_SHAPES, ids=_IDS)
def test_cyclic_comb(inputs, device):
    """comb[i][(i+1) % n] = 1: X'_j = post_j * F + X_{j-1}. A kernel that
    applies comb instead of comb^T writes X_{j+1} and fails outright."""
    _run(inputs, device, seed=42, comb_mode="cyclic")


@pytest.mark.numerics
@pytest.mark.parametrize("inputs", DISTRIBUTION_SHAPES, ids=_IDS)
def test_uniform_comb(inputs, device):
    """comb = 1/n everywhere: every stream becomes the stream mean plus its
    own post_j * F."""
    _run(inputs, device, seed=42, comb_mode="uniform")


@pytest.mark.numerics
@pytest.mark.parametrize("inputs", DISTRIBUTION_SHAPES, ids=_IDS)
def test_large_magnitude(inputs, device):
    """F and X ~ 1e3 — range handling of the multiply-add."""
    _run(inputs, device, seed=42, f_scale=1e3, x_scale=1e3)


# DeepSeek-V4: 61 layers x 2 mHC wraps; X' is fed back as the next wrap's X.
DEPTH = 122
# (|norm drift| limit, rel err limit) after 122 wraps vs a float64 chain.
# Emulated (T 64, C 256, F ~ N(0, 0.1^2)): exact fp32 mixing 9e-9 / 2e-7;
# bf16 streams with exact mixing -2.1e-4 / 7.2e-3; FPU operands truncated to
# tf32: -4.9e-2 / 5.6e-2 (fp32 streams), -2.5e-2 / 3.0e-2 (bf16 streams).
DEPTH_LIMITS = {ttnn.float32: (1e-4, 1e-3), ttnn.bfloat16: (1e-3, 2e-2)}


@pytest.mark.numerics
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
def test_depth_chain(dtype, device):
    """Feed X' back as X for 122 wraps (the residual highway of a 61-layer
    model) and compare with a float64 chain from the same start. A per-call
    bias too small for the per-call gates shows up here as a norm drift."""
    if dtype not in SUPPORTED["dtype"]:
        pytest.skip(f"{dtype} not in SUPPORTED yet")
    drift, rel = run_mhc_post_chain((1, 1, 64, 4 * 256), dtype=dtype, depth=DEPTH, device=device)
    drift_lim, rel_lim = DEPTH_LIMITS[dtype]
    assert abs(drift) <= drift_lim and rel <= rel_lim, (
        f"after {DEPTH} wraps: norm drift {drift:+.3g} (limit {drift_lim}), " f"rel err {rel:.3g} (limit {rel_lim})"
    )
