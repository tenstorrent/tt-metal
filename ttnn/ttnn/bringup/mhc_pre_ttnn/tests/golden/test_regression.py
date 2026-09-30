# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Numerical-stability and data-distribution regression tests for mhc_pre.

NOT driven by the registry. Runs unconditionally in the full suite (subject
to the op being callable). Tagged @pytest.mark.numerics. Catches regressions
the cross-product matrix in test_golden.py doesn't surface. All at the
maxed-out precision point (float32 X and W, fp32_dest_acc_en=True), except
the depth test, which also runs the production bf16 streams once the op
declares them in SUPPORTED.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.bringup.mhc_pre_ttnn.tests.golden.helpers import make_inputs, pytorch_mhc_pre, run_mhc_pre
from ttnn.bringup.mhc_pre_ttnn import SUPPORTED  # type: ignore

# Device is module-scoped by conftest.py (use_module_device hook).

_MIX = 24  # n = 4

DISTRIBUTION_SHAPES = [
    ((1, 1, 64, 4096), (4096, _MIX)),  # C=1024
    ((1, 1, 640, 7168), (7168, _MIX)),  # C=1792, DeepSeek-V4 per device
]
_IDS = [f"T{x[-2]}_nC{x[-1]}" for x, _ in DISTRIBUTION_SHAPES]

_MAXED = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, weight_dtype=ttnn.float32)


def _run(inputs, device, **extras):
    run_mhc_pre(inputs, device=device, extras=extras, **_MAXED)


@pytest.mark.numerics
@pytest.mark.parametrize("inputs", DISTRIBUTION_SHAPES, ids=_IDS)
def test_small_magnitude_streams(inputs, device):
    """X scaled by 1e-3: mean(X^2) = 1e-6 sits right at norm_eps, so the
    RMSNorm epsilon changes `mixes` materially and must be honoured."""
    _run(inputs, device, seed=42, x_scale=1e-3)


@pytest.mark.numerics
@pytest.mark.parametrize("inputs", DISTRIBUTION_SHAPES, ids=_IDS)
def test_large_magnitude_streams(inputs, device):
    """X scaled by 1e3: the sum of squares over n*C elements is ~1e10 —
    exercises range handling of the RMS accumulation."""
    _run(inputs, device, seed=42, x_scale=1e3)


@pytest.mark.numerics
@pytest.mark.parametrize("inputs", DISTRIBUTION_SHAPES, ids=_IDS)
def test_large_sinkhorn_logits(inputs, device):
    """a_res = 30: comb logits span tens of units, so the row softmax needs
    overflow-safe exp and the Sinkhorn converges slowly (near-permutation
    matrices). The doubly-stochastic gate still applies."""
    _run(inputs, device, seed=42, logit_scale=30.0)


@pytest.mark.numerics
@pytest.mark.parametrize("inputs", DISTRIBUTION_SHAPES, ids=_IDS)
def test_identical_streams(inputs, device):
    """All n streams equal — the first layer's input (mhc_expand of the
    embedding). y must be (sum_i pre_i) * X_0."""
    _run(inputs, device, seed=42, identical_streams=True)


# DeepSeek-V4: 61 layers x 2 mHC wraps composed along the residual path.
DEPTH = 122
# Largest singular value of the depth-122 comb product may fall below the
# fp64 reference product's by at most this much. Emulated: fp32 Sinkhorn
# 1e-5 below; the existing FPU-normalised kernel 0.084 below (0.916 vs
# 0.9999, measured on Blackhole 2026-09-29) — a 9% shrink of the highway.
DEPTH_SIGMA_DROP = 1e-3


@pytest.mark.numerics
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
def test_comb_depth_chain(dtype, device):
    """Compose 122 device combs (fresh W / bias / X per wrap, as each layer
    has its own) and compare the product's largest singular value with the
    fp64 reference product. Per-call gates cannot see a bias this small; the
    product can. Each call is also checked by run_mhc_pre's own gates."""
    if dtype not in SUPPORTED["dtype"]:
        pytest.skip(f"{dtype} not in SUPPORTED yet")
    inputs = ((1, 1, 64, 4 * 256), (4 * 256, _MIX))
    n = 4
    prod_dev = torch.eye(n, dtype=torch.float64).expand(64, n, n).clone()
    prod_ref = prod_dev.clone()
    for layer in range(DEPTH):
        _, _, comb_dev = run_mhc_pre(
            inputs,
            device=device,
            extras={"seed": 1000 + layer},
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            weight_dtype=ttnn.float32,
        )
        x, w, b, scale = make_inputs(*inputs, dtype=dtype, weight_dtype=ttnn.float32, seed=1000 + layer)
        _, _, comb_ref = pytorch_mhc_pre(x.double(), w.double(), b.double(), scale=scale)
        prod_dev = comb_dev.reshape(64, n, n).transpose(-1, -2) @ prod_dev
        prod_ref = comb_ref.double().reshape(64, n, n).transpose(-1, -2) @ prod_ref
    s_dev = torch.linalg.matrix_norm(prod_dev, ord=2)
    s_ref = torch.linalg.matrix_norm(prod_ref, ord=2)
    drop = (s_ref - s_dev).max().item()
    assert drop <= DEPTH_SIGMA_DROP, (
        f"depth-{DEPTH} comb product shrinks: sigma_max device min {s_dev.min():.6f} vs "
        f"reference {s_ref.min():.6f} (drop {drop:.3g} > {DEPTH_SIGMA_DROP})"
    )
