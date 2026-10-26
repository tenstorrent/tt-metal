# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The flat routed expert at H 5120 with the intermediate widths DeepSeek-V4.1 runs, unpadded (FIX.1, owner: no padding
around a kernel limit): the shared expert's TP slice, 576 (18 tile columns, TP 4: 8x4 / 4x4 / 2x4) and 1152 (36, TP 2:
2x2), and the routed experts' 2304 (72). 18 and 36 have no gate/up split on the 2 / 3 subgrids at H 5120 (DS41-115);
the plan takes the one-subgrid uneven pair split, and the down projection runs a K of 18 / 36 tiles (not a multiple of
the old 8-tile assert in se6_dcompute.cpp, DS41-76). Before FIX.1 18 and 36 did not plan / compile.

Per case: the plan's intermediate is the real width (no padded lanes), y of every active expert vs the float64 FFN
(tests/reference.py ffn) within noise_limit (F139: calib x the reference at the op's real formats, bfp8 / bf16 x and h
tiles, the FPU operand bits at the call's fidelity, vs float64), zero-count experts write nothing, and the call is
deterministic. Weights: random values the block-float dtype holds exactly (check_exact), as stored [N, K]
(weights_nk), clamped SwiGLU, fp32 DEST down (the model's calls)."""

import importlib.util
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.testing import determinism
from models.demos.common.bringup.testing.fork_cases import noise_limit
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

_HERE = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_flat_routed_expert_ttnn_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
_MANT = {"bf8": 7, "bf4": 3}
H = 5120

# (tag, I, local experts, capacity m, weight dtype, precision): glm = LoFi + bfp8 x / h tiles (the model's
# EXPERTS_PRECISION default), full = HiFi2 + bf16 x / h tiles. m 640: an 8x4 chip's rows at chunk 5120; 1024: 2x2's.
CASES = [
    ("se_tp4_i576_glm", 576, 1, 640, "bf8", "glm"),
    ("se_tp4_i576_full", 576, 1, 640, "bf8", "full"),
    ("se_tp4_i576_e3", 576, 3, 640, "bf8", "glm"),
    ("se_tp2_i1152_glm", 1152, 1, 1024, "bf8", "glm"),
    ("se_tp2_i1152_full", 1152, 1, 1024, "bf8", "full"),
    # the routed experts' width (16-column units, nu 3); skipped on a Galaxy chip (DS41-119)
    ("routed_i2304_e4_bf4", 2304, 4, 640, "bf4", "glm"),
]

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="the flat routed expert is Blackhole-only")


def _counts(E, m):
    if E == 1:
        return [m]  # the shared expert: every row, always on
    c = [m, 0, m // 2 + 7, 37][:E]
    return c + [0] * (E - len(c))  # (the last expert active: every row of y belongs to a region)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_flat_widths(device, case):
    tag, I, E, m, wdtype, prec = case
    if I == 2304 and device.compute_with_storage_grid_size().x == 12:
        # (an unchanged plan: test_flat_uneven_split's cases hang there the same way, on the pre-FIX.1 code too)
        pytest.skip("DS41-119: the flat op at I 2304 hangs on a single Galaxy card (also before FIX.1)")
    full = prec == "full"
    fmt = dict(x_bf16=full, h_bf16=full, fidelity="HiFi2" if full else "LoFi")
    fidelity = int((ttnn.MathFidelity.HiFi2 if full else ttnn.MathFidelity.LoFi).value)
    g = torch.Generator().manual_seed(I + E)
    w = lambda *shape: ref.block_round(  # noqa: E731
        torch.randn(*shape, generator=g) * 0.02, _MANT[wdtype], ties="even", clamp=True
    )
    experts = [(w(I, H), w(I, H), w(H, I)) for _ in range(E)]  # (W1 [I, H], W3 [I, H], W2 [H, I])
    NG = E
    fe = FlatRoutedExpert(
        device,
        [experts],
        m=m,
        H=H,
        I=I,
        gids=[list(range(E))],
        n_global=NG,
        wdtype=wdtype,
        act="clamped_silu",
        pin=1,
        x_bf16=full,
        h_bf16=full,
        weights_nk=True,
        check_exact=True,
    )
    plan = dict(fe.plan)
    logger.info(
        f"{tag}: It {I // 32} nsg {plan['nsg']} np {plan['np']} nu {plan['nu']} g {plan['g']} mt {plan['mt']} "
        f"uneven {plan['uneven']} gate/up cores {len(plan['gu'])} down {plan['nd']} rdown {plan['rdown']}"
    )
    if plan["uneven"]:  # (the gate/up pair-sets cover exactly the I / 32 real tile columns)
        assert sum(plan["rgs"]) * plan["np"] == I // 32, (plan["rgs"], plan["np"], I)
    counts = _counts(E, m)
    offs = [sum(-(-c // 32) * 32 for c in counts[:e]) for e in range(E)]
    rows = offs[-1] + -(-counts[-1] // 32) * 32  # (exactly the regions: the op writes no row outside them)
    rm = lambda t, dt: ttnn.from_torch(  # noqa: E731
        t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cd = rm(torch.tensor([counts], dtype=torch.int32), ttnn.uint32)
    rd = rm(torch.tensor([offs], dtype=torch.int32), ttnn.uint32)

    def inputs(seed):
        gx = torch.Generator().manual_seed(seed)
        x = torch.randn(rows, H, generator=gx).to(torch.bfloat16)
        return x, rm(x, ttnn.bfloat16)

    def call(xd):
        return fe(xd, cd, rd, y_bf16=True, down_fp32=True, math_fidelity=fidelity)

    x, xd = inputs(1)
    out = call(xd)
    y = ttnn.to_torch(out).double().reshape(-1, H)
    for e in range(E):
        c, o = counts[e], offs[e]
        if not c:
            continue
        xs = x[o : o + c]

        def fn(xx, e=e):
            return ref.ffn(xx, *(t_.to(xx.dtype) for t_ in experts[e]), "clamped_silu", formats=fmt)

        want = fn(xs.double())
        lim = noise_limit(fn, [xs], ["bf16"], "bf16")
        got = y[o : o + c]
        rel = float((got - want).norm() / want.norm())
        logger.info(
            f"{tag} expert {e} ({c} rows): rel {rel:.3e} (limit {lim['rel']:.3e}, noise {lim['noise_rel']:.3e})"
        )
        assert rel <= lim["rel"], f"{tag} expert {e}: rel L2 {rel:.3e} > limit {lim['rel']:.3e}"
    _, xd_b = inputs(2)
    determinism.assert_deterministic(lambda: call(xd), lambda: call(xd_b), first=out, label=tag)
