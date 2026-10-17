# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The flat routed expert at shapes whose gate/up columns have no even split over the 16 readers (the uneven split:
each bank's pair-sets over its two readers, 2 + 1): DeepSeek V4.1-Flash H 5120, I 2304 (72 columns = 24 pair-sets of
3), clamped SwiGLU. y per active expert vs the bfp4-quantized-weight reference, two launches (the second with a
moved arena and other counts: the cache-hit address patch)."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

# (tag, H, I, local experts, capacity m, global experts, activation)
CASES = [
    ("v41_e12", 5120, 2304, 12, 2048, 384, "clamped_silu"),
    ("v41_e24", 5120, 2304, 24, 2048, 384, "clamped_silu"),
]


def _counts(E, m, seed):
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(0, 120, (E,), generator=g)
    c[torch.randperm(E, generator=g)[: E // 4]] = 0
    c[seed % E] = m  # one expert at the full capacity
    c[(seed + 3) % E] = m // 2 + 7
    return [int(v) for v in c]


def _act(g, u, act):
    if act == "clamped_silu":  # DeepSeek V4: silu(min(g, 10)) * clamp(u, -10, 10)
        return torch.nn.functional.silu(g.clamp(max=10)) * u.clamp(-10, 10)
    return torch.nn.functional.silu(g) * u


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_flat_uneven_split(device, case):
    tag, H, I, E, m, NG, act = case
    torch.manual_seed(0)
    weights = [[(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(E)]]
    gids = [[(NG // E) * e + 1 for e in range(E)]]
    op = FlatRoutedExpert(device, weights, m=m, H=H, I=I, gids=gids, n_global=NG, act=act, pin=1)
    plan = op.plan
    logger.info(
        f"{tag}: np {plan['np']} g {plan['g']} mt {plan['mt']} uneven {plan['uneven']} rgs {plan['rgs']} "
        f"gate/up cores {len(plan['gu'])} down {plan['nd']} rdown {plan['rdown']}"
    )
    assert plan["uneven"], f"{tag}: expected the uneven split"
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
    Wq = [[q(w_) for w_ in weights[0][e]] for e in range(E)]
    pinned = []
    for launch, seed in enumerate((1, 2)):
        counts = _counts(E, m, seed)
        offs = [sum(-(-c // 32) * 32 for c in counts[:e]) for e in range(E)]
        rows = offs[-1] + -(-counts[-1] // 32) * 32 + 64
        x = torch.zeros(rows, H)
        c_ = torch.zeros(1, NG, dtype=torch.int32)
        r_ = torch.zeros(1, NG, dtype=torch.int32)
        for e, gid in enumerate(gids[0]):
            x[offs[e] : offs[e] + counts[e]] = torch.randn(counts[e], H)
            c_[0, gid], r_[0, gid] = counts[e], offs[e]
        rm = lambda t, dt: ttnn.from_torch(  # noqa: E731
            t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        y = ttnn.to_torch(op(rm(x, ttnn.bfloat16), rm(c_, ttnn.uint32), rm(r_, ttnn.uint32))).float()
        worst = 1.0
        for e in range(E):
            c = counts[e]
            if not c:
                continue
            o = offs[e]
            sel = list(range(0, c, max(1, c // 48)))[:48] + [c - 1]
            xe = x[o : o + c][sel].to(torch.bfloat16).float()
            Wg, Wu, Wd = Wq[e]
            ref = _act(xe @ Wg, xe @ Wu, act) @ Wd
            ok, pcc = comp_pcc(ref, y[o : o + c][sel], 0.99)
            worst = min(worst, float(pcc))
            assert ok, (tag, launch, e, c, pcc)
        logger.info(
            f"{tag} launch {launch}: min PCC {worst:.5f} over {sum(1 for c in counts if c)} experts, counts {counts}"
        )
        pinned.append(
            ttnn.from_torch(
                torch.zeros(32 * 110 // 16, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        )
