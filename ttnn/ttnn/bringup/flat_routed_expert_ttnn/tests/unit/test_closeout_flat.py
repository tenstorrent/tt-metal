# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The flat routed expert alone at the close-out set's Galaxy 8x4 per-chip load (EXPERTS_COMBINE_CLOSEOUT.md), one
chip, the plan's defaults (DRISC only, even splits): every active expert's sampled y rows against the bfp4-quantized
reference, two launches (the second with a moved arena and other counts: the cache-hit address patch). Counts: the
chip's experts' shares of a uniform routing of the chunk (about chunk x top-k / experts each). DSv4.1-Flash is planned
only (test_closeout_plans.py)."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import CHUNKS, GALAXY, MODELS, per_chip
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert, act_ref

CASES = [(m, chunk) for m in MODELS if m != "dsv41_flash" for chunk in CHUNKS if (m, chunk) != ("k27", 16384)]
N_W = 4  # distinct weight sets, round-robin over the experts (a neighbour's weights read by mistake shows)


def _counts(E_glob, K, epc, chunk, seed):
    g = torch.Generator().manual_seed(seed)
    return [
        int(v) for v in torch.binomial(torch.full((epc,), float(chunk)), torch.full((epc,), K / E_glob), generator=g)
    ]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[f"{m}-{c}" for m, c in CASES])
def test_closeout_flat(device, case):
    model, chunk = case
    md, pc = MODELS[model], per_chip(model, GALAXY, chunk)
    H, I, E, NG, act = md["H"], md["I"], pc["epc"], md["E"], md["act"]
    m = chunk  # (every token of the chunk at most once per expert)
    torch.manual_seed(0)
    wsets = [tuple(torch.randn(*sh) * 0.02 for sh in ((H, I), (H, I), (I, H))) for _ in range(N_W)]
    weights = [[wsets[e % N_W] for e in range(E)]]
    gids = [[(NG // E) * e + 1 for e in range(E)]]
    # (layout.rows_hint: the expected rows per expert, chunk x top-k / experts, as the model's caller passes it)
    op = FlatRoutedExpert(
        device, weights, m=m, H=H, I=I, gids=gids, n_global=NG, act=act, pin=1, layout=dict(rows_hint=round(pc["M"]))
    )
    p = op.plan
    logger.info(
        f"{model} chunk {chunk}: E {E} M {pc['M']:.0f} (sub-block {32 * p['mt']} rows); gate/up {len(p['gu'])} x {p['nu'] or 2 * p['np']} "
        f"(units {p['nu']}), down {p['nd']} x {max(p['pcds'])}, drisc {p['drisc']} dual {p['dual']} rdown {p['rdown']}"
    )
    assert p["drisc"] and not p["dual"] and not p["rdown"], "not DRISC only"
    assert p["even_split"], "the gate/up split is not the even one"
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa: E731
    Wq = [tuple(q(w_) for w_ in ws) for ws in wsets]
    pinned = []
    for launch, seed in enumerate((1, 2)):
        counts = _counts(NG, md["K"], E, chunk, seed)
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
            Wg, Wu, Wd = Wq[e % N_W]
            ref = act_ref(xe @ Wg, xe @ Wu, act) @ Wd
            ok, pcc = comp_pcc(ref, y[o : o + c][sel], 0.99)
            worst = min(worst, float(pcc))
            assert ok, (model, chunk, launch, e, c, pcc)
        logger.info(f"{model} chunk {chunk} launch {launch}: min PCC {worst:.5f} over {E} experts, counts {counts}")
        pinned.append(  # the next launch's arena then sits lower: the cache hit must patch every address
            ttnn.from_torch(
                torch.zeros(32 * 110 // 16, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        )
