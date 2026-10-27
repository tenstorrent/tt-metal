# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""max_y_writers (DeepSeek-V4.1, 2026-10-27): the plan keeps at most this many y writers (down cores + reader tails),
so an appended combine's L1 hand-off (combine_rf sparse_combine_rf.L1_MAX_WRITERS, 32 landed counters) fits on every
Galaxy chip. A Galaxy chip's DRAM readers sit in columns 0, 1, E, E + 1 by its harvesting: E = 6 leaves V4.1-Flash 32
down cores of 5 tile columns, E = 7 leaves 40 of 4 (and a weight layout of its own). Capped at 32, an E = 7 chip gets
the E = 6 chip's down split, so the same weight-layout cache key.

- test_plan_cap (any Blackhole chip, plan only): 0 (default) and a cap above the plan's writers leave the plan
  unchanged; a cap below them keeps the first down cores of the uncapped plan's order and splits all H columns over
  them; the model's cap (32) gives the down split an E = 6 chip has (32 x 5 columns), on this chip whatever its E (on
  an E = 7 chip: the 40-core plan cut to its first 32).
- test_capped_op (any Blackhole chip): the op with a cap below this chip's down cores (the capped path runs whatever
  the harvesting) vs the bfp4-weight reference, per active expert, as test_flat_uneven_split.

    scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_max_y_writers.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from ttnn.bringup.combine_rf.sparse_combine_rf import L1_MAX_WRITERS
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import WEIGHT_LAYOUT_KEYS, FlatRoutedExpert

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="the flat routed expert is Blackhole-only")

# DeepSeek-V4.1-Flash on 8x4: H 5120, I 2304, 12 local experts of 96 in a column, capacity 5120 (the model's flat_cap)
H, I, E, NG, M = 5120, 2304, 12, 96, 5120


def _plan(device, cap, m=M):
    return dict(ttnn._ttnn.operations.bringup.flat_routed_expert_plan(device, H, I, E, NG, m, max_y_writers=cap))


def _writers(p):
    return p["nd"] + (len(p["rdn"]) if p["rdown"] else 0)


def _east(p):
    return sorted({x for x, _ in p["readers"]})[2]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_plan_cap(device):
    p0 = _plan(device, 0)
    east = _east(p0)
    logger.info(f"east DRAM reader column {east}: uncapped {p0['nd']} down cores, pcds {sorted(set(p0['pcds']))}")
    assert _plan(device, _writers(p0) + 100) == p0, "a cap above the plan's writers changed the plan"
    # a cap below this chip's down cores: the first ones of the uncapped order, every H column on one of them
    cut = _plan(device, p0["nd"] - 4)
    assert _writers(cut) <= p0["nd"] - 4, (cut["nd"], p0["nd"])
    assert cut["down"] == p0["down"][: cut["nd"]], "capped down cores are not the uncapped plan's first ones"
    assert sum(cut["pcds"]) == cut["rem_cols"] == H // 32 and len(cut["col0s"]) == cut["nd"]
    assert cut["col0s"] == [sum(cut["pcds"][:d]) for d in range(cut["nd"])]
    for k in ("readers", "gu", "relays"):  # (only the down cores change)
        assert cut[k] == p0[k], k
    # the model's cap: the down split of an E = 6 chip (32 cores x 5 columns) on any chip, so the same weight layout
    p32 = _plan(device, L1_MAX_WRITERS)
    assert _writers(p32) <= L1_MAX_WRITERS
    assert (p32["nd"], set(p32["pcds"]), p32["col0s"]) == (32, {5}, list(range(0, 160, 5))), (p32["nd"], p32["pcds"])
    assert p32["down"] == p0["down"][:32]
    if east == 6:  # (E = 6: the cap is a no-op, the plan bit for bit the uncapped one)
        assert p32 == p0
    else:  # (E = 7: 40 cores of 4 columns uncapped)
        assert (p0["nd"], set(p0["pcds"])) == (40, {4}), (p0["nd"], p0["pcds"])
    # what the weight cache key reads (flat_expert.py FlatRoutedExpert): the E = 6 chip's values
    lay = {k: p32[k] for k in WEIGHT_LAYOUT_KEYS if k in p32}
    logger.info(f"capped weight layout keys: { {k: v for k, v in lay.items() if k not in ('pcds', 'col0s')} }")
    assert (lay["n_rd"], lay["nd"], lay["rem_cols"], lay["rdown"], lay["nsg"]) == (16, 32, 160, False, 1), lay


def _counts(E_, m, seed):
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(0, 120, (E_,), generator=g)
    c[torch.randperm(E_, generator=g)[: E_ // 4]] = 0
    c[seed % E_] = m
    c[(seed + 3) % E_] = m // 2 + 7
    return [int(v) for v in c]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_capped_op(device):
    m = 2048  # (test_flat_uneven_split's v41 case)
    nd0 = _plan(device, 0, m)["nd"]
    cap = min(L1_MAX_WRITERS, nd0) - 4  # (below this chip's down cores, whatever its harvesting)
    torch.manual_seed(0)
    weights = [[(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(E)]]
    gids = [[(NG // E) * e + 1 for e in range(E)]]
    op = FlatRoutedExpert(
        device, weights, m=m, H=H, I=I, gids=gids, n_global=NG, act="clamped_silu", pin=1, max_y_writers=cap
    )
    assert op.plan["nd"] <= cap < nd0, (op.plan["nd"], cap, nd0)
    logger.info(f"capped at {cap}: {op.plan['nd']} down cores (uncapped {nd0}), pcds {sorted(set(op.plan['pcds']))}")
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa
    Wq = [[q(w_) for w_ in weights[0][e]] for e in range(E)]
    counts = _counts(E, m, 1)
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
        ref = (torch.nn.functional.silu((xe @ Wg).clamp(max=10)) * (xe @ Wu).clamp(-10, 10)) @ Wd
        ok, pcc = comp_pcc(ref, y[o : o + c][sel], 0.99)  # (test_flat_uneven_split's check)
        worst = min(worst, float(pcc))
        assert ok, (e, c, pcc)
    logger.info(f"capped op: min PCC {worst:.5f} over {sum(1 for c in counts if c)} experts, counts {counts}")
