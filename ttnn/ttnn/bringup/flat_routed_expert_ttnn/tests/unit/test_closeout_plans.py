# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The close-out plans (EXPERTS_COMBINE_CLOSEOUT.md), host only: the flat routed expert's default plan (DRISC only,
even splits) and the level-1 combine's L1 sizing (combine_auto) for every model of the close-out set on the 8x4, 4x4
and 2x4 stages at chunks 5120 and 16384, against the arena a chip has. Needs one real Blackhole for the DRISC plan
(the DRAM cores; a mock cluster plans without them) - nothing runs on it. bh-lb-09's 12 x 10 grid is a Galaxy chip's.

Asserts: every flat plan fits its arena with the expected even split; the combine fits wherever it can, and the cases
it cannot are exactly CANNOT_FIT (fp32 sums for S tokens over at most 16 accumulators do not fit beside one landing
slot per source chip). The combine's ports are an estimate here (4 directions x 2 links on a torus stage, 7 port cells
on the LoudBox 2x4 as placed there); the device runs place them for real. Logs one table row per case (CLOSEOUT_PLAN)."""

import json

import pytest
from loguru import logger

import ttnn
from ttnn.bringup.combine_rf.sparse_combine_rf import combine_auto, fused_arena_reserve
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import CHUNKS, MODELS, STAGES, per_chip
from ttnn.bringup.flat_routed_expert_ttnn.flat_layouts import flat_layout

PORT_CELLS = {"8x4": 8, "4x4": 8, "2x4": 7}
# (gate/up cores, tiles a core, down cores, tile columns a core): the even splits of the close-out plan
SPLITS = {
    "glm53": (64, 2, 32, 6),
    "glm53_flash": (64, 2, 32, 4),
    "k3": (64, 3, 28, 4),
    "k27": (64, 2, 32, 7),
    "m3": (64, 3, 32, 6),
    "dsv41_flash": (48, 3, 32, 5),
}
# (model, stage, chunk) that cannot fit, and why (checked: these and only these)
_SUMS = "combine: fp32 sums of S tokens over <= 16 accumulators leave < one landing slot per source chip"
CANNOT_FIT = {
    ("glm53", "4x4", 16384): _SUMS,
    ("m3", "4x4", 16384): _SUMS,
    ("dsv41_flash", "4x4", 16384): _SUMS,
    ("k27", "2x4", 5120): _SUMS,
    **{(m, "2x4", 16384): _SUMS for m in ("glm53", "glm53_flash", "m3", "dsv41_flash")},
    ("k3", "2x4", 5120): "flat: 112 experts a chip > 64",
    ("k3", "2x4", 16384): "flat: 112 experts a chip > 64",
}


def flat_plan(device, model, epc, m):
    md = MODELS[model]
    return dict(
        ttnn._ttnn.operations.bringup.flat_routed_expert_plan(
            device, md["H"], md["I"], epc, md["E"], m, weights_bf8=False, pin=1, x_bf16=False, h_bf16=False,
            layout=flat_layout(None),
        )  # fmt: skip
    )


def free_cells(plan, grid):
    taken = {tuple(c) for k in ("arena_cores", "gu", "down", "tensix_readers", "relays", "coords") for c in plan[k]}
    return [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]


# K2.7 is being deprecated and stays at chunk 5120
SKIP = {("k27", 16384)}


def plan_row(device, model, stage, chunk, l1_free, n=None, port_cells=None):
    """One case: the flat plan at the stage's per-chip sizing and the combine for n chips (default: the stage's), in
    the arena the fused harness takes: l1_free (the largest free L1 block a core) less fused_arena_reserve."""
    md, pc = MODELS[model], per_chip(model, stage, chunk)
    n = n or pc["chips"]
    row = dict(model=model, stage=stage, chunk=chunk, chips=n, epc=pc["epc"], S=pc["S"], M=round(pc["M"], 1))
    if pc["epc"] > 64:
        return {**row, "fits": False, "why": f"flat: {pc['epc']} experts a chip > 64"}
    p = flat_plan(device, model, pc["epc"], max(256, (chunk + 31) // 32 * 32))
    grid = device.compute_with_storage_grid_size()
    cells = free_cells(p, grid)
    gu_tiles = p["nu"] or 2 * p["np"]
    rb = dict(p["role_bytes"])
    arena_bytes = (l1_free - fused_arena_reserve(p["static_l1_bytes"])) // 2048 * 2048  # (the harness: 32-row units)
    row.update(
        gu=len(p["gu"]), gu_tiles=gu_tiles, down=p["nd"], down_cols=max(p["pcds"]), relays=len(p["relays"]),
        launcher=int(p["drisc"] and not p["dual"]), drisc=p["drisc"], dual=p["dual"], cells=len(cells),
        flat_arena=p["arena_tiles"] * 2048, arena=arena_bytes, static_kb=p["static_l1_bytes"] / 1024,
        even=p["even_split"],
        free_kb={k: (arena_bytes - v) >> 10 for k, v in rb.items() if k != "reader"},
    )  # fmt: skip
    if p["arena_tiles"] * 2048 > arena_bytes:
        return {**row, "fits": False, "why": f"flat arena {p['arena_tiles'] * 2 } KB > {arena_bytes >> 10} KB"}
    ports = PORT_CELLS[stage] if port_cells is None else port_cells
    c = combine_auto(n, pc["S"], md["H"], len(cells), ports, arena_bytes=arena_bytes, piece_max=max(7168, 2 * md["H"]))
    if not c["fits"]:
        return {**row, "fits": False, "why": "combine: " + c["why"]}
    lay = c["layout"]
    row.update(
        fits=True, acc_on_ports=c["acc_on_ports"], ports=ports, accs=c["n_acc"], pool=lay["PSL"], depth=lay["D"],
        inj=lay["QP"], local=lay["QL"], local_keep=c["local_keep"], acc_fp32_kb=lay["acc_fp32"] >> 10,
        modes={k: dict(A=v["n_acc"], D=v["layout"]["D"], inj=v["layout"]["QP"]) for k, v in c["modes"].items()},
    )  # fmt: skip
    row["free_kb"].update(acc=(arena_bytes - lay["acc_used"]) >> 10, port=(arena_bytes - lay["port_used"]) >> 10)
    return row


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_closeout_plans(device):
    grid = device.compute_with_storage_grid_size()
    mv = ttnn.device.get_memory_view(device, ttnn.BufferType.L1)
    l1_free = int(mv.largest_contiguous_bytes_free_per_bank)
    logger.info(f"grid {grid.x} x {grid.y}, {l1_free >> 10} KB of L1 free a core")
    bad = []
    for model in MODELS:
        for stage in STAGES:
            for chunk in CHUNKS:
                if (model, chunk) in SKIP:
                    continue
                row = plan_row(device, model, stage, chunk, l1_free)
                logger.info("CLOSEOUT_PLAN " + json.dumps(row))
                key = (model, stage, chunk)
                if not row["fits"]:
                    if key not in CANNOT_FIT:
                        bad.append(f"{key}: does not fit: {row['why']}")
                    continue
                if key in CANNOT_FIT:
                    bad.append(f"{key}: fits now (listed as {CANNOT_FIT[key]!r})")
                want = SPLITS[model]
                got = (row["gu"], row["gu_tiles"], row["down"], row["down_cols"])
                if got != want:
                    bad.append(f"{key}: split {got}, expected {want}")
                if not row["drisc"] or row["dual"]:
                    bad.append(f"{key}: not DRISC only (drisc {row['drisc']} dual {row['dual']})")
                if not row["even"]:
                    bad.append(f"{key}: the gate/up split is not the even one")
    # the device runs: the LoudBox 2x4 mesh at the Galaxy's per-chip sizing
    for model in MODELS:
        for chunk in CHUNKS:
            if (model, chunk) in SKIP:
                continue
            row = plan_row(device, model, "8x4", chunk, l1_free, n=8, port_cells=PORT_CELLS["2x4"])
            logger.info("CLOSEOUT_PLAN_LB " + json.dumps(row))
            if not row["fits"]:
                bad.append(f"({model}, 2x4 mesh at 8x4 sizing, {chunk}): does not fit: {row['why']}")
    # no even split (I 1056: 66 weight tiles a K row, no count of them a multiple of 16 cores): the plan says so
    p = dict(
        ttnn._ttnn.operations.bringup.flat_routed_expert_plan(
            device, 4096, 1056, 8, 256, 256, weights_bf8=False, pin=1, x_bf16=False, h_bf16=False,
            layout=flat_layout(None),
        )  # fmt: skip
    )
    if p["even_split"]:
        bad.append("I 1056: reported as an even split")
    assert not bad, "\n".join(bad)
