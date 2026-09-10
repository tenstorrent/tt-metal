# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for idea `short_wide_block_width`.

QUESTION: on the focus shape `[1,1,32,16384]` (R=1, C=512), `derive_plan` picks
block_width_tiles=8 -> 64 blocks -> 64 cores -> 512 B reads, exactly ON the
`MIN_BLOCK_ROW_BYTES` floor. Because R==1, `num_blocks_total == num_w_chunks`
identically (there is no row axis to fill), so on THIS geometry the column
width and the core count are the SAME knob: doubling the read transaction size
by doubling `block_width_tiles` necessarily HALVES how many cores get a block.
There is no way to reach a >=1024B read on this shape without giving up cores
-- unless a rule change is found that trades occupancy for bandwidth and wins.

This file forces block_width_tiles directly (bypassing MIN_BLOCK_ROW_BYTES and
PIPELINE_WAVES_PER_CORE entirely -- same technique as
test_tilize_lever_block_width.py) so every rung of the menu is reachable, then
prints the resulting plan for the coordinator. Device-ns measurement is done by
running this file (or the whole-op harness) under `--profile` once per
TILIZE_ABLATE setting (env var, read at process start) -- see the README in
this dir for the exact invocation matrix.

Correctness: every FULL run (no TILIZE_ABLATE) is bit-identity gated
(`torch.equal`) since tilize is a pure re-lay. Ablated runs are perf-only by
construction (wrong output) and carry no correctness assertion, matching
test_tilize_ablation_perf1.py's contract.
"""

import os

import pytest

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

FOCUS_SHAPE = (1, 1, 32, 16384)  # R=1 C=512 -- attention: LOOSE_CASES[0]

# Domain-sweep geometries named in the task (plus the focus shape's own
# neighbours already characterized by the pipeline-waves lever).
DOMAIN_SHAPES = [
    ((1, 1, 32, 32768), "short_wide_wide"),  # C=1024, bw=16 today
    ((1, 1, 64, 12288), "short_wide_2row"),
    ((1, 1, 1, 50304), "ragged_tail_logits"),
    ((1, 1, 1024, 1024), "square_mid"),
    ((1, 1, 2048, 2048), "square_large"),
    ((1, 1, 16384, 32), "tall_narrow"),
    ((1, 1, 2048, 64), "full_width_small"),
]

# Every rung reachable on FOCUS_SHAPE (C=512 is divisible by all of these).
FOCUS_BLOCK_WIDTHS = [2, 4, 8, 16, 32]


def _build(device, shape, forced_bw, monkeypatch):
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    grid = device.compute_with_storage_grid_size()
    if forced_bw is not None:
        # Force the extent directly -- bypasses MIN_BLOCK_ROW_BYTES and
        # PIPELINE_WAVES_PER_CORE so every menu rung is reachable, exactly the
        # technique test_tilize_lever_block_width.py uses.
        monkeypatch.setattr(pd, "RAGGED_COLUMN_TAIL", False)
        monkeypatch.setattr(pd, "_largest_divisor_at_most", lambda n, limit: forced_bw)
    monkeypatch.setattr(pd, "_PLAN_CACHE", {})

    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # [1,1,1,50304] has H=1 (not tile-aligned): tilize refuses an unpadded call
    # there. 0-fill is inert for every already-aligned shape (pad_active stays
    # False), so passing it unconditionally keeps one code path for the sweep.
    tt_output = tilize(tt_input, pad_value=0)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
    cores = len(plan.assignment)
    ablate = os.environ.get("TILIZE_ABLATE", "<none>")
    print(
        f"\n[short_wide_block_width ablate={ablate} bw_forced={forced_bw}] {shape}: "
        f"bw={plan.block_width_tiles} chunks={plan.num_w_chunks} "
        f"row_groups={plan.num_row_groups} blocks={plan.num_blocks_total} "
        f"cores={cores}/{grid.x * grid.y} "
        f"waves/core={plan.tensor_row_blocks * plan.num_w_chunks / max(cores, 1):.2f} "
        f"read={plan.block_row_bytes}B L1/core={plan.l1_per_core_bytes // 1024}KiB "
        f"tail={'yes' if plan.tail_group is not None else 'no'}"
    )
    return plan, tt_input, tt_output, torch_input


@pytest.mark.parametrize("forced_bw", FOCUS_BLOCK_WIDTHS, ids=[f"bw{w}" for w in FOCUS_BLOCK_WIDTHS])
def test_focus_shape_menu(device, forced_bw, monkeypatch):
    """MENU rows 1-5: force bw in {2,4,8,16,32} on the focus shape and measure.

    Run once per TILIZE_ABLATE setting (env var; see README) to separate the
    read stage, the write stage and the whole-op wall. bw=8 is production
    (unforced would also give 8, but we force it too so all five rows are
    measured by the SAME code path).
    """
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    plan, tt_input, tt_output, torch_input = _build(device, FOCUS_SHAPE, forced_bw, monkeypatch)
    assert plan.block_width_tiles == forced_bw
    if not os.environ.get("TILIZE_ABLATE"):
        assert torch.equal(ttnn.to_torch(tt_output), torch_input), f"bw={forced_bw}: not bit-identical"


@pytest.mark.parametrize("shape,label", DOMAIN_SHAPES, ids=[d[1] for d in DOMAIN_SHAPES])
def test_domain_production(device, shape, label, monkeypatch):
    """Production plan (unforced) on each domain shape -- the BEFORE number for
    any rule-change proposal, and the correctness gate for the domain sweep."""
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    plan, tt_input, tt_output, torch_input = _build(device, shape, None, monkeypatch)
    if not os.environ.get("TILIZE_ABLATE"):
        assert torch.equal(ttnn.to_torch(tt_output), torch_input), f"{label}: not bit-identical"


# `short_wide_wide` [1,1,32,32768] is ALSO an R==1 geometry (production already
# picks bw=8, 128 chunks, 64 cores -- confirmed by test_domain_production above),
# so it is the same "widen bw past the occupancy target" experiment as the focus
# shape, at a different C. Forced bw=16 there gives 64 chunks -> 32 cores (half
# occupancy, 1024B reads) -- the direct analogue of FOCUS_SHAPE's bw16 row.
WIDE_TWIN_SHAPE = (1, 1, 32, 32768)
WIDE_TWIN_WIDTHS = [8, 16]  # 8 = production (64 cores, 512B); 16 = widen-past-occupancy probe


@pytest.mark.parametrize("forced_bw", WIDE_TWIN_WIDTHS, ids=[f"bw{w}" for w in WIDE_TWIN_WIDTHS])
def test_wide_twin_widen_probe(device, forced_bw, monkeypatch):
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    plan, tt_input, tt_output, torch_input = _build(device, WIDE_TWIN_SHAPE, forced_bw, monkeypatch)
    assert plan.block_width_tiles == forced_bw
    if not os.environ.get("TILIZE_ABLATE"):
        assert torch.equal(ttnn.to_torch(tt_output), torch_input), f"bw={forced_bw}: not bit-identical"
