# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Placement variants of g6_clone's small flat plans (perf1, stagger_placement STEP B).

A flat plan (page_stride 1: flat_page_range or logical_rows_partial_tile) with fewer pages than grid
cores runs one page per core on the first cores of a row-first fill, so it spans the whole grid width.
Here it moves into a block of k columns x r rows taken from the columns nearest the interior DRAM
column, each core getting `ppc` consecutive pages, filled row-first inside the block.

Why the interior DRAM column (STEP A, clock_offset_map.json): the per-tile wall clocks of this chip are
offset by 0/24/155/296/444 cycles, symmetric about the chip's centre line, which is the interior DRAM
column (Blackhole: DRAM columns at physical x0 and x9). The two worker columns flanking it read within
24 cycles of each other; the outer columns up to 444 later. The true (corrected) kernel starts differ
by at most ~230 cycles over the grid. DEVICE KERNEL DURATION is first start to last end over raw
timestamps, so a plan spread over the grid width carries up to ~444 cycles of clock offset in the
reported number that no core actually spends.

The landmark comes from the API (ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment: the
NoC0 anchor of a bank is the worker just east of its DRAM column), so it follows harvesting.

PLACE env forms (the lane bench reads them):
    ""             the op as it is (row-first, one page per core)
    "k,r,ppc"      fixed block: k centre columns x r rows, ppc pages per core
    "k,r,ppc,c"    the same block filled column-major (consecutive pages down a column)
    "rule"         rule_block(): k, r, ppc derived from the plan (the candidate rule)
"""

from __future__ import annotations

import math

import ttnn


def interior_dram_center(device, grid) -> float:
    """Logical x of the chip's centre line: half a column west of the westmost interior DRAM anchor
    (the worker just east of the interior DRAM column), else the grid's middle."""
    anchors = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    interior = sorted({a.x for a in anchors if 0 < a.x < grid.x})
    return interior[0] - 0.5 if interior else (grid.x - 1) / 2


def center_out_columns(device, grid) -> list:
    """Logical columns ordered by distance from the centre line, ties east first."""
    c = interior_dram_center(device, grid)
    return sorted(range(grid.x), key=lambda x: (abs(x - c), -x))


def block_cores(device, grid, k: int, r: int, col_major: bool = False) -> list:
    """The k centre columns (a contiguous logical range) x rows 0..r-1, row-major (column-major with
    col_major: consecutive pages stack down a column instead of across the band)."""
    cols = sorted(center_out_columns(device, grid)[:k])
    if cols != list(range(cols[0], cols[0] + k)):
        raise RuntimeError(f"stagger_placement: centre columns {cols} are not contiguous")
    if col_major:
        return [(x, y) for x in cols for y in range(r)]
    return [(x, y) for y in range(r) for x in cols]


def assign(cores, tensor_pages: int, ppc: int) -> list:
    """(x, y, start_page, num_pages): consecutive ranges of ppc pages (the last one short) over the
    first ceil(pages / ppc) cores of `cores`."""
    n_cores = -(-tensor_pages // ppc)
    if n_cores > len(cores):
        raise RuntimeError(f"stagger_placement: {n_cores} cores needed, block has {len(cores)}")
    out, start = [], 0
    for x, y in cores[:n_cores]:
        count = min(ppc, tensor_pages - start)
        out.append((x, y, start, count))
        start += count
    return out


def rule_block(device, grid, tensor_pages: int, num_banks: int):
    """The candidate rule: (k, r, ppc) of a small flat plan, or None to keep the op's split.

    One page per core in a band of k = min(pages, num_banks, grid.x) columns nearest the interior DRAM
    column, filled row-first over r = ceil(pages / k) rows from row 0. With k = num_banks, page p sits in
    bank p mod num_banks and column (p mod k) of the band, so each column's cores read and write one
    bank only; the band also leaves out the grid's outer columns (latest real starts, largest clock
    offsets). A plan the band cannot hold (r > grid.y) keeps the op's split."""
    k = max(1, min(tensor_pages, num_banks, grid.x))
    r = -(-tensor_pages // k)
    if r > grid.y:
        return None
    return k, r, 1


def parse_place(place: str):
    if place in ("", "rowfirst"):
        return None
    if place == "rule":
        return "rule"
    fields = place.split(",")
    k, r, ppc = (int(v) for v in fields[:3])
    return (k, r, ppc, len(fields) > 3 and fields[3] == "c")


# (tensor_pages, page bytes, (k, r, ppc, col_major)) of every plan re-placed in this process.
REPLACED = []


def patched_compute_plan(desc, orig_compute_plan, place):
    """compute_plan with small flat plans re-placed per `place` (parse_place form). Plans that are not
    page_stride 1, or use the whole grid, are returned unchanged."""

    def compute_plan(device, tensor_pages, aligned_page_bytes, geometry=None, num_banks=1):
        plan = orig_compute_plan(device, tensor_pages, aligned_page_bytes, geometry, num_banks)
        grid = device.compute_with_storage_grid_size()
        if place is None or plan["page_stride"] != 1 or tensor_pages >= grid.x * grid.y:
            return plan
        if place == "rule":
            block = rule_block(device, grid, tensor_pages, num_banks)
            if block is None:
                return plan
            k, r, ppc = block
            col_major = False
        else:
            k, r, ppc, col_major = place
        placed = assign(block_cores(device, grid, k, r, col_major), tensor_pages, ppc)
        roles = [(ttnn.CoreCoord(x, y), "reader+writer", start, count) for x, y, start, count in placed]
        covered = sorted(start + i for _, _, start, count in roles for i in range(count))
        if covered != list(range(tensor_pages)):
            raise RuntimeError("stagger_placement: split does not cover every page once")
        core_pages_max = max(c for *_, c in roles)
        block_pages_l1_cap = desc.L1_BUDGET_BYTES // (desc.CB_DEPTH_KNOB * aligned_page_bytes)
        block_pages = min(-(-core_pages_max // desc.BLOCKS_PER_CORE_KNOB), block_pages_l1_cap)
        num_blocks_max = -(-core_pages_max // block_pages)
        plan.update(
            {
                "roles": roles,
                "num_cores": len(roles),
                "core_ranges": desc._row_run_ranges([(c.x, c.y) for c, *_ in roles]),
                "core_pages_g1": None,
                "core_pages_g2": None,
                "core_pages_max": core_pages_max,
                "core_pages_min": min(c for *_, c in roles),
                "block_pages": block_pages,
                "num_blocks_max": num_blocks_max,
                "cb_depth_blocks": min(desc.CB_DEPTH_KNOB, num_blocks_max),
                "route_args": [(desc.DEFAULT_UNICAST_VC, desc.DEFAULT_UNICAST_VC, 0)] * len(roles),
                "placement": (k, r, ppc, col_major),
            }
        )
        REPLACED.append((tensor_pages, aligned_page_bytes, (k, r, ppc, col_major)))
        return plan

    return compute_plan


def install(place_text: str):
    """Patch the op module's compute_plan (g6_clone.py imports it by name)."""
    import importlib

    op_mod = importlib.import_module("ttnn.operations.g6_clone.g6_clone")
    desc = importlib.import_module("ttnn.operations.g6_clone.g6_clone_program_descriptor")
    if place_text == "diff":
        # The exact host diff (host_diff/g6_clone_program_descriptor.py), checked plan by plan against
        # the "rule" monkeypatch: same cores, same pages, same core ranges.
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "g6c_desc_diff", Path(__file__).parent / "host_diff" / "g6_clone_program_descriptor.py"
        )
        diff_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diff_mod)
        # Knob monkeypatches of the tests land on desc; mirror them into the diff module at call time.
        knobs = [name for name in vars(desc) if name.isupper()]
        rule = patched_compute_plan(desc, desc.compute_plan, "rule")

        def compute_plan(device, tensor_pages, aligned_page_bytes, geometry=None, num_banks=1):
            for name in knobs:
                setattr(diff_mod, name, getattr(desc, name))
            diff_mod.SMALL_FLAT_BAND = True
            got = diff_mod.compute_plan(device, tensor_pages, aligned_page_bytes, geometry, num_banks)
            want = rule(device, tensor_pages, aligned_page_bytes, geometry, num_banks)
            key = lambda plan: [((c.x, c.y), s, n) for c, _, s, n in plan["roles"]]
            assert key(got) == key(want), (tensor_pages, key(got), key(want))
            assert got["core_ranges"].num_cores() == want["core_ranges"].num_cores()
            assert len(got["core_ranges"].ranges()) == len(want["core_ranges"].ranges())
            for name in ("block_pages", "cb_depth_blocks", "num_blocks_max", "core_pages_max", "regime"):
                assert got[name] == want[name], (name, got[name], want[name])
            return got

        op_mod.compute_plan = compute_plan
        return "diff"
    place = parse_place(place_text)
    op_mod.compute_plan = patched_compute_plan(desc, desc.compute_plan, place)
    return place
