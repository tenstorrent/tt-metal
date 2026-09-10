# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for idea `core_count_vs_waves`.

QUESTION. On the focus shape `[1,1,32,16384]` (R=1, C=512) the plan is
bw=8 -> 64 blocks -> 64/64 cores -> exactly ONE tile-row wave per core, so a
core reads, computes, then writes strictly in series and the whole grid
alternates a read-only phase with a write-only phase. The op's ONLY existing
lever for a second wave is `PIPELINE_WAVES_PER_CORE`, which buys it from the
COLUMN axis (halve bw) and therefore halves the 512 B read transaction —
`MIN_BLOCK_ROW_BYTES` says that trade does not pay here.

There is a second axis the op has never used: the CORE COUNT. At 32 cores the
same 64 blocks of the SAME width land 2-per-core, so the pipeline gets its
second stage with the read transaction untouched at 512 B. The premise is that
the grid is massively over-provisioned per core on this shape (~3.0 GB/s per
core against a measured ~17.9 GB/s single-core ceiling for 2 KB bf16
transactions), so occupancy may be nearly free while the pipeline depth is gain.

HOW THE LEVER IS PULLED. The core budget enters the op at exactly one point:
`create_program_descriptor` does `grid = device.compute_with_storage_grid_size()`
and hands it to `derive_plan`. So this bench monkeypatches the module-global
`derive_plan` with a shim that substitutes a smaller `CoreCoord` grid;
everything downstream (block solve, `split_work_to_cores`, CB descriptors, all
three kernels) is the op's own code, unmodified. No kernel is rewritten here —
the "part" under test IS the work split, so the honest baseline variant is the
op itself at 64/64 cores.

Holding the block width at 8 across the menu takes two more knobs:
`FAST_TILIZE_WIDTH_CAP = 8` (caps `w_cap`, hence the solved extent) and
`PIPELINE_WAVES_PER_CORE = 1` (so the column axis is not asked for waves too).
Without them a smaller grid makes the solve RE-WIDEN the block to refill it,
which would confound the core-count axis with the read-size axis. The
`natural` test below measures that unforced behaviour separately.

Precision contract untouched: no dtype, no `fp32_dest_acc_en`, no
`math_fidelity`, no `math_approx_mode`, no `dst_full_sync_en` is changed by any
variant — the only thing that moves is how many cores the same blocks sit on.

Correctness gate is BIT IDENTITY (`torch.equal`); tilize does no arithmetic.

Run for numbers (device kernel ns, one CSV row per test, in parametrize order):
    scripts/run_safe_pytest.sh --profile --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/core_count_vs_waves/test_core_count_vs_waves.py
A `--profile` run masks the exit code (Tracy wrapper), so correctness is taken
from a plain (non-profile) run of the same file.

===========================================================================
VERDICT: REGRESSION. Buying the wave from the core-count axis LOSES.
Wormhole B0 n150, 8x8 = 64/64 cores, 1 GHz, 12 DRAM banks. bf16 -> bf16,
ROW_MAJOR interleaved DRAM -> TILE interleaved DRAM. Device kernel ns,
3 fresh reps per rung, median (all 30 dispatches bit-identical):

  cores  waves/core  read     ns (3 reps)              median  GB/s   GB/s/core
    64       1       512 B    12368 12321 13001        12368   169.6    2.65   <- BASELINE
    48      1-2      512 B    13344 12636 12954        12954   161.9    3.37
    32       2       512 B    13061 12904 13357        13061   160.6    5.02
    24      2-3      512 B    14336 13719 14111        14111   148.6    6.19
    16       4       512 B    13751 13307 13346        13346   157.1    9.82
     8       8       512 B    18210 17844 18082        18082   116.0   14.50

Every rung with fewer cores is SLOWER than the full grid. The premise (the
grid is over-provisioned per core, so occupancy is nearly free) is HALF
right and that is exactly why the idea fails:

  * halving the grid costs only ~5%, so the per-core rate really does
    almost double (2.65 -> 5.02 GB/s/core) — occupancy is cheap;
  * but the pipeline depth it buys is worth ~0. The zone report
    (`perf_experiments/zone_report.py`) shows the overlap IS harvested and
    the tail IS flattened, and the wall does not move:

      64 cores: BRISC span mean  9566 max 11885 (tail 1.24x)
                writer_wait_out  6094 = 63.7% of span, writer_issue 1924
                NCRISC span mean 5428 max  7831 (tail 1.44x)
      32 cores: BRISC span mean 11383 max 12621 (tail 1.11x)
                writer_wait_out  4764 = 41.9% of span, writer_issue 4051
                NCRISC span mean 8065 max  8505 (tail 1.05x)

    `writer_wait_out` falls 6094 -> 4764 ns and 63.7% -> 41.9% of the BRISC
    span, and both mean-vs-max tails collapse (1.24 -> 1.11, 1.44 -> 1.05).
    So the overlap and the tail flattening are REAL. They are just not
    where the time is: `writer_issue` scales 1924 -> 4051 ns, i.e. LINEARLY
    with blocks per core — the writer's store issue gets no cheaper on a
    less contended NoC1 — while the reader gains only 1.34x per block
    (5301 -> 3943 ns/block). Two waves therefore ADD more serial per-core
    writer time than the recovered idle time pays for.

    At 8 cores the per-core rate (14.5 GB/s) approaches the measured
    ~17.9 GB/s single-core ceiling and the curve falls off a cliff — the
    over-provisioning premise stops holding somewhere between 16 and 8
    cores, which is the only place it was ever load-bearing.

UNFORCED SOLVE (`test_menu_natural`): handing derive_plan a smaller grid
without pinning the width lets it re-widen the block to refill the grid.
Medians: 64c 12878 (bw 8); 32c 13146 (bw 8); 16c 13651 (bw 8);
8c 13483 (bw 16, 1024 B read) — the wider read recovers 18082 -> 13483 at
8 cores (1.34x), which is the read-size axis, not this idea. Still slower
than the full grid.

DOMAIN (`test_domain_*`, 3 reps each, all 42 dispatches bit-identical).
The candidate rule (`rule_cores`: halve the grid whenever a core owns
exactly one wave) is CORRECT everywhere and SLOWER wherever it fires:

  shape                 base ns (median)  rule ns (median)  cores  ratio
  [1,1,32,16384] focus       12651             13188        64->32  0.96x
  [1,1,2048,64]              5256              6501         64->32  0.81x
  [1,1,32,2048]              3795              5276         64->32  0.72x
  [1,1,32,32768]             23795             23607        inert   1.01x
  [1,1,1024,1024]            22886             23420        inert   0.98x
  [1,1,2048,2048]            87918             87696        inert   1.00x
  [1,1,16384,32]             17657             17782        inert   0.99x

The narrow-read geometries lose the most (0.72x on a 64 B read), which is
the same mechanism: the per-core issue cost, not the phase serialization,
is what those shapes are limited by.
===========================================================================
"""

import importlib
import math

import pytest

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

# `ttnn.operations.tilize.tilize` as an ATTRIBUTE is the op function (the package
# re-exports it, shadowing the submodule), so the module object is fetched from
# sys.modules instead — `_output_tensor_spec` lives on the module.
tilize_mod = importlib.import_module("ttnn.operations.tilize.tilize")

FOCUS_SHAPE = (1, 1, 32, 16384)  # R=1 C=512 — attention: LOOSE_CASES[0]

# Every rung is a whole number of 8-wide grid ROWS, so the used cores are the
# first N of the same row-wise order the op already assigns in (`row_wise=True`
# in derive_plan): the reduced grid is a PREFIX of the full one, not a
# differently-shaped region.
CORE_RUNGS = [64, 48, 32, 24, 16, 8]
REPS = 3

DOMAIN_SHAPES = [
    ((1, 1, 32, 16384), "focus"),
    ((1, 1, 32, 32768), "short_wide_wide"),
    ((1, 1, 1024, 1024), "square_mid"),
    ((1, 1, 2048, 2048), "square_large"),
    ((1, 1, 16384, 32), "tall_narrow"),
    ((1, 1, 2048, 64), "full_width_small"),
    ((1, 1, 32, 2048), "width_chunked_small"),
]

# The candidate RULE: a core owning exactly ONE wave (one tile-row of one
# block) has no pipeline at all, so hand the same blocks to fewer cores until it
# owns `RULE_TARGET_WAVES`. Stated only in quantities `derive_plan` already has.
RULE_TARGET_WAVES = 2


def _grid_for(cores):
    assert cores % 8 == 0, "rungs are whole 8-wide grid rows"
    return ttnn.CoreCoord(8, cores // 8)


def _patch_grid(monkeypatch, cores, hold_bw=None):
    """Substitute the core budget `derive_plan` solves against.

    `hold_bw` pins `FAST_TILIZE_WIDTH_CAP` (and flattens the column-axis wave
    ladder) so the solved block width — hence the read transaction — cannot
    move with the core count.
    """
    real = pd.derive_plan
    grid = _grid_for(cores)

    def shim(input_tensor, output_tensor, *, low_l1, grid=None, pad_value=None):
        return real(input_tensor, output_tensor, low_l1=low_l1, grid=_grid_for(cores), pad_value=pad_value)

    if hold_bw is not None:
        monkeypatch.setattr(pd, "FAST_TILIZE_WIDTH_CAP", hold_bw)
        monkeypatch.setattr(pd, "PIPELINE_WAVES_PER_CORE", 1)
    monkeypatch.setattr(pd, "_PLAN_CACHE", {})
    monkeypatch.setattr(pd, "derive_plan", shim)
    return grid


def waves_per_core(plan):
    cores = max(1, plan.num_cores_used)
    rows_per_block = plan.tensor_row_blocks // max(1, plan.num_row_groups)
    return rows_per_block * math.ceil(plan.num_blocks_total / cores)


def rule_cores(plan):
    """Cores the candidate RULE would use, given the full-grid plan. Returns the
    plan's own core count (inert) wherever a core already owns >1 wave."""
    cores = plan.num_cores_used
    if waves_per_core(plan) >= RULE_TARGET_WAVES:
        return cores
    want = max(1, plan.num_blocks_total // RULE_TARGET_WAVES)
    return min(cores, max(8, (want // 8) * 8))


def _describe(tag, shape, plan, budget):
    cores = plan.num_cores_used
    rows_per_block = plan.tensor_row_blocks // max(1, plan.num_row_groups)
    bytes_moved = 2 * 2 * math.prod(shape)  # bf16 in + bf16 out
    print(
        f"\n[ccvw {tag}] {shape}: budget={budget} cores={cores} "
        f"bw={plan.block_width_tiles} chunks={plan.num_w_chunks} rowgroups={plan.num_row_groups} "
        f"blocks={plan.num_blocks_total} blocks/core={plan.num_blocks_total / max(1, cores):.2f} "
        f"rows/block={rows_per_block} waves/core={waves_per_core(plan)} "
        f"read={plan.block_row_bytes}B L1/core={plan.l1_per_core_bytes // 1024}KiB "
        f"traffic={bytes_moved / 1e6:.3f}MB"
    )


def _run(device, shape, tag, budget):
    # torch is imported inside the function: scripts/validate_no_global_torch_imports.py
    # forbids a module-level torch import anywhere under ttnn/ttnn/.
    import torch

    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=device.compute_with_storage_grid_size())
    _describe(tag, shape, plan, budget)
    assert torch.equal(ttnn.to_torch(tt_output), torch_input), f"{tag}: not bit-identical"
    return plan


def _reference_plan(device, shape):
    """The plan the op SHIPS on `shape`, derived with NO dispatch (so the perf
    CSV keeps exactly one row per test)."""
    import torch

    grid = device.compute_with_storage_grid_size()
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    y = ttnn.allocate_tensor_on_device(
        tilize_mod._output_tensor_spec(x.shape, x.dtype, x.memory_config(), ttnn.Tile([32, 32]), None, core_grid),
        device,
    )
    plan = pd.derive_plan(x, y, low_l1=False, grid=grid)
    ttnn.deallocate(y)
    ttnn.deallocate(x)
    return plan


# --- MENU: the focus shape, block width HELD at 8, core budget swept --------
MENU_PARAMS = [(rep, cores) for rep in range(REPS) for cores in CORE_RUNGS]


@pytest.mark.parametrize("rep,cores", MENU_PARAMS, ids=[f"r{r}_c{c}" for r, c in MENU_PARAMS])
def test_menu_focus(device, rep, cores, monkeypatch):
    """One CSV row per (rep, core budget). bw is pinned to 8 for every rung, so
    waves/core == 64 / cores while the 512 B read transaction never moves.
    `cores == 64` IS the shipped baseline (the pins are inert there)."""
    _patch_grid(monkeypatch, cores, hold_bw=8)
    plan = _run(device, FOCUS_SHAPE, f"menu rep{rep} c{cores}", cores)
    assert plan.block_width_tiles == 8, "the menu holds the read transaction at 512 B"
    assert plan.num_blocks_total == 64
    assert plan.num_cores_used == cores


# --- what the UNFORCED solve does with a smaller grid ----------------------
NATURAL_RUNGS = [64, 32, 16, 8]
NATURAL_PARAMS = [(rep, cores) for rep in range(REPS) for cores in NATURAL_RUNGS]


@pytest.mark.parametrize("rep,cores", NATURAL_PARAMS, ids=[f"n{r}_c{c}" for r, c in NATURAL_PARAMS])
def test_menu_natural(device, rep, cores, monkeypatch):
    """Same core budget, but the op's own width solve is left free — it
    RE-WIDENS the block to refill the smaller grid, a different point on the
    (read size x waves) plane, measured separately so the menu stays clean."""
    _patch_grid(monkeypatch, cores, hold_bw=None)
    _run(device, FOCUS_SHAPE, f"natural rep{rep} c{cores}", cores)


# --- DOMAIN: the candidate rule, everywhere -------------------------------
@pytest.mark.parametrize("shape,label", DOMAIN_SHAPES, ids=[d[1] for d in DOMAIN_SHAPES])
@pytest.mark.parametrize("rep", range(REPS), ids=[f"rep{r}" for r in range(REPS)])
def test_domain_baseline(device, shape, label, rep, monkeypatch):
    """BASELINE leg of the domain sweep: the op exactly as it ships."""
    monkeypatch.setattr(pd, "_PLAN_CACHE", {})
    _run(device, shape, f"domain-baseline rep{rep} {label}", 64)


@pytest.mark.parametrize("shape,label", DOMAIN_SHAPES, ids=[d[1] for d in DOMAIN_SHAPES])
@pytest.mark.parametrize("rep", range(REPS), ids=[f"rep{r}" for r in range(REPS)])
def test_domain_rule(device, shape, label, rep, monkeypatch):
    """CANDIDATE leg: the shipped block width, spread over `rule_cores(...)`
    cores. Where the rule is inert this is the same plan as the baseline leg
    (the printed `cores=` says which)."""
    ref = _reference_plan(device, shape)
    cores, bw_ref = rule_cores(ref), ref.block_width_tiles
    _patch_grid(monkeypatch, cores, hold_bw=bw_ref)
    plan = _run(device, shape, f"domain-rule rep{rep} {label} ref_bw={bw_ref}", cores)
    assert plan.block_width_tiles == bw_ref, "the rule must not move the read transaction"
