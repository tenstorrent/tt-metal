# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm ProgramDescriptor — CBs, kernels, work distribution, args.

Blocking Model (op_design.md §1).  Every block factor / buffer depth /
core-assignment below is a **named parameter with exactly one source of truth**.
Every CB page count, loop trip count and grid size is *derived* from those
parameters — never restated as a second literal and never taken from a whole-op
dimension (T, B, the full flat width).

Work unit = one **token-block** = ``TOKENS_PER_BLOCK`` tokens of one batch.
``B * ceil(T / TOKENS_PER_BLOCK)`` such blocks are spread over the whole compute
grid with ``row_wise=True`` from phase 1 — there is no single-core phase.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"


# ===========================================================================
# Blocking Model knobs — THE single source of truth (op_design.md §1.1 / §1.2)
# ===========================================================================
#
# Each of these is a tunable.  Nothing downstream re-states its value: CB page
# counts, kernel loop bounds and the grid all derive from them.

# --- block factors ---
TOKENS_PER_BLOCK = 32  # tokens per work unit (= one output tile-row: the re-tile floor)
NORM_CHUNK_TOKENS = 8  # tokens per normalize sub-pass (coarsest that fits L1, §6.2)
GATE_CHUNK_TILES = 64  # output tiles per gate-chain invocation (phase C block factor)

# ---------------------------------------------------------------------------
# Cross-core re-tile group size (Refinement 2) — the parallelism knob
# ---------------------------------------------------------------------------
#
# `TOKENS_PER_BLOCK` is a HARD floor on the work unit: the head-major -> flat
# re-tile fuses exactly one tile-row of tokens into one output tile-row, so
# `B * ceil(T / TOKENS_PER_BLOCK)` is all the parallelism a per-block split can
# ever expose.  At the shapes a decode / short-prefill caller issues that number
# is tiny (T=32 -> 1 block, T=128 -> 4), so up to 109 of 110 cores sat idle while
# the latency stayed at the single-core block time.
#
# This knob splits ONE token-block across `group_cores` cores in TWO axes at
# once, joined by a single row-major all-to-all exchange per block:
#
#   * the NORMALIZE half is split on the TOKEN axis — core `c` of the group owns
#     tokens [c*tokens_per_core, (c+1)*tokens_per_core) and reads only its own
#     slice of `o` (no read amplification);
#   * the GATE half is split on the FLAT-COLUMN axis — core `c` owns output tile
#     columns [c*cols_per_core, (c+1)*cols_per_core), reads only its own slice of
#     `gate` and writes only its own slice of `out` (no read/write amplification);
#   * the two are joined because a token's untilized row-major feature row is
#     CONTIGUOUS in flat-feature order, so column-owner `d`'s share of token row
#     `t` is exactly one contiguous `chunk_bytes` slice of it.  Every core writes
#     its `tokens_per_core` rows' worth of each of the `group_cores` chunks into
#     the owning core's `cb_rm_flat_rows` at row offset `t * chunk_bytes` — the
#     design's lamp #1 (`op_design.md` §1.5, "cross-core re-tile"), which works
#     precisely because `cb_rm_flat_rows` is a plain row-major L1 stripe
#     addressed by token row and a remote writer filling row `t` honours exactly
#     the contract the local untilize already honoured.
#
# `group_cores = 1` is the trivial, byte-identical default path: no exchange, no
# semaphores, `cb_rm_local` IS `cb_rm_flat_rows` (see `_kernel_defines`), and
# every derived quantity collapses to its pre-Refinement-2 value.  So the knob
# costs nothing where it does not pay, which is what lets the dispatch policy
# below fall back to it.
#
# "auto" (the default) is the DISPATCH POLICY the refinement asks for — see
# `_retile_group_cores()` for the objective it minimises and the measurements
# behind it.  An int pins the group size for measurement / regression tests.
RETILE_GROUP_CORES = "auto"

# Ceiling on the "auto" group size.  MEASURED (Blackhole p150, 11x10 grid, median
# of 5 trial-major interleaved trials, `test_onorm_retile_group.py::test_group_trial`,
# spread <= 4.4 %).  Speedup vs `group_cores = 1`:
#
#   shape        cores at best g   g=2     g=4     g=8     g=16    g=32
#   B=1,T=32     1 -> 32          1.93x   3.58x   6.32x  10.62x  16.12x
#   B=1,T=128    4 -> 64          1.89x   3.35x   5.55x   8.59x   7.67x
#   B=1,T=640   20 -> 80          1.68x   2.57x   2.62x   2.53x   2.72x
#   B=8,T=640  110 -> 110         1.22x   1.26x   1.05x   0.94x   0.87x
#
# The curve is monotone up to the point where the grid runs out of groups, and 32
# is where the last cell that still gains (B=1/T=32, 1 token per core) peaks — so
# 32 is the cap, and the `_work` objective is what keeps the shapes that would
# LOSE at 32 (B=8/T=640: 0.87x) away from it.  Lower this to bound the exchange's
# message count if a future arch has a costlier NoC handshake.
MAX_RETILE_GROUP_CORES = 32

# Depth of `cb_rm_local` in normalize-chunk units (`group_cores > 1` only).  The
# staging buffer between compute's untilize (producer) and the writer's scatter
# (consumer); depth 2 lets compute build chunk i+1 while the writer scatters
# chunk i.  Costs `v_tiles * norm_chunk_tokens` pages per unit of depth, and it
# is funded many times over by `cb_rm_flat_rows` / `cb_flat_tiles` shrinking by
# `group_cores`.
RM_LOCAL_DEPTH = 2

# Tiles per noc_async_read / noc_async_write group — ONE barrier per group, so
# this many transfers are in flight at once.  Used by BOTH dataflow halves (the
# reader's `o`/`gate` streams and the writer's output stream read the same CT arg),
# so raising it widens both sides together.
# MEASURED (Blackhole p150, 11x10 grid, tests/.../test_onorm_trials.py, 5-7
# interleaved trials/config, <=1% spread): raising 4 -> 8 together with
# DM_DEPTH 2 -> 4 is 1.164x at 2 cores, 1.163x at 4, 1.108x at 20, 1.061x at 110.
# The win shrinks as cores rise because the op approaches the DRAM roofline
# (~412 GB/s aggregate at B=8/T=640, ~80% of Blackhole peak), which is exactly the
# behaviour op_design.md §1.4 predicts.
DM_BLOCK_TILES = 8

# --- buffer depths (in block-factor units) ---
# DM_DEPTH deepens cb_gate_tiles / cb_out_tiles so the reader can prefetch (and the
# writer drain) a group while compute works on the previous one.  4 is measured
# above; DM_DEPTH=2 at DM_BLOCK_TILES=8 leaves 1.035x on the table.
DM_DEPTH = 4
# O_DEPTH deepens cb_o_tiles. Kept at 2: the reader is NOT the critical path at
# these settings (at the new defaults, B=1/T=640: NCRISC 83.5us vs 88.0us compute
# vs 93.1us kernel), and O_DEPTH=3 measured within noise while costing +128 KB of
# L1. This stays a live knob for a future shape where the reader IS the bound.
O_DEPTH = 2

# --- P7b sigmoid engine (Refinement 1) ---
# `sigmoid(gate)` is the op's whole SFPU volume: `flat_tiles_per_block` tile-ops
# per token-block, all of them issued by ONE TRISC.  This knob names which one.
#
#   "math"   -- sigmoid_tile on the MATH thread, inside the `unary<Sigmoid<>>`
#               eltwise chain.  The Phase-0 shape; the default.
#   "pack"   -- sigmoid_tile_pack on the PACK thread (TRISC2) at the chain's pack
#               stage, via `apply_activation_from_pack`.  Same arithmetic (the
#               same LLK 6-entry LUT), different issuing thread.
#   "ablate" -- MEASUREMENT ONLY.  Drops the sigmoid entirely and copies `gate`
#               straight through, keeping every CB wait/push, every DEST window
#               and every NoC transfer identical.  The output is NUMERICALLY
#               WRONG by construction; this exists so `/perf-measure`'s ablation
#               method can price the SFPU payload against the surrounding
#               scaffolding.  Never a shipping setting -- `validate_knobs()` in
#               onorm.py is what keeps it out of the public entry point.
SIGMOID_ENGINE = "math"

# Tiles per DEST window in the gate phases (P7b sigmoid AND its 1:1 twin P7c, the
# FPU multiply).  This is the `compute_block_size` catalog lever applied to the
# op's hottest loop: the SFPU vector-op count per tile is fixed by the hardware,
# but the *per-DEST-window* cost — tile_regs acquire/commit/wait/release, the SFPU
# call setup, the unpack/math/pack pipeline fill and drain — is paid once per
# window, and Phase 0 opened one window PER TILE (`InputLifecycle::Streaming`
# clamps the chain's block_size to 1).  Raising this coarsens both gate phases
# together; they are paired 1:1 by design, so blocking one alone would just move
# the per-window cost to the other half.
#
# Ceiling is the DEST budget: `DEST_AUTO_LIMIT` is 4 tiles under a 32-bit DEST
# (`fp32_dest_acc_en=True`) and 8 under the 16-bit DEST that Refinement 1b made
# the default.  That ceiling is therefore a function of the CALLER's compute
# config, not a constant — `_dest_tile_limit()` below is its one source of truth
# and the descriptor clamps this request down to it (see `gate_dest_tiles`).
#
# MEASURED (Blackhole p150, median of 5 trial-major interleaved trials,
# tests/.../test_onorm_sigmoid_engine.py): 1 -> 2 -> 4 is monotonic at
# 1.000x -> 1.004x -> 1.006x (B1/T128), 1.000x -> 1.004x -> 1.005x (B1/T640),
# 1.000x -> 1.002x -> 1.005x (B8/T640) — the amortize-a-fixed-cost signature, and
# above the <=0.1% trial spread.  It is SMALL because the phase is dominated by
# the SFPU payload itself, not by per-window overhead: with the sigmoid ablated
# away the same knob is worth 1.016x (B1/T128), which is the per-window cost this
# lever actually removes.  Free in L1 (no CB changes), so 4 was the R1 default.
#
# REFINEMENT 1b re-swept it against the 16-bit DEST that is now the default. The
# 16-bit DEST doubles `DEST_AUTO_LIMIT` to 8, so 8 became reachable for the first
# time — R1b's named "free rider".  MEASURED (median of 5 trial-major interleaved
# trials, at the shipping compute config): the curve TURNS OVER at 4.
#
#   vs d1        d1       d2        d4          d8
#   B=1,T=128  1.0000   1.0054   1.0078*    1.0046      (spread <=0.26%)
#   B=1,T=640  1.0000   1.0031   1.0046*    1.0044      (spread <=0.56%)
#   B=8,T=640  1.0000   0.9891   1.0009*    0.9960      (spread <=5.2%, noisy)
#
# So the free rider does NOT pay: 8 is best-case a tie and worst-case a small
# loss, and 4 stays the measured optimum.  The KNOB keeps its now-doubled
# headroom (the ceiling is derived, not hardcoded) for Refinement 3 to re-sweep
# against the final structure; the shipped VALUE is the measured optimum.
#
# This is the REQUESTED value — `_dest_tile_limit()` clamps it per compute
# config, so a caller who asks for `fp32_dest_acc_en=True` still gets a legal
# window from the same knob.
GATE_DEST_TILES = 4

# Wire codes for SIGMOID_ENGINE.  The kernel branches on the integer; this dict
# is the single source of truth for the mapping (the kernel's ONORM_SIGMOID_*
# defines are emitted from it below, so neither side restates a literal).
_SIGMOID_ENGINE_CODES = {"math": 0, "pack": 1, "ablate": 2}

# Ablation guard.  "ablate" produces WRONG output by construction, so it must be
# opted into explicitly *as well as* selected.  A stray SIGMOID_ENGINE="ablate"
# left behind in a config therefore fails loudly at descriptor-build time rather
# than silently shipping garbage from the public entry point.
ALLOW_SIGMOID_ABLATION = False

# --- hardware tile geometry (not a knob) ---
TILE_H = 32
TILE_W = 32
# DEST_AUTO_LIMIT under half sync (dst_full_sync_en=False), by DEST width —
# dest_helpers.hpp:95-99.  The ceiling on any per-DEST-window tile count.  A
# 32-bit DEST halves the tile budget, so this is a function of the caller's
# `fp32_dest_acc_en`, not a constant.  ONE source of truth for both values.
_DEST_TILE_LIMIT_FP32 = 4  # fp32_dest_acc_en=True  -> 32-bit DEST
_DEST_TILE_LIMIT_16B = 8  # fp32_dest_acc_en=False -> 16-bit DEST (the default)


def _dest_tile_limit(fp32_dest_acc_en: bool) -> int:
    """Tiles that fit one DEST window for this compute config."""
    return _DEST_TILE_LIMIT_FP32 if fp32_dest_acc_en else _DEST_TILE_LIMIT_16B


# L1 headroom held back from the CB budget.  `get_max_worker_l1_unreserved_size()`
# reports the unreserved L1 span, but the statically-allocated CB region starts
# ABOVE a per-program base (kernel binaries, runtime args, profiler buffers), so
# the CB total that the runtime actually validates is larger than the sum of our
# pages.  Measured on Blackhole p150: a 741-page (1517568 B) CB set was reported by the
# runtime as growing to 1628928 B against a max L1 of 1572864 B — i.e. a 111360 B
# base, so the real CB ceiling is 1572864 - 111360 = 1461504 B, which is
# get_max_worker_l1_unreserved_size() - 70656.  Holding back 72 KB makes this
# file's budget assert — which names the knobs to lower, and in what order — fire
# BEFORE the runtime's own bare "beyond max L1 size" throw, without rejecting knob
# settings that genuinely do fit.
_L1_CB_BASE_RESERVE = 72 * 1024


# ===========================================================================
# Circular-buffer indices (semantic names; the number is just the slot)
# ===========================================================================

CB_O_TILES = 0  # reader  -> compute : head-major `o` tiles
CB_GATE_TILES = 1  # reader  -> compute : flat pre-sigmoid `gate` tiles
CB_WEIGHT = 2  # reader  -> compute : RMSNorm scale, held for the whole kernel
CB_SCALER = 8  # reader  -> compute : reduce scaler (1/V), held
CB_OUT_TILES = 16  # compute -> writer  : finished flat output tiles
CB_SUMSQ = 24  # compute -> compute : per-token partial sum of squares
CB_RSTD = 25  # compute -> compute : rsqrt(mean + eps), col-0 valid
CB_RM_LOCAL = 26  # compute -> writer   : ROW-MAJOR rows this core untilized, awaiting scatter
CB_NORMED = 27  # compute -> compute : o * rstd
CB_ONORM = 28  # compute -> compute : o * rstd * weight (untilize input)
CB_RM_FLAT_ROWS = 29  # writer  -> compute : ROW-MAJOR flat feature rows (re-tile working set)
CB_FLAT_TILES = 30  # compute -> compute : flat token-major tiles
CB_GATE_SIG = 31  # compute -> compute : sigmoid(gate), materialised so the FPU feeds off L1

# The kernels do NOT restate these numbers.  They are injected as preprocessor
# defines (below) so this block is the single source of truth for the slot map:
# re-numbering a CB here lands in the kernels automatically, and a kernel that
# references a slot this dict does not define fails to compile instead of
# silently addressing the wrong buffer.
#
# `ONORM_CB_RM_LOCAL` is the one slot whose value depends on the cross-core
# re-tile group size, so it is added by `_kernel_defines()` rather than listed
# here: at `group_cores == 1` it ALIASES `CB_RM_FLAT_ROWS` (compute untilizes
# straight into the stripe it later tilizes — the pre-Refinement-2 path, one CB,
# no copy), and only at `group_cores > 1` does it become its own staging buffer
# whose consumer is the writer's scatter.
_CB_SLOTS = {
    "ONORM_CB_O_TILES": CB_O_TILES,
    "ONORM_CB_GATE_TILES": CB_GATE_TILES,
    "ONORM_CB_WEIGHT": CB_WEIGHT,
    "ONORM_CB_SCALER": CB_SCALER,
    "ONORM_CB_OUT_TILES": CB_OUT_TILES,
    "ONORM_CB_SUMSQ": CB_SUMSQ,
    "ONORM_CB_RSTD": CB_RSTD,
    "ONORM_CB_NORMED": CB_NORMED,
    "ONORM_CB_ONORM": CB_ONORM,
    "ONORM_CB_RM_FLAT_ROWS": CB_RM_FLAT_ROWS,
    "ONORM_CB_FLAT_TILES": CB_FLAT_TILES,
    "ONORM_CB_GATE_SIG": CB_GATE_SIG,
}

# The SIGMOID_ENGINE wire codes travel the same way as the CB slot map: emitted
# as preprocessor defines from this one dict, so the kernel's `if constexpr`
# ladder compares against names, never against re-typed integers.
_SIGMOID_DEFINES = [(f"ONORM_SIGMOID_{name.upper()}", str(code)) for name, code in _SIGMOID_ENGINE_CODES.items()]


def _kernel_defines(group_cores: int):
    """The kernels' preprocessor wire: CB slot map + SIGMOID_ENGINE codes.

    One source of truth for both.  ``ONORM_CB_RM_LOCAL`` is resolved here (see
    ``_CB_SLOTS``): it aliases ``CB_RM_FLAT_ROWS`` in the single-core-per-block
    path and is its own staging CB once the block is split across cores.
    """
    slots = dict(_CB_SLOTS)
    slots["ONORM_CB_RM_LOCAL"] = CB_RM_LOCAL if group_cores > 1 else CB_RM_FLAT_ROWS
    return [(name, str(index)) for name, index in slots.items()] + _SIGMOID_DEFINES


# ===========================================================================
# Semaphores (cross-core re-tile exchange only; `group_cores > 1`)
# ===========================================================================
#
# Two MONOTONE counters per core, both created with initial value 0 on the host
# and never reset by a kernel — the target is `(blk + 1) * group_cores`, so a
# member that races ahead into the next block can never clobber a value another
# member has not observed yet (which a set-to-0 reset would).
SEM_RM_FREE = 0  # receiver -> senders : "my re-tile stripe is free for this block"
SEM_RM_DATA = 1  # senders -> receiver : "my chunks for your stripe have landed"


def _div_up(a: int, b: int) -> int:
    """Ceiling division (ttnn.div_up is not exposed in every build)."""
    return (a + b - 1) // b


def _f32_bits(value: float) -> int:
    """fp32 bit pattern of `value`, as the kernels' scalar ops consume it."""
    return struct.unpack("I", struct.pack("f", float(value)))[0]


def _compute_config_descriptor(compute_kernel_config) -> ttnn.ComputeConfigDescriptor:
    """Translate a ttnn.DeviceComputeKernelConfig into a ComputeConfigDescriptor.

    ``ttnn.generic_op`` takes a ``ComputeConfigDescriptor``; the public parameter
    is a ``DeviceComputeKernelConfig`` (a ``WormholeComputeKernelConfig`` in
    practice).  This is the one place the two are bridged, field by field.
    ``packer_l1_acc`` / ``throttle_level`` have no descriptor counterpart and are
    not used by this kernel.
    """
    math_fidelity = compute_kernel_config.math_fidelity
    if math_fidelity == ttnn.MathFidelity.Invalid:
        # WormholeComputeKernelConfig's own default; normalise to the op's.
        math_fidelity = ttnn.MathFidelity.HiFi4

    return ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        math_approx_mode=bool(compute_kernel_config.math_approx_mode),
        fp32_dest_acc_en=bool(compute_kernel_config.fp32_dest_acc_en),
        dst_full_sync_en=bool(compute_kernel_config.dst_full_sync_en),
    )


def _retile_group_cores(device, num_token_blocks: int, tokens_per_block: int, flat_tiles: int) -> int:
    """Resolve ``RETILE_GROUP_CORES`` — how many cores cooperate on one token-block.

    A legal group size must divide BOTH split axes: ``tokens_per_block`` (the
    normalize half's token slice) and ``flat_tiles`` (the gate half's output
    column slice).  ``"auto"`` is the dispatch policy: take the largest legal
    power of two that fits in the grid capacity left over after every token-block
    already has a core, capped by ``MAX_RETILE_GROUP_CORES`` — so the exchange is
    paid for only with cores that would otherwise be idle, and a core-saturated
    shape stays on the byte-identical ``group_cores == 1`` path.
    """
    grid_size = device.compute_with_storage_grid_size()
    total_cores = grid_size.x * grid_size.y

    def _legal(g: int) -> bool:
        return 1 <= g <= total_cores and tokens_per_block % g == 0 and flat_tiles % g == 0

    if RETILE_GROUP_CORES != "auto":
        group_cores = int(RETILE_GROUP_CORES)
        assert _legal(group_cores), (
            f"onorm: RETILE_GROUP_CORES={group_cores} is not a legal cross-core re-tile group size. "
            f"It must be in 1..{total_cores} and divide BOTH TOKENS_PER_BLOCK={tokens_per_block} "
            f"(the token slice each core normalizes) and flat_tiles={flat_tiles} (the output column "
            f"slice each core gates and writes). Use a power of two."
        )
        return group_cores

    # The objective is the CRITICAL-PATH work per core, in whole-block units:
    #
    #     work(g) = ceil(num_token_blocks / num_groups(g)) / g
    #
    # — the slowest group's serial block count, divided by the fraction of a block
    # each of its members actually does.  Minimising it captures both effects the
    # measurements show, in one number:
    #   * OCCUPANCY at a small block count (T=32: 1 block, so work(g) = 1/g falls
    #     all the way to the cap — measured 16.1x at g=32);
    #   * LOAD BALANCE at a large one, which the naive "spend only spare cores"
    #     policy misses entirely.  B=8/T=640 has 160 blocks on 110 cores, so at
    #     g=1 fifty cores carry TWO blocks and sixty carry one: work(1) = 2.
    #     g=2 gives 55 groups x 3 blocks at half a block each = 1.5, i.e. a 1.33x
    #     shorter critical path on a shape a spare-capacity rule calls "saturated".
    #     Measured 1.22x.
    # Ties go to the SMALLER group: same critical-path work, but fewer exchange
    # messages per unit of it (the message COUNT per core per block is fixed at
    # TOKENS_PER_BLOCK, so per unit of work it grows with g) and a shallower serial
    # block loop, hence fewer per-block exchange barriers.  Measured: at B=1/T=128
    # g=16 and g=32 tie on work and g=16 is 1.12x faster.
    def _work(g: int) -> float:
        num_groups = min(num_token_blocks, total_cores // g)
        return _div_up(num_token_blocks, num_groups) / g

    best = 1
    candidate = 2
    while candidate <= MAX_RETILE_GROUP_CORES:
        if _legal(candidate) and _work(candidate) < _work(best):
            best = candidate
        candidate *= 2
    return best


def _grid_assignment(device, num_token_blocks, group_cores):
    """Spread the token-blocks over the whole compute grid, ``group_cores`` per block.

    ``row_wise=True`` is mandatory: the default column-major layout puts every
    core on the same shared NoC links (measured 2.91x slower on a DRAM<->DRAM
    stream).  It matters twice over here — the exchange group is a *contiguous
    run* of that same row-wise core order, so a group of `group_cores` cores sits
    inside one grid row wherever the row is wide enough, and the all-to-all stays
    a short-hop exchange along one row instead of crossing shared column links.

    ``split_work_to_cores`` still owns the grid -> core-set mapping (it is asked
    for exactly ``num_groups * group_cores`` cores, one unit each, so it returns
    that many cores in row-wise order); the blocks-over-groups split is the same
    ``base``/``remainder`` distribution it would apply itself, lifted one level up
    from cores to groups.  At ``group_cores == 1`` the two coincide exactly and
    this reproduces the pre-Refinement-2 assignment core-for-core.

    Returns ``(num_cores, all_cores, assignment)`` where each assignment entry is
    ``(core, slice_index, start_block, num_blocks, group_coords)`` and
    ``group_coords`` is the group's ``[x0, y0, x1, y1, ...]`` VIRTUAL NoC coords,
    in slice order, identical for every member.
    """
    grid_size = device.compute_with_storage_grid_size()
    total_cores = grid_size.x * grid_size.y

    num_groups = min(num_token_blocks, total_cores // group_cores)
    assert num_groups >= 1, (
        f"onorm: RETILE_GROUP_CORES={group_cores} needs at least that many cores, but the grid has "
        f"only {total_cores}. Lower RETILE_GROUP_CORES."
    )
    cores_needed = num_groups * group_cores

    num_cores, all_cores, _, _, _, _ = ttnn.split_work_to_cores(grid_size, cores_needed, row_wise=True)
    assert num_cores == cores_needed, f"onorm: asked for {cores_needed} cores, split gave {num_cores}"
    core_list = ttnn.corerange_to_cores(all_cores, None, True)
    assert len(core_list) == cores_needed

    blocks_per_group, remainder = divmod(num_token_blocks, num_groups)
    assignment = []
    start_block = 0
    for g in range(num_groups):
        num_blocks = blocks_per_group + (1 if g < remainder else 0)
        members = core_list[g * group_cores : (g + 1) * group_cores]
        group_coords = []
        for core in members:
            virtual = device.worker_core_from_logical_core(core)
            group_coords += [virtual.x, virtual.y]
        for slice_index, core in enumerate(members):
            assignment.append((core, slice_index, start_block, num_blocks, group_coords))
        start_block += num_blocks
    assert (
        start_block == num_token_blocks
    ), f"onorm: work split covered {start_block} of {num_token_blocks} token-blocks"
    return num_cores, all_cores, assignment


def create_program_descriptor(
    o: ttnn.Tensor,
    gate: ttnn.Tensor,
    weight: ttnn.Tensor,
    output: ttnn.Tensor,
    epsilon: float,
    compute_kernel_config,
) -> ttnn.ProgramDescriptor:
    device = o.device()

    # ================= 1. GEOMETRY, DERIVED FROM THE KNOBS =================
    batch, tokens, num_heads, head_dim = list(o.shape)
    flat_width = int(gate.shape[-1])

    v_tiles = _div_up(head_dim, TILE_W)  # column tiles per head-major image
    flat_tiles = _div_up(flat_width, TILE_W)  # column tiles per flat tile-row
    token_tile_rows = _div_up(tokens, TILE_H)  # `Tt`: gate/out tile-rows per batch

    tile_rows_per_block = TOKENS_PER_BLOCK // TILE_H
    blocks_per_batch = _div_up(tokens, TOKENS_PER_BLOCK)
    num_token_blocks = batch * blocks_per_batch

    # --- cross-core re-tile group: how many cores share ONE token-block ---
    # (Refinement 2.  `group_cores == 1` is the pre-Refinement-2 path exactly.)
    group_cores = _retile_group_cores(device, num_token_blocks, TOKENS_PER_BLOCK, flat_tiles)
    # The two split axes of one token-block. Everything per-core below derives
    # from these two and nothing restates a whole-block quantity.
    tokens_per_core = TOKENS_PER_BLOCK // group_cores  # normalize half: token slice
    cols_per_core = flat_tiles // group_cores  # gate half: output column slice

    # Per-core (not per-block) tile counts. The kernels derive the same values
    # from the same knobs; only the flat one is needed host-side, for CB sizing.
    flat_tiles_per_core = tile_rows_per_block * cols_per_core
    # A core with `tokens_per_core` tokens cannot run a coarser normalize chunk
    # than that, and a core with `flat_tiles_per_core` output tiles cannot run a
    # coarser gate chunk — so both block factors are REQUESTS the descriptor
    # clamps, the same idiom GATE_DEST_TILES uses for the DEST budget. Clamping
    # (rather than asserting) is what keeps both knobs live and independent of the
    # group size: raising `group_cores` never turns a legal knob setting into a
    # host assert, it just stops the chunk growing past the slice it must fit in.
    norm_chunk_tokens = min(NORM_CHUNK_TOKENS, tokens_per_core)
    gate_chunk_tiles = min(GATE_CHUNK_TILES, flat_tiles_per_core)
    norm_chunks_per_block = tokens_per_core // norm_chunk_tokens
    gate_chunks_per_block = flat_tiles_per_core // gate_chunk_tiles

    # --- knob consistency (a violated knob relation is a silent wrong answer) ---
    assert TOKENS_PER_BLOCK % TILE_H == 0, "TOKENS_PER_BLOCK must be a multiple of the tile height"
    assert tokens_per_core % norm_chunk_tokens == 0, (
        f"NORM_CHUNK_TOKENS={NORM_CHUNK_TOKENS} (clamped to {norm_chunk_tokens}) must divide the "
        f"{tokens_per_core} tokens this core owns (TOKENS_PER_BLOCK={TOKENS_PER_BLOCK} / "
        f"RETILE_GROUP_CORES={group_cores}); use a power of two"
    )
    assert flat_tiles_per_core % gate_chunk_tiles == 0, (
        f"GATE_CHUNK_TILES={GATE_CHUNK_TILES} (clamped to {gate_chunk_tiles}) must divide the "
        f"{flat_tiles_per_core} flat output tiles this core owns; use a power of two"
    )
    assert 1 <= DM_BLOCK_TILES <= 8, "DM_BLOCK_TILES is a 1..8 knob"
    assert SIGMOID_ENGINE in _SIGMOID_ENGINE_CODES, (
        f"onorm: SIGMOID_ENGINE={SIGMOID_ENGINE!r} is not one of "
        f"{sorted(_SIGMOID_ENGINE_CODES)}. 'math' and 'pack' are the two shipping "
        f"engines; 'ablate' is a measurement-only setting that produces WRONG output."
    )
    assert SIGMOID_ENGINE != "ablate" or ALLOW_SIGMOID_ABLATION, (
        "onorm: SIGMOID_ENGINE='ablate' drops the sigmoid and produces NUMERICALLY "
        "WRONG output. It exists only for /perf-measure ablation profiling. Set "
        "onorm_program_descriptor.ALLOW_SIGMOID_ABLATION = True to opt in."
    )
    # The DEST window ceiling follows the CALLER's DEST width, so GATE_DEST_TILES
    # is a REQUEST that is clamped down to what this config can actually stage.
    # Clamping (rather than asserting against the active limit) is what keeps the
    # public `fp32_dest_acc_en=True` path working from the same module-level knob:
    # the knob's range now reaches 8, which a caller-supplied 32-bit DEST cannot
    # hold, and such a caller must get a legal window rather than an assert.
    dest_tile_limit = _dest_tile_limit(bool(compute_kernel_config.fp32_dest_acc_en))
    assert 1 <= GATE_DEST_TILES <= _DEST_TILE_LIMIT_16B, (
        f"onorm: GATE_DEST_TILES={GATE_DEST_TILES} must be 1..{_DEST_TILE_LIMIT_16B} "
        f"(DEST_AUTO_LIMIT under half sync, 16-bit DEST — the widest this op ever gets)."
    )
    # ...and it can never exceed the gate chunk it subdivides, which the cross-core
    # column split shrinks (`flat_tiles / group_cores`).
    gate_dest_tiles = min(GATE_DEST_TILES, dest_tile_limit, gate_chunk_tiles)
    assert gate_chunk_tiles % gate_dest_tiles == 0, (
        f"onorm: the effective GATE_DEST_TILES={gate_dest_tiles} (requested "
        f"{GATE_DEST_TILES}, clamped to this config's DEST limit {dest_tile_limit}) "
        f"must divide the effective GATE_CHUNK_TILES={gate_chunk_tiles}; use a power of two."
    )
    assert DM_DEPTH >= 2 and O_DEPTH >= 2, "streaming depths must be >= 2 to overlap read with compute"
    # `o`'s token axis is un-padded (tiled dims are (HV, V)) while gate/out's IS
    # tile-padded (tiled dims are (T, FLAT)).  T % TOKENS_PER_BLOCK == 0 is what
    # makes the two views coincide; a partial last block would read past `o`.
    assert (
        tokens % TOKENS_PER_BLOCK == 0
    ), f"onorm: T={tokens} must be a multiple of TOKENS_PER_BLOCK={TOKENS_PER_BLOCK}"

    # Every CB is one tile per page, and the reader/writer address all four
    # buffers with that same `page_bytes` CT arg — so all four must agree.
    tile_bytes = o.buffer_page_size()
    for name, tensor in (("gate", gate), ("weight", weight), ("output", output)):
        assert tensor.buffer_page_size() == tile_bytes, (
            f"onorm: {name} page size {tensor.buffer_page_size()} B differs from o's {tile_bytes} B; "
            f"the kernels stream every tensor with one shared page_bytes compile-time arg"
        )

    # ================= 2. WORK DISTRIBUTION =================
    _, all_cores, assignment = _grid_assignment(device, num_token_blocks, group_cores)

    # ================= 3. CIRCULAR BUFFERS =================
    # Streaming input/output CBs get `DM_BLOCK_TILES * DM_DEPTH` (the
    # double-buffer knob); intermediates between two *sequential* compute
    # helpers get the full block they must hold (both helpers own all three
    # TRISCs, so they cannot pipeline).  Nothing here scales with B or T.
    cb_pages = {
        CB_O_TILES: v_tiles * norm_chunk_tokens * O_DEPTH,
        CB_GATE_TILES: DM_BLOCK_TILES * DM_DEPTH,
        CB_WEIGHT: v_tiles,
        CB_SCALER: 1,
        CB_OUT_TILES: DM_BLOCK_TILES * DM_DEPTH,
        CB_SUMSQ: norm_chunk_tokens,
        CB_RSTD: norm_chunk_tokens,
        CB_NORMED: v_tiles * norm_chunk_tokens,
        CB_ONORM: v_tiles * norm_chunk_tokens,
        # EXACTLY the stripe this core tilizes, and no more: the tilize address
        # generator assumes ONE contiguous [TOKENS_PER_BLOCK, cols_per_core*TILE_W]
        # row-major stripe of row stride `cols_per_core*TILE_W` elements, so a
        # larger CB would let the ring wrap mid-block.  At group_cores == 1 this is
        # the whole [TOKENS_PER_BLOCK, FLAT] block, as before; the cross-core column
        # split narrows the stripe by exactly `group_cores`, which is why the
        # exchange COSTS no L1 and in fact frees a lot of it.
        CB_RM_FLAT_ROWS: flat_tiles_per_core,
        CB_FLAT_TILES: flat_tiles_per_core,
        CB_GATE_SIG: gate_chunk_tiles,
    }
    if group_cores > 1:
        # The scatter staging buffer: compute untilizes one normalize chunk's
        # row-major token rows into it, the writer scatters them and pops.  Only
        # exists when the block IS split — at group_cores == 1 the define
        # ONORM_CB_RM_LOCAL aliases CB_RM_FLAT_ROWS and compute untilizes straight
        # into the stripe it later tilizes (one CB, no copy, no exchange).
        cb_pages[CB_RM_LOCAL] = v_tiles * norm_chunk_tokens * RM_LOCAL_DEPTH

    # The reader/writer transfer whole DM groups out of one get_write_ptr /
    # get_read_ptr, so a group must never straddle the CB's ring wrap.
    #
    # DM_BLOCK_TILES is a REQUEST, clamped per stream to that stream's own
    # granularity — a transfer group can never usefully exceed the amount the
    # consumer takes in one bite, and the cross-core split shrinks both bites:
    #   * `o`             — one normalize chunk (`v_tiles * norm_chunk_tokens`);
    #   * `gate` / `out`  — this core's column slice of one tile-row (`cols_per_core`).
    # Clamping (rather than asserting against the raw knob) is what keeps
    # DM_BLOCK_TILES and RETILE_GROUP_CORES independent: a large group size makes
    # each stream shorter, and that must not turn a legal DM_BLOCK_TILES into a
    # host assert.  At group size 1 both clamps are inactive at every shipped
    # setting, so this is byte-identical there.
    o_dm_block_tiles = min(DM_BLOCK_TILES, v_tiles * norm_chunk_tokens)
    flat_dm_block_tiles = min(DM_BLOCK_TILES, cols_per_core)
    for cb_index, group in (
        (CB_O_TILES, o_dm_block_tiles),
        (CB_GATE_TILES, flat_dm_block_tiles),
        (CB_OUT_TILES, flat_dm_block_tiles),
    ):
        assert cb_pages[cb_index] % group == 0, (
            f"onorm: CB {cb_index} has {cb_pages[cb_index]} pages, not a multiple of its "
            f"{group}-tile DM transfer group (DM_BLOCK_TILES={DM_BLOCK_TILES} clamped to the "
            f"stream's granularity). Pick a DM_BLOCK_TILES that divides it (a power of two "
            f"always does), or raise O_DEPTH / NORM_CHUNK_TOKENS / DM_DEPTH."
        )

    # A gate-phase DEST window stages GATE_DEST_TILES tiles at once, so every CB
    # those two phases wait on / reserve must hold at least that many pages.
    # cb_gate_tiles (DM_BLOCK_TILES * DM_DEPTH) and cb_out_tiles are the tightest.
    for cb_index in (CB_GATE_TILES, CB_GATE_SIG, CB_FLAT_TILES, CB_OUT_TILES):
        assert cb_pages[cb_index] >= gate_dest_tiles, (
            f"onorm: GATE_DEST_TILES={gate_dest_tiles} exceeds CB {cb_index}'s "
            f"{cb_pages[cb_index]} pages — a DEST window would wait on more tiles than "
            f"the CB can hold and deadlock. Raise DM_BLOCK_TILES / DM_DEPTH, or lower "
            f"GATE_DEST_TILES."
        )

    total_cb_bytes = sum(cb_pages.values()) * tile_bytes
    # The CB-available slice of L1 (unreserved L1 minus the CB-region base), i.e.
    # the actual budget the knobs must fit inside.
    l1_available = ttnn.get_max_worker_l1_unreserved_size() - _L1_CB_BASE_RESERVE
    assert total_cb_bytes <= l1_available, (
        f"onorm: CB footprint {total_cb_bytes} B exceeds the CB-available L1 per core "
        f"({l1_available} B). "
        f"Lower GATE_CHUNK_TILES first (currently {GATE_CHUNK_TILES} — a pure perf/L1 "
        f"trade), then NORM_CHUNK_TOKENS (currently {NORM_CHUNK_TOKENS}). "
        f"CB_RM_FLAT_ROWS / CB_FLAT_TILES are the re-tile working set and are not "
        f"reducible at a fixed RETILE_GROUP_CORES (currently {group_cores}) — but "
        f"RAISING RETILE_GROUP_CORES divides both of them by the group size."
    )

    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_index,
                    data_format=o.dtype,
                    page_size=tile_bytes,
                )
            ],
        )
        for cb_index, pages in cb_pages.items()
    ]

    # ================= 4. KERNELS =================
    # Reader (NCRISC / NoC0) — ALL DRAM reads live here.  Reads issued on NoC1
    # measured 4.8x slower, so the writer never reads `gate`.
    reader_ct_args = [
        v_tiles,
        TOKENS_PER_BLOCK,
        tokens_per_core,
        flat_tiles,
        cols_per_core,
        tile_rows_per_block,
        blocks_per_batch,
        tokens,
        token_tile_rows,
        o_dm_block_tiles,  # per-stream clamped DM group, not the raw knob
        flat_dm_block_tiles,
        _f32_bits(1.0 / head_dim),  # the reduce scaler: explicit, host-supplied 1/V
        tile_bytes,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(o).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(gate).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())

    # Writer (BRISC / NoC1) — all DRAM writes, plus the cross-core re-tile
    # scatter (also a write, so it belongs on NoC1 with them).
    writer_ct_args = [
        flat_tiles,
        cols_per_core,
        tile_rows_per_block,
        blocks_per_batch,
        token_tile_rows,
        flat_dm_block_tiles,  # per-stream clamped DM group, not the raw knob
        tile_bytes,
        group_cores,
        tokens_per_core,
        norm_chunk_tokens,
        norm_chunks_per_block,
        v_tiles,
        SEM_RM_FREE,
        SEM_RM_DATA,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    compute_ct_args = [
        norm_chunk_tokens,
        norm_chunks_per_block,
        v_tiles,
        cols_per_core,
        tile_rows_per_block,
        gate_chunk_tiles,
        gate_chunks_per_block,
        _SIGMOID_ENGINE_CODES[SIGMOID_ENGINE],
        gate_dest_tiles,  # the config-clamped effective window, not the raw request
    ]

    o_addr = o.buffer_address()
    gate_addr = gate.buffer_address()
    weight_addr = weight.buffer_address()
    out_addr = output.buffer_address()
    eps_bits = _f32_bits(epsilon)

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    for core, slice_index, start_block, num_blocks, group_coords in assignment:
        reader_rt_args[core.x][core.y] = [o_addr, gate_addr, weight_addr, start_block, num_blocks, slice_index]
        # `group_coords` is the whole exchange group's virtual NoC coords in slice
        # order; every member carries the same list and indexes it by destination.
        writer_rt_args[core.x][core.y] = [out_addr, start_block, num_blocks, slice_index, *group_coords]
        compute_rt_args[core.x][core.y] = [num_blocks, eps_bits]

    kernel_defines = _kernel_defines(group_cores)

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "onorm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        defines=kernel_defines,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "onorm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        defines=kernel_defines,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "onorm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        defines=kernel_defines,
        runtime_args=compute_rt_args,
        config=_compute_config_descriptor(compute_kernel_config),
    )

    # Two monotone counters, only when the block is actually split.  The initial 0
    # MUST come from the host: a member increments a REMOTE core's cell with no
    # happens-before edge to that core's kernel start, so any kernel-side init
    # would race and could clobber an early increment.
    semaphores = (
        [
            ttnn.SemaphoreDescriptor(id=SEM_RM_FREE, core_ranges=all_cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_RM_DATA, core_ranges=all_cores, initial_value=0),
        ]
        if group_cores > 1
        else []
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
