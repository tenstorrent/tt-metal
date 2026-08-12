# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor (CBs, kernels, args).

This file owns the Blocking Model (op_design.md §1). Every knob is defined
EXACTLY ONCE here and every dependent quantity (CB page counts, loop trip
counts, grid sizing, kernel args) is computed *from* it — never from a whole-op
dimension and never as a magic literal:

  knob                 symbol             defined as
  -------------------  -----------------  ------------------------------------
  read byte target     TARGET_READ_BYTES  module constant (1024, measured — see below)
  max block width      WT_BLOCK_MAX       max(2, TARGET_READ_BYTES // (32*elem))
  block width (tiles)  WT_BLOCK           min(Wt, WT_BLOCK_MAX)
  tail block width     WT_TAIL            Wt - (n_wchunks-1)*WT_BLOCK
  column-blocks/row    n_wchunks          ceil(Wt / WT_BLOCK)
  grid cores           grid_cores         1 if not use_multicore else grid.x*y
  CB depth             CB_DEPTH           2 if use_double_buffer and it fits L1
  cast flag            NEEDS_CAST         out_dtype != in_dtype

One block = one output tile-row x WT_BLOCK output tile-columns. Blocks are
linearized `b = wchunk * nt_h + r` and that linear space is spread across the
grid by `split_work_to_cores(grid, nt_h * n_wchunks, row_wise=True)`, which
subsumes both distribution regimes with no gate expression: when
`Wt <= WT_BLOCK_MAX` (`n_wchunks == 1`) the block index *is* the tile-row index,
so it degenerates to the pure height split; when `nt_h == 1` (wide-short) it
degenerates to the pure width split and still fills the grid.
"""

from pathlib import Path

import math
import struct

import ttnn
from ttnn.operations._op_contract import UnsupportedAxisValue

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- CB slots (semantic names are the primary identifier) -------------------
CB_INPUT_STICKS = 0  # reader -> compute : row-major sticks, tile-sized pages
CB_OUTPUT_TILES = 16  # compute -> writer : tiled output pages
# R7 (A5b) reader-private scratch: a DRAM-aligned staging area for the
# narrow-stick read (1-byte datum at an odd block width). Never pushed or
# popped, and only allocated on that path.
CB_READ_STAGE = 1
# CB slots the per-CB `unpack_to_dest_mode` vector (R7) must cover. 64, not 32:
# `data_format.cpp` requires `size() >= buf_formats.size()`, and that is the
# ARCH's CB count (32 on Wormhole, 64 on Blackhole) — `circular_buffer_constants.h`
# documents 64 as "the max for array sizing" precisely so a host-side vector can
# be arch-portable. This op occupies two of the slots; the rest stay Default.
NUM_CB_SLOTS = 64

# --- Knob: read transaction byte target ------------------------------------
# Expressed in BYTES so every dtype lands on the same measured sweet spot
# (bf16 -> WT_BLOCK 16, fp32 -> 8, uint8 -> 32); sweep this ONE line to move the
# transaction size, and `_bench_tilize.py`'s `lever_b6_read_*` arms re-measure it.
#
# 1024, not the 512 B "one-packet" value: 512 B is NOC_MAX_BURST_SIZE on
# **Wormhole**; on Blackhole (this box) NOC_MAX_BURST_SIZE is 256 words x 64 B =
# 16 KB, so the one-packet argument does not bind at 512 B and the sweep decides.
# Measured (grid 11x10, bf16, DEVICE KERNEL DURATION, phase-0 bench):
#   [1,1,2048,2048]  128 B 152.6us | 256 B 86.3us | 512 B 55.9us | 1024 B 44.5us
#                    | 2048 B 44.7us | 4096 B 47.0us
#   [1,1,32,16384]   512 B 7.69us | 1024 B 7.41us | 2048 B 7.87us | 4096 B 9.37us
#   [1,1,2048,64] and [1,1,32,64] are unaffected (WT_BLOCK = min(Wt, ...) clamps).
# 1024 B is the joint optimum and regresses no benched regime.
TARGET_READ_BYTES = 1024

# --- Knob: pipeline depth (minimum blocks per core) ------------------------
# Blocks-per-core IS the pipeline depth: a core's reader, compute and writer
# overlap only across DIFFERENT blocks (one block's `tile_h` sticks must all land
# before its tiles can be tilized, and its tiles must all be tilized before they
# can be written), so a core holding exactly ONE block runs
# read -> compute -> write strictly serially.
#
# `total_blocks` is a function of the SHAPE and of `WT_BLOCK`, so on a shape with
# few blocks the default "spread over as many cores as there are blocks" split
# hands every core a single block and forfeits all overlap. This knob trades grid
# fill for pipeline depth: cores are capped at `total_blocks //
# min_blocks_per_core`. 1 is the trivial value (byte-identical to the Phase-0
# split), and the cap only ever BINDS when
# `total_blocks < grid_cores * MIN_BLOCKS_PER_CORE` — the grid-fill-deficit
# regime, never on a shape that already fills the grid with depth to spare
# (`pipeline_capped_cores` is that gate, stated once).
#
# Measured (R3, BH 11x10, DEVICE KERNEL DURATION, off-arm = min_blocks_per_core=1):
#   l1_to_l1  [1,1,512,2048]   64 cores x 1 blk 6 722 ns -> 32 x 2 **5 091 ns (1.32x)**
#   wide_short[1,1,32,16384]   32 cores x 1 blk 7 184 ns -> 16 x 2   7 156 ns (1.00x)
#   tall_narrow[1,1,2048,64]   64 cores x 1 blk 4 931 ns -> 32 x 2   4 930 ns (1.00x)
# i.e. hiding compute behind DM only pays where DM is FAST ENOUGH for compute to
# be co-binding, which is R1's measured L1<->L1 profile (ablate_compute 0.596x)
# and NOT the DRAM profile (ablate_compute 0.994x on the square). Hence the
# regime gate in `placement_defaults` rather than a global default.
MIN_BLOCKS_PER_CORE = 2

# --- Knob: placement of a partially-filled grid (master.md A1/A3) -----------
# Whether the active cores are SPREAD over the grid or packed into its first
# rows. Only observable when `active_cores < grid_cores`.
#
# Measured (R3, off-arm = spread_cores=0, both at 32 cores x 1 block):
#   wide_short [1,1,32,16384]  packed 7 184 / 7 275 ns -> spread **6 873 / 6 808 ns (1.05x)**
#   tall_narrow[1,1,2048,64]   packed 4 968 ns          -> spread   4 913 ns (1.01x, noise)
#   l1_to_l1   [1,1,512,2048]  packed 5 091 ns          -> spread   5 479 ns (**0.93x, a regression**)
# Opposite signs, and the mechanism is A3: DRAM banks sit in a few columns, so
# route diversity is what a packed slab of DRAM readers lacks — whereas an
# L1-interleaved operand's "banks" ARE the worker cores, so spreading the
# consumers lengthens their L1 routes instead of shortening them. Gated on the
# operands' buffer type in `placement_defaults`, never applied globally.
SPREAD_CORES = 1

# --- Knob: per-core read-issue stagger (master.md A3) ----------------------
# Rotate each core's read ISSUE ORDER by its own block index. Same transfers,
# same destinations. Measured below; DRAM-path only (an L1 read does not queue
# behind a DRAM bank).
STAGGER_READS = 0

# --- Knob: stateful, bank-grouped NoC reads (R6 / master.md B13 + D21) -----
# The low-work-per-core regimes are ISSUE-bound, not bandwidth-bound: measured,
# `[1,1,32,64]` spends 968 ns of a 1 901 ns call on 32 reads that move 4 KB in
# total (~30 ns per read), and `[1,1,32,32]` spends 1 084 ns on 32 reads of 64 B.
# What costs there is the SIX command-buffer register writes a plain
# `noc_async_read` performs (RET_LO, TARG_LO, TARG_MID, TARG_COORD, LEN,
# CMD_CTRL), not the bytes. `set_state` programs the source NODE once and each
# subsequent issue writes only three registers.
#
# The state is per source NODE, so the sticks are issued in BANK-PHASE order
# (page p of an interleaved buffer lives in bank `p % num_banks`), which is what
# makes one `set_state` cover several sticks. `read_bank_period` below is a pure
# performance hint — the kernel compares the programmed node against every
# address and re-programs on a mismatch, so a wrong period costs a few extra
# `set_state`s and can never read the wrong bytes.
STATEFUL_READS = 0

# --- Knob: incremental address generation inside a bank group (R6 / D21) ---
# `TensorAccessor::get_noc_addr(page)` costs TWO divisions by the bank count to
# split the page into (bank, page-in-bank), and this part's bank count is 7 — not
# a power of two, so both are software divides. Inside one bank-phase group the
# pages are consecutive IN THAT BANK, so their addresses form an arithmetic
# progression: two accessor calls per group give the rest by addition (~14 per
# block instead of 32 at `tile_h = 32`, 7 banks). The stride is DERIVED from two
# real accessor addresses and used only when they agree on the source node, so a
# wrong `read_bank_period` costs accessor calls and never correctness.
FAST_ADDRGEN = 0

# --- Knob: fold the dataflow kernels away on the resident path (R6 / C14-2) -
# master.md C14 has TWO degrees: removing the NoC traffic (Refinement 2 — the CBs
# ARE the shards) and removing the KERNELS (a resident reader exists only to run
# the CB handshake). This is the second. 1 = emit the compute kernel only, which
# self-arms the input CB and self-drains the output CB; 0 = keep the three-kernel
# structure. `examples/zero_copy_fold` measured folding at 0.74x on Wormhole at
# 2 tiles/core, so this ships at whatever THIS part measures, not at that.
FOLD_RESIDENT = 0

# --- Knob: the tilize LLK teardown (R6) ------------------------------------
# `tilize_uninit` is per-CALL fixed cost, and the low-work regimes are exactly
# where per-call fixed cost is the wall: `[1,1,32,64]` spends ~0.5 us of a ~1.9 us
# call inside a tilize of TWO tiles, i.e. almost all of it on init/uninit rather
# than on the 26 ns/tile the grid-filling shapes measure. 1 = the helper's default
# `InitAndUninit`, 0 = `InitOnly` (init kept, teardown skipped).
TILIZE_UNINIT = 1

# --- Knob: collapse the per-block input wait on the resident path (R6) -----
# `WaitMode::WaitBlock` is the right default on the streamed path (it is what
# lets the reader run ahead of compute), but a RESIDENT input has no reader to
# run ahead of: the shard is already in L1 and the reader arms the CB with every
# page in a single `cb_push_back`, so the per-block waits can only re-observe a
# semaphore that is already set. 1 = `WaitUpfront` on the resident path (one wait
# per call), 0 = `WaitBlock` everywhere, i.e. the pre-R6 kernel byte-for-byte.
# NEVER armed on a streamed input: there the CB holds `cb_depth * wt_block`
# pages, not the whole assignment, so an upfront wait would deadlock.
WAIT_UPFRONT_RESIDENT = 1

# --- Knob: cross-spec reshard direction (R4 / A3c) -------------------------
# Which SIDE of a cross-spec reshard holds the resident block: 1 = PULL (the
# output shard, op_design §4.3's contract), 0 = PUSH (the input shard).
# `None` = decide per geometry with `reshard_direction_is_pull`, which is the
# SHIPPED path — the two directions measure with opposite signs on geometries
# that differ only in which side has more cores, so neither can be a global
# default. 1/0 force one direction on any geometry (both bench arms).
RESHARD_PULL = None

# --- Knob: fp32 identity precision (R7 / A4) -------------------------------
# `Fp32Mode::Fast` truncates fp32 -> tf32 on the way into DEST, which is the
# right default for a tilize whose consumers are FPU ops (they re-truncate
# anyway — `tilize_helpers.hpp:52-71`). But tilize's own contract is a BIJECTION
# ON BYTE POSITIONS, so an fp32 -> fp32 call must come back bit-exact, and Fast
# measured PCC 0.999998 / max diff 1.6e-2 on it (R2b's probe_011). `Lossless` is
# the one configuration the helper documents for that: it needs BOTH
# `fp32_dest_acc_en` and `UnpackToDestFp32` on the input CB (the helper
# static_asserts both), which is why this is a policy triple rather than a flag.
# Armed ONLY on the fp32 -> fp32 identity: a narrowing fp32 -> bf16/bf8b cast
# lands in a 7-bit mantissa either way, so Fast costs it nothing.
FP32_LOSSLESS_IDENTITY = 1

# --- Knob: 8-bit DEST width (R7 / A5b) -------------------------------------
# An 8-bit datum has no DEST representation of its own: the regular tilize path
# is unpack -> SrcA -> (A2D datacopy) -> DEST -> pack, and with a 16-bit DEST the
# packer reads the int8 payload as a float16 denormal and writes ZERO. Measured:
# the whole uint8 tile comes back 0 (probe_023), with the reader's L1 verified
# correct (probe_026) — so it is a DEST-width problem, not a dataflow one.
# `fp32_dest_acc_en` widens DEST to 32 bits, which is also the branch
# `get_single_pack_src_format` needs to select an integer pack source for UInt8
# (`data_format.cpp:271-283` — UInt16 has its own early branch, UInt8 does not).
FP32_DEST_ACC_8BIT = 1

TILE_WIDTH = 32  # a tile is ALWAYS 32 wide; only its height varies
# The tile HEIGHT the op uses when the caller passes no `tile=`. Defined ONCE
# here and imported by the op file so the taggers, validate(), the entry point
# and the descriptor can never disagree about what "the default tile" is.
DEFAULT_TILE_HEIGHT = 32

# --- Lever knobs (the perf-gate counterfactual surface) --------------------
# Every performance lever this op lands is a NAMED knob here, so its off-arm is
# re-runnable from the bench (`levers=dict(<knob>=0)`) instead of being an ad-hoc
# kernel edit. The defaults below ARE the production path — `dict(DEFAULT_LEVERS)`
# reproduces the shipped kernel byte-for-byte, so an unmeasured knob costs
# nothing. `stub_*` are the /perf-measure ablation arms (keep the sync
# scaffolding, drop one payload) and are never on in production.
DEFAULT_LEVERS = {
    "multicore": 1,  # A0: full grid vs a single core (also the user-facing kwarg)
    "width_split": 1,  # A0/A1: 2-D linearization (b = wchunk*nt_h + r) vs height-only
    "row_wise": 1,  # A1: spread cores across the DRAM-facing (row) axis
    "target_read_bytes": TARGET_READ_BYTES,  # B6/B7: read transaction size -> WT_BLOCK
    # R3 (A0/B0/C16): the PIPELINE-DEPTH knob — the minimum number of blocks a
    # core is given, enforced by capping the core count at `total_blocks //
    # min_blocks_per_core`. Blocks per core is exactly the number of
    # read/compute/write stages that can overlap in a core's own CB pipeline, so
    # a core holding ONE block runs read -> compute -> write strictly serially
    # (measured: on `[1,1,32,16384]` the three ablation stage costs SUM to 104 %
    # of the removable wall, vs 72 % on the square). 1 = the trivial off-arm and
    # byte-identical to Phase 0's split. `None` = "take the regime default"
    # (`placement_defaults`), which is what the SHIPPED path does — an explicit
    # value forces the knob on ANY shape, so both arms stay measurable everywhere.
    "min_blocks_per_core": None,
    # R3 (A1/A3): PLACEMENT of a partially-filled grid. `grid_to_cores(n)` returns
    # the first n cores of the row-major enumeration — a solid slab in the first
    # rows — so a shape whose block count is below the grid size reaches DRAM over
    # a handful of shared links. 1 spreads the same n cores uniformly over the
    # whole grid (same count, same block assignment, different cores); 0 is the
    # packed Phase-0 placement. Structurally a no-op once n == grid_cores.
    # `None` = the regime default (`placement_defaults`), as above.
    "spread_cores": None,
    "coalesce_writes": 1,  # B5: whole-tile-page writes vs per-face writes
    "barrier_per_block": 1,  # B7: one barrier per block vs one per transaction
    "noc_split": 1,  # B9: reader NoC0 / writer NoC1 vs swapped
    # R3 (master.md A3): rotate each core's READ ISSUE ORDER by its own block
    # index. Same transfers, same L1 destinations, different order — it exists
    # because on a wide-short tensor (`nt_h == 1`) every core reads the SAME
    # `tile_h` source pages, so an unstaggered fleet requests one page (one DRAM
    # bank) at a time. `None` = the regime default (`placement_defaults`).
    "stagger_reads": None,
    # R6 (master.md B13 + D21): configure the NoC command buffer ONCE per source
    # bank and issue the block's remaining same-shape reads with `with_state`
    # (three register writes instead of six). `None` = the shipped default
    # (`STATEFUL_READS`), 0 = plain `noc_async_read` per stick, i.e. the
    # `read_sticks_for_tilize` helper call byte-for-byte. Only ever armed where
    # bank grouping can pay (`read_bank_period` decides); a lever value of 1 on a
    # shape whose period is 0 is still the plain path.
    "stateful_reads": None,
    # R6 (master.md D21): derive the addresses inside a bank-phase group by
    # addition instead of asking `TensorAccessor` (two divides by the bank count)
    # per stick. `None` = the shipped default (`FAST_ADDRGEN`), 0 = one accessor
    # call per stick. Like `stateful_reads` it is only ever armed where bank
    # grouping is defined (`read_bank_period`).
    "fast_addrgen": None,
    # R6 (master.md C14, SECOND degree): fold the two dataflow kernels away on
    # the same-spec resident path, where they exist only to run the CB handshake
    # (zero NoC bytes on both sides). 1 = compute-only program, 0 = the
    # three-kernel structure. Unreachable off `MODE_RESIDENT`.
    "fold_resident": None,
    # R6: keep (1) or skip (0) the tilize LLK teardown. Skipping it leaves the
    # unpacker configured for tilize when the kernel ends, which the NEXT compute
    # kernel's own `compute_kernel_hw_startup`/init would have to overwrite — so
    # it only ships if the measurement is worth that coupling.
    "tilize_uninit": None,
    # R6: `WaitUpfront` instead of `WaitBlock` on the resident path (see
    # `WAIT_UPFRONT_RESIDENT`). `None` = the regime default, 0 = `WaitBlock`
    # everywhere. Forcing 1 on a streamed input is not offered — it deadlocks by
    # construction, so the knob is gated on residency rather than trusted.
    "wait_upfront": None,
    "double_buffer": 1,  # C16: CB depth 2 vs 1  (also the user-facing kwarg)
    # A2/C14 off-arm: consume a resident shard through a TensorAccessor instead
    # of aliasing the CB onto it. 1 = the NON-zero-copy counterfactual (the
    # interleaved path merely TOLERATING a sharded tensor), 0 = the shipped
    # zero-copy placement. Only legal where the streamed reader can address the
    # input (interleaved, or a shard whose width is the full row).
    "force_streamed": 0,
    # R4 (A3c): which SIDE of a cross-spec reshard is the resident block, i.e.
    # the direction of the one cross-core transfer this op has.
    #   1 = PULL  (op_design §4.3): the OUTPUT shard is resident; each output
    #       core reads the input bands it needs from whichever core holds them.
    #   0 = PUSH: the INPUT shard is resident; each input core tilizes its own
    #       block and writes whole tile pages to whichever core owns them.
    # Both are bijections and neither needs a semaphore or a multicast (§1.1),
    # so the design's stated reason for pull does not discriminate between them
    # — the transaction SHAPE does, and it is per-geometry. `None` = take the
    # measured regime default (`reshard_default_pull`), which is what the
    # shipped path does; an explicit value forces one side on any shape.
    "reshard_pull": None,
    # R7 (A4): the fp32 -> fp32 bit-exact triple (`Fp32Mode::Lossless` +
    # `fp32_dest_acc_en` + `UnpackToDestFp32` on the input CB). `None` = the
    # shipped default (`FP32_LOSSLESS_IDENTITY`, armed on the fp32 identity
    # only), 0 = `Fp32Mode::Fast` everywhere — which is the pre-R7 kernel
    # byte-for-byte and the off-arm for the "what does bit-exactness cost?"
    # measurement. Never armed off the fp32 identity, where it is a no-op.
    "fp32_lossless": None,
    # R7 (A5b): widen DEST to 32 bits for an 8-bit datum. `None` = the shipped
    # default (`FP32_DEST_ACC_8BIT`), 0 = the 16-bit DEST, i.e. the arm that
    # produces an all-zero uint8 tile. Kept as a knob because it is the
    # correctness lever for 8-bit AND a throughput cost everywhere else, so both
    # arms have to stay measurable.
    "fp32_dest_acc_8bit": None,
    # R7: the bfloat8_b PACKER. `None` = the shipped gate (precise only for an
    # fp32 input, /numeric-formats-metal 1.7), 1 = precise everywhere, 0 = fast
    # everywhere. The one purely-perf knob of the three numeric ones: precise
    # rounds the block-float mantissas and costs an extra pack pass, fast
    # truncates. No effect off a bfloat8_b output.
    "bfp8_precise": None,
    # R7 (A5b / master.md B11): the narrow-stick STAGED read. `None` = the
    # shipped auto-detect (armed exactly where the per-stick L1 destination
    # would be off the DRAM alignment grid), 0 = the direct read everywhere.
    # The off-arm is NUMERICALLY WRONG on the shapes that need staging (it is
    # the bug this refinement fixed, and it trips the watcher under --dev) — it
    # exists so the COST of alignment-correctness is a measured number rather
    # than an unknown.
    "stage_reads": None,
    "stub_read": 0,  # ablation: drop the NoC read payload
    "stub_compute": 0,  # ablation: drop the tilize math
    "stub_write": 0,  # ablation: drop the NoC write payload
}


def resolve_levers(levers=None) -> dict:
    """Merge caller overrides onto DEFAULT_LEVERS, rejecting unknown knobs."""
    resolved = dict(DEFAULT_LEVERS)
    if levers:
        unknown = set(levers) - set(DEFAULT_LEVERS)
        if unknown:
            raise ValueError(f"tilize: unknown lever knob(s) {sorted(unknown)}")
        resolved.update(levers)
    return resolved


# Fallback per-core CB budget (bytes) used only when the device cannot be
# queried for its real unreserved-L1 size. Depth-2 auto-falls back to depth-1
# rather than OOMing (master.md C16 / the ttnn.concat precedent).
_CB_BUDGET_FALLBACK_BYTES = 400 * 1024
# Share of the per-core unreserved L1 this op is willing to spend on CBs. Not
# 1.0: on the sharded paths the same L1 also holds the shard buffers, and the
# output tensor itself when the caller asks for L1 interleaved.
_CB_BUDGET_L1_FRACTION = 0.5


def _round_up(value: int, multiple: int) -> int:
    """`tt::round_up`. Local because this build does not export ttnn.round_up."""
    return ((value + multiple - 1) // multiple) * multiple


def _div_up(a: int, b: int) -> int:
    """`tt::div_up`. Local because this build does not export ttnn.div_up."""
    return (a + b - 1) // b


def wt_block_max(elem_size: int, target_read_bytes: int = TARGET_READ_BYTES) -> int:
    """Max tiles per compute block for this element size.

    `max(2, ...)` keeps row_bytes >= 64 B (2 x the 32 B DRAM read-alignment
    unit) even for 1-byte dtypes.
    """
    return max(2, target_read_bytes // (TILE_WIDTH * elem_size))


def _unreserved_l1_bytes() -> int:
    """Per-core unreserved L1, queried from the device.

    The module constant is the fallback when the binding is absent. Only ever
    used to decide whether depth-2 fits (never to size a CB), so a conservative
    value degrades to depth-1 rather than to a wrong CB.
    """
    unreserved = getattr(ttnn, "get_max_worker_l1_unreserved_size", None)
    if unreserved is not None:
        try:
            return int(unreserved())
        except Exception:  # pragma: no cover - the query is best-effort
            pass
    return int(_CB_BUDGET_FALLBACK_BYTES / _CB_BUDGET_L1_FRACTION)


def l1_bytes_per_core(tensor, num_l1_banks: int) -> int:
    """Per-core L1 the TENSOR ITSELF occupies — L1 the CBs cannot have.

    - DRAM tensor -> 0 (it costs no worker L1 at all).
    - L1 INTERLEAVED -> its pages are round-robined over the L1 banks (one bank
      per worker core), so the worst-case per-core footprint is
      `ceil(pages / banks) * aligned_page_size`.
    - L1 SHARDED -> 0 **by design**, not by omission: the sharded path (op_design
      §4.2, Refinement 2) aliases the CB directly onto the shard buffer, so the
      shard's L1 *is* the CB's L1 and counting it would double-count.

    This is what makes the depth-2 decision below honest on the `dram_to_l1` /
    `l1_to_l1` / `l1_to_dram` directions: `get_max_worker_l1_unreserved_size()`
    is a static device property, so without this subtraction a large
    L1-interleaved operand would be invisible to the fallback.
    """
    memory_config = tensor.memory_config()
    if memory_config.buffer_type != ttnn.BufferType.L1 or memory_config.is_sharded():
        return 0
    return _div_up(tensor.buffer_num_pages(), max(1, num_l1_banks)) * tensor.buffer_aligned_page_size()


def cb_budget_bytes(unreserved_bytes: int, l1_resident_bytes: int) -> int:
    """Per-core L1 bytes this op may spend on CBs.

    Two bounds, whichever is tighter: a fixed share of unreserved L1 (headroom
    for fragmentation and anything else the program allocates), and what is
    actually left once the L1-resident operands are paid for.
    """
    return max(0, min(int(unreserved_bytes * _CB_BUDGET_L1_FRACTION), unreserved_bytes - l1_resident_bytes))


def cb_depth_for(*, want_depth2: bool, depth2_bytes: int, budget_bytes: int) -> int:
    """The CB-depth knob's value: 2 only when the caller asked for it AND it
    fits the budget. Pure so the L1 fallback is unit-testable without a device."""
    return 2 if (want_depth2 and depth2_bytes <= budget_bytes) else 1


def tile_geometry(shape, tile_height: int):
    """Alignment-aware tile geometry (op_design.md §5.1).

    `ceil` and per-image from the start: in TILE layout each image is tile-padded
    independently, so nt_h = nimg * ceil(H/tile_h), NOT floor(nimg*H/tile_h).
    """
    shape = list(shape)
    H = shape[-2] if len(shape) >= 2 else 1
    W = shape[-1] if len(shape) >= 1 else 1
    Hp = _round_up(H, tile_height)
    Wp = _round_up(W, TILE_WIDTH)
    nimg = math.prod(shape[:-2]) if len(shape) > 2 else 1
    nt_h = nimg * (Hp // tile_height)
    Wt = Wp // TILE_WIDTH
    return nt_h, Wt, Hp, Wp


# ---------------------------------------------------------------------------
# Pad plan (Refinement 5 — the padded path, op_design.md §8.3)
# ---------------------------------------------------------------------------
#
# Padding changes exactly TWO things and nothing else:
#   (a) the GEOMETRY the whole op is blocked over becomes the PADDED shape (the
#       output's tile grid is the pad target's, not the input's), and
#   (b) the reader gains a CT-selected second body that fills the pad positions
#       in L1 before reading the real sub-rectangle over the top.
# Everything downstream — blocking, core split, CBs, compute, writer — is
# unchanged and unaware, because it only ever sees the padded tile grid.
#
# `pad_plan` is the ONE place a `(output_padded_shape, pad_value)` pair becomes
# numbers, so validate(), the output allocation and the kernel args can never
# disagree about which cell a padded call lands in.

_FLOAT_DTYPES = (ttnn.bfloat16, ttnn.float32)
# Block-float formats: 16 datums share one exponent, so there is no fixed
# per-datum size and `Tensor.element_size()` RAISES on them. Output-only for
# this op (a ROW_MAJOR input can never be block-float).
_BLOCK_FLOAT_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat4_b)


def element_size_or_zero(tensor) -> int:
    """`tensor.element_size()`, or 0 for a block-float dtype where it raises.

    0 is a sentinel, never an arithmetic value: the only consumers are R7's
    output-format pad rewrite (which excludes block-float structurally — raw
    stores cannot address a shared-exponent tile) and the CT arg feeding it.
    """
    return 0 if tensor.dtype in _BLOCK_FLOAT_DTYPES else int(tensor.element_size())


def numeric_policy(in_dtype, out_dtype, in_elem_size, levers=None) -> dict:
    """The op's whole dtype surface, decided in ONE place (R7 / A4 + A5b).

    tilize has no arithmetic and therefore no `ComputeKernelConfig` surface to
    expose — but it does have three numeric-format decisions, and each of them
    is a *pair* of host config + kernel template argument that must agree or the
    helper's own static_asserts fire:

      needs_cast        out != in  -> `UnpackAndPackReconfigure`, else
                        `NoReconfigure` (a ~150 ns fixed cost with nothing to
                        cast — op_design §7.1).
      fp32_lossless     fp32 -> fp32 must be BIT-EXACT (identity is the op's
                        contract), which the tilize helper only guarantees under
                        `Fp32Mode::Lossless` + `fp32_dest_acc_en` +
                        `UnpackToDestFp32` on the input CB. All three, or none:
                        the helper static_asserts the other two whenever the
                        first is set, and conversely asserts that FAST fp32
                        tilize does NOT see `UnpackToDestFp32`.
      fp32_dest_acc     also forced by an 8-bit input, where it is a CORRECTNESS
                        lever rather than a precision one (see FP32_DEST_ACC_8BIT).
      bfp8_pack_precise only for fp32 -> bfp8_b. The fast packer clears the
                        golden PCC gate for bf16 -> bfp8_b and costs ~1.4x less;
                        gating on the INPUT dtype is the /numeric-formats-metal
                        rule, not a blanket "precise for all block-float".

    Returns the four host-side values plus the one CT flag the compute kernel
    needs, so the descriptor never re-derives any of them.
    """
    lv = levers if levers is not None else {}
    lossless_knob = lv.get("fp32_lossless")
    if lossless_knob is None:
        lossless_knob = FP32_LOSSLESS_IDENTITY
    dest_acc_8bit_knob = lv.get("fp32_dest_acc_8bit")
    if dest_acc_8bit_knob is None:
        dest_acc_8bit_knob = FP32_DEST_ACC_8BIT

    fp32_lossless = bool(lossless_knob) and in_dtype == ttnn.float32 and out_dtype == ttnn.float32
    # Keyed off the ELEMENT SIZE, not a dtype list: the constraint is the DEST
    # width a 1-byte datum needs, so int8 / fp8 inherit it without an edit.
    dest_acc_8bit = bool(dest_acc_8bit_knob) and in_elem_size == 1
    precise_knob = lv.get("bfp8_precise")
    # The shipped gate is the INPUT dtype: bf16 -> bfp8_b clears the golden PCC
    # gate on the fast packer (measured 0.99997), so only a wide fp32 input pays.
    _precise_pack = (in_dtype == ttnn.float32) if precise_knob is None else bool(precise_knob)
    return {
        "needs_cast": int(out_dtype != in_dtype),
        "fp32_lossless": int(fp32_lossless),
        # `UnpackToDestFp32` is EXCLUSIVE with the fast tilize path, so it is
        # tied to `fp32_lossless` and never set on its own.
        "unpack_to_dest_fp32": int(fp32_lossless),
        "fp32_dest_acc_en": bool(fp32_lossless or dest_acc_8bit),
        "bfp8_pack_precise": bool(out_dtype == ttnn.bfloat8_b and _precise_pack),
    }


def auto_padded_shape(shape, tile_height: int = DEFAULT_TILE_HEIGHT):
    """The padded shape `pad_mode="auto"` implies: the last two dims rounded UP
    to the tile grid, everything else untouched.

    Degenerate ranks SYNTHESIZE their tile dims rather than rounding them (a
    scalar has no tile dims of its own): rank 0 -> one tile, rank 1 -> one
    tile-row. `tile_height` (not a hardcoded 32) drives the H axis so a tiny-tile
    call rounds to the tile it actually asked for.
    """
    shape = list(shape)
    if len(shape) == 0:
        return [tile_height, TILE_WIDTH]
    if len(shape) == 1:
        return [tile_height, _round_up(shape[0], TILE_WIDTH)]
    out = list(shape)
    out[-2] = _round_up(out[-2], tile_height)
    out[-1] = _round_up(out[-1], TILE_WIDTH)
    return out


# ---------------------------------------------------------------------------
# Output allocation (Refinement 8 / T1 — the tile geometry has to reach the
# BUFFER, not only the CBs)
# ---------------------------------------------------------------------------


def output_tensor_spec(shape, dtype, memory_config, tile_height: int):
    """The output's `TensorSpec`, carrying the requested tile geometry.

    `ttnn.allocate_tensor_on_device(shape, dtype, layout, device, mem_config)`
    builds its spec through `PageConfig(layout)` with NO tile, i.e. always the
    default 32x32 one. A tiny-tile call would then get an output buffer whose
    page is a 32-row tile while every CB, kernel arg and LLK in this op used
    `tile_height` rows — a silent geometry disagreement, not an error. Only the
    `TensorSpec` overload can carry `tile=`, so the allocation goes through it.

    ONE path, not two: the same call builds the 32x32 spec, so the default
    geometry cannot drift away from the tiny-tile one. The three constructor
    overloads mirror the three placements `_spec_from_memory_config` already
    distinguishes (interleaved / legacy 2-D shard / nd shard).
    """
    tile = ttnn.Tile([tile_height, TILE_WIDTH])
    if not memory_config.is_sharded():
        return ttnn.TensorSpec(shape, dtype, ttnn.TILE_LAYOUT, memory_config.buffer_type, tile)
    if memory_config.memory_layout in LEGACY_SHARD_SCHEMES:
        return ttnn.TensorSpec(
            shape,
            dtype,
            ttnn.TILE_LAYOUT,
            memory_config.memory_layout,
            memory_config.shard_spec,
            memory_config.buffer_type,
            tile,
        )
    return ttnn.TensorSpec(
        shape,
        dtype,
        ttnn.TILE_LAYOUT,
        memory_config.nd_shard_spec,
        memory_config.buffer_type,
        tile,
    )


def _f32_to_bf16_bits(bits: int) -> int:
    """fp32 bit pattern -> bf16 bit pattern, round-to-nearest-EVEN.

    RNE, not truncation, because the oracle materializes the fill with torch's
    own float->bfloat16 conversion; a truncating pack differs from it by one ulp
    on half the fill values and shows up as a nonzero-fill-only mismatch.
    """
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF


def pad_fill_word(value, dtype, elem_size: int) -> int:
    """The 32-bit L1 store word the reader writes into pad positions.

    Two rules, both of them the bug the golden `pad_value` sign buckets exist to
    catch (op_design.md §8.3, risks 5 and 6):

      - packed in the INPUT element format, never the output's. A `dtype=` cast
        happens later, at pack time, on data that already carries the fill.
      - REPLICATED across the whole 32-bit word (2x for bf16/uint16, 4x for
        uint8, 1x for fp32/uint32). The reader stores words, so a fill written
        once leaves every other pad element stale — invisible at pad_value=0.0
        and garbage on any nonzero fill.

    A negative integer fill goes through the two's-complement mask, which is the
    signed->unsigned bit_cast the design names.
    """
    mask = (1 << (8 * elem_size)) - 1
    if dtype in _FLOAT_DTYPES:
        f32 = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        raw = _f32_to_bf16_bits(f32) if elem_size == 2 else f32
    else:
        raw = int(value)
    raw &= mask
    word = 0
    for k in range(max(1, 4 // elem_size)):
        word |= raw << (8 * elem_size * k)
    return word & 0xFFFFFFFF


def format_roundtrip(value, dtype, elem_size: int):
    """The value an L1 element of `dtype` actually holds after packing `value`.

    The fill's precision oracle: `pad_cast_fix` compares the input format's
    round-trip against the output format's to decide whether the pad positions
    are worth rewriting at output precision.
    """
    if dtype in _FLOAT_DTYPES:
        f32 = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        if elem_size == 2:
            return struct.unpack("<f", struct.pack("<I", _f32_to_bf16_bits(f32) << 16))[0]
        return float(struct.unpack("<f", struct.pack("<I", f32))[0])
    mask = (1 << (8 * elem_size)) - 1
    return int(value) & mask


def pad_cast_fix(pad, in_dtype, in_elem_size, out_dtype, out_elem_size):
    """R7: does this padded call need its pad positions rewritten in the OUTPUT
    format, and with which word?

    The reader's fill lands in the INPUT circular buffer and is therefore packed
    in the INPUT format (op_design.md §8.3 — correct, and unavoidable: the CB
    carries the input tensor's raw bytes). On a WIDENING cast that rounds the
    caller's number before the output ever sees it —
    `tilize(bf16, dtype=float32, pad_value=10.2)` produced 10.1875, one bf16 ulp
    off the fp32 10.2 both the caller and the golden oracle ask for.

    Armed ONLY when the output format holds the fill STRICTLY better than the
    input's does, so:
      - same dtype, or any integer pair -> off (both round-trips are identical);
      - a NARROWING cast -> off (the pack already lands on the output's own
        nearest value, which is what the oracle compares against);
      - an exactly-representable fill (0, 10.0, 42.0, 3.5, -18.0, -32.5) -> off,
        even on a widening cast. The rewrite is paid for by the ONE case that
        needs it, never by the padded path at large.
    Block-float output is excluded structurally: a bfp8_b tile has shared
    exponents, so its pad positions cannot be written with raw stores at all —
    and it is a narrowing cast anyway, so the gate below never reaches it.
    """
    off = (0, 0)
    if pad is None or not pad["enabled"] or out_dtype == in_dtype:
        return off
    if out_dtype == ttnn.bfloat8_b:
        return off
    value = pad["pad_value"]
    held_in = format_roundtrip(value, in_dtype, in_elem_size)
    held_out = format_roundtrip(value, out_dtype, out_elem_size)
    if held_out == held_in or abs(held_out - value) >= abs(held_in - value):
        return off
    return 1, pad_fill_word(value, out_dtype, out_elem_size)


def pad_plan(input_shape, tile_height: int, output_padded_shape, pad_value, dtype, elem_size: int):
    """Resolve a padded call into the numbers the op needs, or None.

    Returns None when the caller asked for no padding at all (the Track A hot
    path — `pad` stays None all the way down and every branch below is dead
    code the compiler never sees).

    `enabled` is the CT flag: it is False for a *degenerate* pad — `pad_mode=
    "auto"` on an already-aligned input, where the target equals the input shape
    — so that cell takes the aligned reader byte-for-byte rather than a pad
    reader that happens to fill nothing. That is what makes "the aligned path is
    unchanged" structural rather than a convention.
    """
    if output_padded_shape is None and pad_value is None:
        return None
    padded = (
        [int(d) for d in output_padded_shape]
        if output_padded_shape is not None
        else auto_padded_shape(input_shape, tile_height)
    )
    # rank 0/1 have no tile dims of their own; the pad target supplies them, so
    # the logical view is left-expanded with 1s to meet it.
    logical = [int(d) for d in input_shape]
    if len(logical) < len(padded):
        logical = [1] * (len(padded) - len(logical)) + logical
    return {
        "padded_shape": padded,
        "logical_shape": logical,
        "enabled": padded != logical,
        # The raw fill, kept alongside the packed word so `pad_cast_fix` can ask
        # what the OUTPUT format would hold (R7).
        "pad_value": pad_value if pad_value is not None else 0,
        "pad_word": pad_fill_word(pad_value if pad_value is not None else 0, dtype, elem_size),
        # Reader-side geometry. `hp` is the PADDED rows per image and `h_real`
        # the real ones, so a row is real iff `img < nimg_real and row < h_real`
        # — which covers the H tail and whole pad tile-ROWS with one test.
        "hp": padded[-2],
        "h_real": logical[-2],
        "nimg_real": math.prod(logical[:-2]),
        "real_row_bytes": logical[-1] * elem_size,
    }


def blocking(
    shape,
    tile_height: int,
    elem_size: int,
    target_read_bytes: int = TARGET_READ_BYTES,
    wt_block_override: int | None = None,
):
    """The whole Blocking Model for one call, derived from the knobs above.

    `wt_block_override` is the ONE place the block-width knob can be set by
    something other than the byte target: on a sharded side the SHARD hands you
    the block width (op_design.md §6.2 — a narrower block would need a strided
    CB page, which a shard alias cannot express), so the placement plan passes
    `wt_shard` here rather than restating the blocking arithmetic.
    """
    nt_h, Wt, _, _ = tile_geometry(shape, tile_height)
    wt_block = (
        wt_block_override if wt_block_override is not None else min(Wt, wt_block_max(elem_size, target_read_bytes))
    )
    n_wchunks = _div_up(Wt, wt_block)
    wt_tail = Wt - (n_wchunks - 1) * wt_block
    total_blocks = nt_h * n_wchunks
    tail_block_start = (n_wchunks - 1) * nt_h
    return {
        "nt_h": nt_h,
        "Wt": Wt,
        "wt_block": wt_block,
        "wt_tail": wt_tail,
        "n_wchunks": n_wchunks,
        "total_blocks": total_blocks,
        "tail_block_start": tail_block_start,
    }


# ---------------------------------------------------------------------------
# Placement plan (Refinement 2 — sharded I/O)
# ---------------------------------------------------------------------------
#
# A shard is NOT a memory_config value the interleaved path tolerates: the shard
# is the per-core BLOCK, already resident in that core's L1 (op_design.md §4.1).
# So the placement plan picks, per side, between
#
#   RESIDENT — the CB is *aliased onto the shard buffer*
#              (`ttnn.cb_descriptor_from_sharded_tensor`); reader/writer
#              degenerate to the CB handshake and move ZERO bytes.
#   STREAMED — the existing `TensorAccessor` path (interleaved, or a shard this
#              core does not own — the cross-spec / DRAM-shard cases).
#
# and the three modes below fall out of that choice. `wt_block` is the shard's
# own width on any resident side: the shard hands you the block.

LEGACY_SHARD_SCHEMES = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)

MODE_STREAMED = "streamed"  # both sides through TensorAccessor (Phase 0 path)
MODE_RESIDENT = "resident"  # same-spec L1 shards: both CBs aliased, zero NoC
MODE_CROSSOVER_IN = "crossover_in"  # input shard resident, output streamed
MODE_CROSSOVER_OUT = "crossover_out"  # output shard resident, input streamed
# Refinement 4 (A3c) — CROSS-SPEC reshard: both sides sharded, different specs.
# The shard is still the per-core block, but it is a DIFFERENT block on each
# side, so one side is resident and the other is gathered across cores.
MODE_RESHARD_OUT = "reshard_out"  # output shard resident: each output core PULLS
MODE_RESHARD_IN = "reshard_in"  # input shard resident: the pull's mirror image

# Which modes leave each side's CB aliased onto its shard. Stated once so a new
# mode cannot half-register itself.
RESIDENT_IN_MODES = (MODE_RESIDENT, MODE_CROSSOVER_IN, MODE_RESHARD_IN)
RESIDENT_OUT_MODES = (MODE_RESIDENT, MODE_CROSSOVER_OUT, MODE_RESHARD_OUT)


def shard_view(memory_config):
    """`(grid, (shard_h, shard_w), orientation)` in the folded 2-D view.

    ONE view over both sharding APIs, and it must work on a *caller-constructed*
    MemoryConfig as well as a live tensor's: a constructed nd config answers
    `shard_spec is None` (only the live one carries the 2-D projection), and a
    constructed legacy config answers `nd_shard_spec is None`. So take whichever
    form is present and fold the nd shard shape onto 2-D the same way
    `tile_geometry` folds the tensor shape.
    """
    if not memory_config.is_sharded():
        return None
    shard_spec = memory_config.shard_spec
    if shard_spec is not None:
        h, w = tuple(shard_spec.shape)
        return shard_spec.grid, (int(h), int(w)), shard_spec.orientation
    nd = memory_config.nd_shard_spec
    shard_shape = [int(d) for d in nd.shard_shape]
    return nd.grid, (math.prod(shard_shape[:-1]), shard_shape[-1]), nd.orientation


def shard_2d(memory_config):
    """The shard's `(h, w)` in the folded 2-D view, or None when interleaved."""
    view = shard_view(memory_config)
    return None if view is None else view[1]


def shard_identity(memory_config):
    """Everything that decides WHICH core holds WHICH region.

    2-D form only, because that is the one form BOTH APIs always answer: a
    caller-constructed legacy MemoryConfig has `nd_shard_spec is None` while the
    live tensor's has it filled in, so keying on the nd form would make a spec
    unequal to itself. `same_shard_placement` compares the nd form separately,
    and only when both sides expose one.
    """
    view = shard_view(memory_config)
    if view is None:
        return None
    grid, shape, orientation = view
    # NOT keyed on `memory_layout`: the same nd spec reports ND_SHARDED as a
    # caller-constructed MemoryConfig and BLOCK_SHARDED once a tensor is built
    # from it, so keying on it would make a spec unequal to itself (measured).
    # Grid + folded shard shape + orientation is what places the data; the nd
    # form (compared separately) is what separates the two APIs.
    return (str(memory_config.buffer_type), str(grid), shape, str(orientation))


def _nd_identity(memory_config):
    try:
        nd = memory_config.nd_shard_spec
    except Exception:  # pragma: no cover - defensive
        return None
    if nd is None:
        return None
    return (tuple(nd.shard_shape), str(nd.grid), str(nd.orientation))


def same_shard_placement(a, b) -> bool:
    """True when two sharded specs put the same region on the same core.

    That — not "both are sharded" — is what makes a pair zero-copy-able: the
    input shard and the output shard must be the same logical block, or the
    core would tilize its own rows into someone else's tiles.
    """
    if shard_identity(a) != shard_identity(b) or shard_identity(a) is None:
        return False
    nd_a, nd_b = _nd_identity(a), _nd_identity(b)
    return nd_a is None or nd_b is None or nd_a == nd_b


def shard_residency(memory_config, *, tile_height: int):
    """`(nt_h_shard, wt_shard)` when this shard can BE the per-core block.

    Requires the shard to live in L1 (a DRAM shard is not resident in any
    worker's L1, so there is nothing to alias) and to be a whole number of
    tiles in both dimensions — a partial tile-row cannot be a tilize block.
    """
    geometry = shard_2d(memory_config)
    if geometry is None or memory_config.buffer_type != ttnn.BufferType.L1:
        return None
    h, w = geometry
    if h % tile_height or w % TILE_WIDTH:
        return None
    return h // tile_height, w // TILE_WIDTH


def shard_folds_contiguously(memory_config, shape) -> bool:
    """Does this shard cover a CONTIGUOUS run of folded tile-rows?

    `shard_view` folds an nd shard onto 2-D as `(prod(dims[:-1]), dims[-1])`,
    which is only the truth when the shard's leading dims either select a single
    image (`prod == 1`) or span each image's whole height. Otherwise the shard
    holds rows `{i*H + h}` for several `i` and a strict subset of `h` — several
    DISJOINT runs of the folded index, so it cannot be one core's contiguous
    `[b0, b0+nb)` block range.

    Only the crossover / reshard modes need this: they place a shard at a
    position in the global linearization. Same-spec RESIDENT does not — there
    each core tilizes whatever sits in its own L1 and never names a global block
    — and the streamed path does not either (the accessor addresses by page id).
    A legacy 2-D shard is already stated in the folded view, so it is always
    contiguous.
    """
    nd = _nd_identity(memory_config)
    if nd is None:
        return True
    shard_shape = list(nd[0])
    if len(shard_shape) < 3:
        return True
    H = shape[-2] if len(shape) >= 2 else 1
    return math.prod(shard_shape[:-2]) == 1 or shard_shape[-2] == H


def source_bands(memory_config, W: int, elem_size: int):
    """The READ side's page geometry: `(n_bands, band_bytes)`, or None.

    A tensor's page is ONE ROW OF ITS SHARD — `shard_w * elem` bytes for a
    sharded tensor, the whole row (`W * elem`) for an interleaved one. So the
    source page grid is `[folded_row][band]` with `n_bands = ceil(W / shard_w)`,
    and the page id of `(folded_row, band)` is `folded_row * n_bands + band`.
    Measured, not assumed (`probes/probe_012.py`): a `(64,128)` width shard of
    `[1,1,64,512]` reports page_size 256 B and 256 pages = 64 rows x 4 bands;
    an nd `(2,64,96)` shard of `[7,128,128]` reports 192 B and 1792 = 896 x 2.

    Refinement 4 (A3c): this is what lets an output core PULL from a source
    whose shard is NARROWER than a row — the read is split at band boundaries
    and each segment is one NoC read from whichever core holds that band. Before
    R4 the reader could only index whole-row pages, so such an input was refused.

    Returns None when a band is not a whole number of tile-columns: the segment
    boundaries are then not tile-column boundaries, so a segment length would
    not be a multiple of `TILE_WIDTH * elem` and the L1 destination cursor would
    walk off the NoC alignment grid.
    """
    row_bytes = W * elem_size
    geometry = shard_2d(memory_config)
    if geometry is None:
        return 1, row_bytes
    shard_w = geometry[1]
    if shard_w % TILE_WIDTH:
        return None
    if shard_w >= W:
        return 1, row_bytes
    return _div_up(W, shard_w), shard_w * elem_size


def reshard_direction_is_pull(*, in_memory_config, out_memory_config, band_bytes: int, out_tile_bytes: int) -> bool:
    """Which side of a CROSS-SPEC reshard should hold the resident block.

    Measured on `[1,1,1024,1024]` bf16 L1->L1 (R4 bench, `DEVICE KERNEL
    DURATION`; the ratio is the loser over the winner):

    | input placement     | output placement    | pull cores | push cores | winner      |
    |---------------------|---------------------|-----------:|-----------:|-------------|
    | WIDTH  (1024,128)@8 | HEIGHT (128,1024)@8 |          8 |          8 | push 1.566x |
    | HEIGHT (128,1024)@8 | HEIGHT (256,1024)@4 |          4 |          8 | push 1.359x |
    | HEIGHT (256,1024)@4 | HEIGHT (128,1024)@8 |          8 |          4 | pull 1.615x |

    Rows 2 and 3 are the SAME pair reversed and they swap the winner, so the
    first term is not the direction at all — it is the CORE COUNT. The resident
    side's shard grid *is* the core set the program launches on (master.md A2),
    so making the smaller grid resident throws away parallelism, whichever way
    the bytes then travel.

    Row 1 holds the core count equal and isolates the second term, the
    transaction shape: a pull reads ONE BAND of a row (256 B there), while a
    push always writes a whole tile page (2048 B), because an output tile is one
    page no matter how the output is sharded. At equal parallelism the coarser
    transaction wins.

    `op_design` §4.3 named pull, and its stated reason — §1.1's bijection, so no
    fan-out and no semaphore — is TRUE OF BOTH DIRECTIONS and therefore does not
    discriminate between them. This gate keeps pull everywhere pull is not
    measurably worse, and is the same shape as R3's `placement_defaults`: a knob
    whose two values have opposite signs on two real geometries cannot have a
    global default.
    """
    in_cores = shard_view(in_memory_config)[0].num_cores()
    out_cores = shard_view(out_memory_config)[0].num_cores()
    if in_cores != out_cores:
        return out_cores > in_cores
    return band_bytes >= out_tile_bytes


def plan_placement(
    *,
    shape,
    tile_height: int,
    in_memory_config,
    out_memory_config,
    Wt: int,
    nt_h: int,
    in_tile_bytes: int,
    out_tile_bytes: int,
    cb_budget_bytes: int | None = None,
    force_streamed: bool = False,
    reshard_pull: "bool | None" = None,
    in_shape=None,
    pad_enabled: bool = False,
):
    """Choose RESIDENT / STREAMED per side. Pure — no tensors, no device.

    Returns a dict with `mode`, `wt_block` (None = use the byte-target clamp),
    the shard geometry the core planner needs, and `error` (a support-gap
    message) when neither mechanism can address the call.

    `shape` is the OUTPUT (padded) tile grid — that is what the blocking and the
    output shards are stated in. `in_shape` is the input's own logical shape,
    which differs only on the padded path and is what the READ side is stated in
    (its pages are real rows of real width). They are the same object off the
    pad path, so nothing before Refinement 5 can notice the split.
    """
    in_shape = shape if in_shape is None else in_shape
    in_sharded = in_memory_config.is_sharded()
    out_sharded = out_memory_config.is_sharded()
    # A padded call can never consume the INPUT shard in place: the pad
    # positions live inside the block, and a resident reader would have to write
    # the fill into the CALLER'S OWN input tensor to place them. So padding
    # disqualifies input residency (never output residency — an output tile is
    # whole, and compute packs the fill into it like any other datum).
    no_resident_in = force_streamed or pad_enabled
    in_res = None if no_resident_in else shard_residency(in_memory_config, tile_height=tile_height)
    out_res = None if force_streamed else shard_residency(out_memory_config, tile_height=tile_height)

    # The read side's page geometry (R4). `in_tile_bytes` is by construction
    # `tile_height * TILE_WIDTH * elem_size`, so it carries the element size
    # without a second parameter that could disagree with it.
    elem_size = max(1, in_tile_bytes // (tile_height * TILE_WIDTH))
    bands = source_bands(in_memory_config, in_shape[-1] if len(in_shape) else 1, elem_size)
    if pad_enabled and bands is not None and bands[0] > 1:
        # The padded reader assembles each stick from ONE source page (it has to
        # clamp the read at the real row width); a banded source would need the
        # clamp applied per segment. Nothing in TARGET reaches this — a padded
        # cell whose input shard is narrower than a row — so it is refused
        # rather than guessed at.
        return {
            "mode": None,
            "wt_block": None,
            "shard": None,
            "sharded_side": None,
            "bands": None,
            "error": (
                "tilize: a padded call cannot read a source sharded narrower than a row "
                f"({bands[0]} bands per row) — the pad clamp is stated per source page"
            ),
        }

    def _plan(mode, wt_block=None, shard=None, sharded_side=None):
        return {
            "mode": mode,
            "wt_block": wt_block,
            "shard": shard,
            "sharded_side": sharded_side,
            "bands": bands,
            "error": None,
        }

    # (a) Same-spec L1 -> L1: BOTH CBs alias their shard, zero DRAM traffic on
    #     both sides. No core needs to know which shard it holds — it tilizes
    #     the block sitting in its own L1 — so orientation is a non-issue here.
    if in_res is not None and out_res is not None and same_shard_placement(in_memory_config, out_memory_config):
        nt_h_shard, wt_shard = in_res
        return _plan(
            MODE_RESIDENT,
            wt_block=wt_shard,
            shard={"nt_h_shard": nt_h_shard, "wt_shard": wt_shard},
            sharded_side="in",
        )

    # (b) ONE side resident, the other streamed. Two schemes share this shape:
    #
    #     CROSSOVER (A3b, Refinement 2) — exactly one side sharded. That side is
    #       a CB alias pinned to its own cores; the other keeps its accessor.
    #     RESHARD (A3c, Refinement 4) — BOTH sides sharded with different specs.
    #       The shard is still the per-core block, but it is a different block on
    #       each side, so a core must touch bytes another core owns. `op_design`
    #       §4.3 pins the topology: **pull, not push** — the OUTPUT shard is the
    #       resident side, each output core owns its output block and reads the
    #       input pages it needs from whichever core holds them, over L1->L1. No
    #       semaphore, no multicast (§1.1: the map is a bijection, nothing to fan
    #       out), no DRAM staging. `MODE_RESHARD_IN` is the mirror image, taken
    #       only for the geometries the output side cannot express.
    #
    # Either way the resident core's block range is its shard's position in the
    # linearization `b = wchunk*nt_h + r` — contiguous, because the
    # linearization is column-block-major.
    if in_sharded and out_sharded:
        # The `reshard_pull` knob picks which side is TRIED FIRST; the other stays
        # as the fallback for a geometry the preferred side cannot express (an nd
        # shard that does not fold contiguously, an indivisible shard grid, ...).
        # `None` = the measured per-geometry gate.
        pull = (
            reshard_direction_is_pull(
                in_memory_config=in_memory_config,
                out_memory_config=out_memory_config,
                band_bytes=bands[1] if bands is not None else 0,
                out_tile_bytes=out_tile_bytes,
            )
            if reshard_pull is None
            else bool(reshard_pull)
        )
        candidates = [("out", out_res, out_memory_config), ("in", in_res, in_memory_config)]
        if not pull:
            candidates.reverse()
    elif in_sharded:
        candidates = [("in", in_res, in_memory_config)]
    elif out_sharded:
        candidates = [("out", out_res, out_memory_config)]
    else:
        candidates = []

    for side, res, mc in candidates:
        if res is None:
            continue
        nt_h_shard, wt_shard = res
        if nt_h % nt_h_shard or Wt % wt_shard:
            continue
        # This mode places the shard at a POSITION in the global linearization,
        # so the shard must be one contiguous run of folded tile-rows. Each side
        # is measured against the shape IT is stated in (see `in_shape` above).
        if not shard_folds_contiguously(mc, in_shape if side == "in" else shape):
            continue
        n_sh_rows = nt_h // nt_h_shard
        n_sh_cols = Wt // wt_shard
        # Shard k -> (row-block, column-block) is row-major over the shard grid.
        # That is unambiguous for a 1-D shard grid (HEIGHT / WIDTH) and for
        # ROW_MAJOR; a COL_MAJOR 2-D grid is left to the streamed path rather
        # than guessed at.
        if not (n_sh_rows == 1 or n_sh_cols == 1 or shard_view(mc)[2] == ttnn.ShardOrientation.ROW_MAJOR):
            continue
        # The STREAMED side is the other one. When the input streams it must be
        # band-addressable (R4); a streamed output is always tile-paged.
        if side == "out" and bands is None:
            continue
        # A3d: the resident side costs no extra L1 (it IS the shard), but the
        # STREAMED side's CB is `wt_shard` pages — so a wide shard would grow the
        # CB with W. When it no longer fits the budget, fall back to the
        # fully-streamed path, whose WT_BLOCK is clamped by the byte target and
        # therefore constant in W — but only when that fallback can address the
        # input at all.
        streamed_tile_bytes = out_tile_bytes if side == "in" else in_tile_bytes
        fits = cb_budget_bytes is None or wt_shard * streamed_tile_bytes <= cb_budget_bytes
        if not fits and bands is not None:
            continue
        if in_sharded and out_sharded:
            mode = MODE_RESHARD_OUT if side == "out" else MODE_RESHARD_IN
        else:
            mode = MODE_CROSSOVER_OUT if side == "out" else MODE_CROSSOVER_IN
        return _plan(
            mode,
            wt_block=wt_shard,
            shard={
                "nt_h_shard": nt_h_shard,
                "wt_shard": wt_shard,
                "n_sh_rows": n_sh_rows,
                "n_sh_cols": n_sh_cols,
            },
            sharded_side=side,
        )

    # (c) Everything else streams through TensorAccessor: interleaved on both
    #     sides (Phase 0), and the cross-spec / DRAM-shard cases where the bytes
    #     genuinely live on another core or in DRAM.
    if bands is None:
        h, w = shard_2d(in_memory_config)
        return {
            "mode": None,
            "wt_block": None,
            "shard": None,
            "sharded_side": None,
            "bands": None,
            "error": (
                f"tilize: a ROW_MAJOR input sharded {h}x{w} cannot be addressed — its pages are "
                f"partial rows of {w} elements, which is not a whole number of {TILE_WIDTH}-wide "
                "tile columns, so a gathered read would not land on the NoC alignment grid"
            ),
        }
    return _plan(MODE_STREAMED)


def shard_grid_cores(device, tensor):
    """The cores that actually hold shards, in shard order (master.md A2).

    NOT a re-spread `split_work_to_cores` line: a shard's cores are fixed by its
    spec, and core k in this list holds shard k.
    """
    cores = ttnn.get_optimal_worker_cores_for_sharded_tensor(tensor)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    return list(cores), all_cores


def placement_defaults(in_memory_config, out_memory_config):
    """Regime-selected defaults for the two R3 placement knobs.

    Pure (memory configs only, no device, no tensors) so the gate is unit-testable.
    Both knobs are measured to have OPPOSITE signs on the two interleaved data
    paths (numbers next to `MIN_BLOCKS_PER_CORE` / `SPREAD_CORES`), so neither is
    applied globally:

      both operands DRAM  -> spread the cores (route diversity to the DRAM banks),
                             no pipeline cap: DM is the wall and compute is
                             already overlap-hidden, so trading cores for depth
                             buys nothing.
      both operands L1    -> cap for pipeline depth (L1 DM is ~1.7x faster, so
                             compute is CO-BINDING and worth hiding), packed
                             placement: an L1-interleaved operand's banks ARE the
                             worker cores.
      anything else       -> the Phase-0 placement verbatim. A mixed direction is
                             half of each regime and is NOT measured here, so it
                             keeps the shipped behaviour rather than inheriting a
                             guess.

    A sharded side never reaches this: its cores are fixed by the shard spec
    (`shard_grid_cores`), and `plan_cores` is not called at all.
    """
    types = (in_memory_config.buffer_type, out_memory_config.buffer_type)
    sharded = in_memory_config.is_sharded() or out_memory_config.is_sharded()
    all_dram = not sharded and all(t == ttnn.BufferType.DRAM for t in types)
    all_l1 = not sharded and all(t == ttnn.BufferType.L1 for t in types)
    return {
        "min_blocks_per_core": MIN_BLOCKS_PER_CORE if all_l1 else 1,
        "spread_cores": SPREAD_CORES if all_dram else 0,
        "stagger_reads": STAGGER_READS if all_dram else 0,
        "reshard_pull": RESHARD_PULL,
    }


def read_bank_period(num_banks: int, tile_height: int, is_dram_interleaved: bool) -> int:
    """Source pages per bank cycle for the R6 stateful-read arm, or 0 = off.

    An interleaved buffer round-robins its pages over `num_banks` banks, so the
    sticks whose page index is congruent mod `num_banks` share a source NODE and
    one `set_state` covers all of them (master.md B13). Returns 0 — the plain
    per-stick `noc_async_read` path — whenever grouping cannot pay:

      * a SHARDED or L1 source: an L1-interleaved buffer's banks are the worker
        cores (110 here), i.e. more banks than a block has sticks, so every group
        would hold one stick and the state would be re-programmed every read;
        a sharded source is addressed by the R4 band gather instead.
      * `num_banks >= tile_height`: same degeneracy stated on the geometry.
      * `num_banks <= 1`: nothing to group by.

    Pure (ints + a bool) so the gate is unit-testable without a device. The
    period is a PERFORMANCE hint only — the kernel re-programs the state whenever
    an address's node does not match the one it programmed, so a wrong period
    cannot produce a wrong read.
    """
    if not is_dram_interleaved:
        return 0
    return num_banks if 1 < num_banks < tile_height else 0


def pipeline_capped_cores(total_blocks: int, grid_cores: int, min_blocks_per_core: int):
    """The pipeline-depth cap on the core count, or None when it does not bind.

    Returns None whenever the cap would not reduce the core count the plain split
    already uses — which is the whole gate: on a shape that already fills the grid
    with `>= min_blocks_per_core` blocks per core (the square, `l1_to_l1`) this is
    a no-op and the default split is used unchanged. It binds only in the
    grid-fill-deficit regime (`total_blocks < grid_cores * min_blocks_per_core`),
    where the default split would hand a core a pipeline it cannot overlap.
    """
    if min_blocks_per_core <= 1:
        return None
    cap = max(1, total_blocks // min_blocks_per_core)
    default_cores = min(grid_cores, total_blocks)
    return cap if cap < default_cores else None


def spread_core_list(split_grid, num_cores: int, row_wise: bool):
    """`num_cores` cores spread UNIFORMLY over the grid, not packed into a corner.

    master.md A1/A3: DRAM banks sit in a few columns, so the NoC routes a set of
    readers takes depends on WHERE those readers are. `grid_to_cores(n, ...)`
    returns the FIRST n cores of the row-major enumeration — for a partially
    filled grid that is a solid block in the first few rows, i.e. every one of
    those cores reaches DRAM over the same handful of links. Taking every
    `grid/num_cores`-th core of the same enumeration instead keeps the identical
    core COUNT and block assignment and only changes which cores they are.

    A no-op (returns the packed list) once `num_cores == grid`, which is why a
    grid-filling shape cannot be perturbed by this lever.
    """
    grid_total = split_grid.x * split_grid.y
    all_cores = ttnn.grid_to_cores(grid_total, split_grid.x, split_grid.y, row_wise)
    if num_cores >= grid_total:
        return list(all_cores[:num_cores])
    return [all_cores[(k * grid_total) // num_cores] for k in range(num_cores)]


def _materialize_cores(split_grid, num_cores: int, row_wise: bool, spread_cores: bool, per_core_blocks):
    """Turn a core COUNT + per-core block counts into (cores, all_cores, counts).

    The one place a count becomes a core list, so the packed and spread placements
    cannot drift apart in how they build `all_cores`.
    """
    if spread_cores:
        cores = spread_core_list(split_grid, num_cores, row_wise)
    else:
        cores = list(ttnn.grid_to_cores(num_cores, split_grid.x, split_grid.y, row_wise))
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    return cores, all_cores, per_core_blocks


def plan_cores(
    device,
    total_blocks: int,
    *,
    use_multicore: bool,
    row_wise: bool = True,
    max_cores=None,
    min_blocks_per_core: int = 1,
    spread_cores: bool = False,
):
    """Core assignment: `grid_cores` is a PARAMETER whose trivial value is 1.

    Returns (cores, all_cores, per_core_blocks) where per_core_blocks[i] is the
    block count for cores[i] and the list is in the SAME order the split produced
    — `row_wise` MUST match on both calls or the runtime-arg assignment silently
    mismatches the split's group order (master.md A1 / op_design risk 18).
    """
    grid = device.compute_with_storage_grid_size()
    split_grid = grid if use_multicore else ttnn.CoreCoord(1, 1)

    # R3: pipeline depth. Fold the cap into `max_cores` so there is exactly ONE
    # place that turns a core count into an assignment. `None` when it does not
    # bind, which keeps the grid-filling shapes on the default split verbatim.
    pipeline_cap = pipeline_capped_cores(total_blocks, split_grid.x * split_grid.y, min_blocks_per_core)
    if pipeline_cap is not None:
        max_cores = pipeline_cap if max_cores is None else min(max_cores, pipeline_cap)

    if max_cores is not None:
        # A capped core count: either the R3 pipeline-depth cap above (production)
        # or the height-only-split off-arm (which caps at nt_h). An even
        # base/remainder split over `num_cores` cores — the same distribution
        # ttnn.split_work_to_cores makes, just over a smaller core count than the
        # block count would give.
        num_cores = max(1, min(max_cores, total_blocks, split_grid.x * split_grid.y))
        base, rem = divmod(total_blocks, num_cores)
        per_core_blocks = [base + (1 if k < rem else 0) for k in range(num_cores)]
        return _materialize_cores(split_grid, num_cores, row_wise, spread_cores, per_core_blocks)

    (
        num_cores,
        all_cores,
        core_group_1,
        _core_group_2,
        blocks_per_core_g1,
        blocks_per_core_g2,
    ) = ttnn.split_work_to_cores(split_grid, total_blocks, row_wise)

    cores = ttnn.grid_to_cores(num_cores, split_grid.x, split_grid.y, row_wise)
    per_core_blocks = [blocks_per_core_g1 if core_group_1.contains(core) else blocks_per_core_g2 for core in cores]
    if not spread_cores or num_cores >= split_grid.x * split_grid.y:
        # The Phase-0 path, byte-identical: the split's own core list and its own
        # CoreRangeSet (contiguous ranges, cheaper to dispatch than singletons).
        return cores, all_cores, per_core_blocks
    # The block COUNTS the split produced are kept and re-attached positionally to
    # the spread core list — same counts, same order, different cores.
    return _materialize_cores(split_grid, num_cores, row_wise, spread_cores, per_core_blocks)


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    use_multicore: bool = True,
    use_double_buffer: bool = True,
    tile_height: int = DEFAULT_TILE_HEIGHT,
    pad=None,
    levers=None,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    lv = resolve_levers(levers)

    # ---------- 1. geometry + the block knobs ----------
    elem_size = input_tensor.element_size()
    # R5: the op is blocked over the OUTPUT's tile grid, which on a padded call
    # is the pad target rather than the input's own shape. `in_shape` stays the
    # input's, because the READ side addresses real rows of real width. Off the
    # pad path the two are the same list and every expression below is
    # byte-identical to Phase 0's.
    in_shape = list(input_tensor.shape)
    pad_enabled = bool(pad is not None and pad["enabled"])
    shape = list(pad["padded_shape"]) if pad_enabled else in_shape

    # One tile-column of one stick. The reader derives its per-block transfer
    # size as `w * tile_row_bytes`, so the block width is never restated.
    tile_row_bytes = TILE_WIDTH * elem_size

    in_tile_bytes = tile_height * TILE_WIDTH * elem_size  # RM input is never block-float
    out_tile_bytes = output_tensor.buffer_page_size()

    # R7 (A4/A5b): the whole dtype surface in one call — the cast flag, the fp32
    # bit-exact triple and the 8-bit DEST width. Host config and kernel template
    # argument come from the SAME dict, so they cannot drift apart.
    numeric = numeric_policy(input_tensor.dtype, output_tensor.dtype, elem_size, lv)
    needs_cast = numeric["needs_cast"]
    out_elem_size = element_size_or_zero(output_tensor)
    # R7: only a widening cast whose fill the OUTPUT format holds strictly
    # better arms the writer's pad rewrite (see `pad_cast_fix`).
    out_pad_fix, out_pad_word = pad_cast_fix(pad, input_tensor.dtype, elem_size, output_tensor.dtype, out_elem_size)

    grid = device.compute_with_storage_grid_size()
    num_l1_banks = grid.x * grid.y
    # A1/A5 note: on the `*_to_l1` / `l1_to_*` directions the operands live in
    # the SAME per-core L1 the CBs spend, so the budget subtracts them (Phase 0
    # could assume DRAM-only operands and did not). A *sharded* operand is
    # counted below instead, as the alias bytes of the CB it backs.
    interleaved_l1_bytes = l1_bytes_per_core(input_tensor, num_l1_banks) + l1_bytes_per_core(
        output_tensor, num_l1_banks
    )
    unreserved_bytes = _unreserved_l1_bytes()

    # ---------- 2. placement plan (which side is resident, which streams) ----
    # R3/R4: the placement knobs default to their REGIME value (measured, and of
    # opposite sign on different data paths); an explicit lever value forces the
    # knob on any shape, so both arms stay measurable everywhere. Resolved ONCE,
    # before the plan, because `reshard_pull` decides which side the plan makes
    # resident.
    placement = placement_defaults(input_tensor.memory_config(), output_tensor.memory_config())

    def _knob(name):
        return placement[name] if lv[name] is None else lv[name]

    nt_h_geo, Wt_geo, _, _ = tile_geometry(shape, tile_height)
    plan = plan_placement(
        shape=shape,
        tile_height=tile_height,
        in_memory_config=input_tensor.memory_config(),
        out_memory_config=output_tensor.memory_config(),
        Wt=Wt_geo,
        nt_h=nt_h_geo,
        in_tile_bytes=in_tile_bytes,
        out_tile_bytes=out_tile_bytes,
        # A3d clamp budget: measured before the alias bytes are known (the alias
        # IS the shard, already allocated), which is the conservative direction —
        # it only ever keeps a big crossover CB out, never lets one in.
        cb_budget_bytes=cb_budget_bytes(unreserved_bytes, interleaved_l1_bytes),
        force_streamed=bool(lv["force_streamed"]),
        reshard_pull=_knob("reshard_pull"),
        in_shape=in_shape,
        pad_enabled=pad_enabled,
    )
    if plan["error"] is not None:
        # A support gap, not a contract violation: validate() raises the same
        # typed refusal ahead of dispatch for every geometry it can see without
        # the allocated buffers (it cannot see a cliff shard's per-bank size).
        raise UnsupportedAxisValue(plan["error"])

    tile_descriptor = ttnn.TileDescriptor(tile_height, TILE_WIDTH)
    resident_in = plan["mode"] in RESIDENT_IN_MODES
    resident_out = plan["mode"] in RESIDENT_OUT_MODES

    # R4: the source page grid the reader indexes. `(1, row_bytes)` — the whole
    # row is one page — is the interleaved / HEIGHT-shard case and emits exactly
    # the Phase-0 reader; anything narrower makes the reader gather band by band.
    # Never 0: the gather arm's discarded branch still has to compile.
    n_bands, band_bytes = plan["bands"]
    if resident_in:  # the input is never read at all — keep the trivial value
        n_bands, band_bytes = 1, band_bytes

    # R6 (B13/D21): the stateful-read arm. It is the SAME transfers in the same
    # block, issued with the NoC command buffer configured once per source bank
    # instead of once per stick. Armed only where the plain per-stick reader is
    # actually the emitted code (no pad body, no band gather, not resident) and
    # where the R3 stagger — which owns the issue ORDER — is off.
    in_mem = input_tensor.memory_config()

    # R7 (A5b): the NARROW-STICK ALIGNMENT GAP, and whether this call falls in it.
    #
    # The reader's per-stick DRAM->L1 destination is `l1_base + s * w *
    # tile_row_bytes`, and a DRAM read requires a DRAM-ALIGNED L1 destination
    # (64 B here). `w * tile_row_bytes` is fixed by the tile layout — the tilize
    # LLK derives its own row stride from the block width, so padding the L1
    # stride is not available. At >= 2 bytes per datum every legal `w` clears 64
    # B; at ONE byte an odd `w` gives 32 / 96 / 160 ... and the odd sticks land
    # at phase 32. `wt_block_max`'s `max(2, ...)` floor does NOT save it, because
    # `WT_BLOCK = min(Wt, WT_BLOCK_MAX)` clamps to Wt (Wt == 1 is the extreme).
    # Measured before the fix: watcher "invalid address alignment in NOC
    # transaction", uint8 `[1,1,32,32]` / `[1,1,32,96]` wrong, `[1,1,32,64]` /
    # `[1,1,32,128]` clean. The reader stages such a block through a scratch CB.
    #
    # Only the DRAM source is affected: an L1 source needs L1 alignment (16 B),
    # which every whole tile-column clears. A resident side reads nothing.
    dram_align = int(ttnn.get_dram_alignment())
    src_is_dram = (not resident_in) and in_mem.buffer_type == ttnn.BufferType.DRAM
    stateful_knob = STATEFUL_READS if lv["stateful_reads"] is None else lv["stateful_reads"]
    addrgen_knob = FAST_ADDRGEN if lv["fast_addrgen"] is None else lv["fast_addrgen"]
    bank_period = 0
    if (
        (stateful_knob or addrgen_knob)
        and not resident_in
        and n_bands == 1
        and not pad_enabled
        and not int(_knob("stagger_reads"))
    ):
        dram_grid = device.dram_grid_size()
        bank_period = read_bank_period(
            dram_grid.x * dram_grid.y,
            tile_height,
            is_dram_interleaved=(not in_mem.is_sharded() and in_mem.buffer_type == ttnn.BufferType.DRAM),
        )
    stateful_reads = 1 if (bank_period and stateful_knob) else 0
    fast_addrgen = 1 if (bank_period and addrgen_knob) else 0
    if not (stateful_reads or fast_addrgen):
        bank_period = 0  # the arm is unreachable; keep the CT vector honest

    # The shard hands you the block width on any resident side (op_design §6.2).
    blk = blocking(shape, tile_height, elem_size, lv["target_read_bytes"], wt_block_override=plan["wt_block"])
    nt_h = blk["nt_h"]
    Wt = blk["Wt"]
    wt_block = blk["wt_block"]
    n_wchunks = blk["n_wchunks"]
    total_blocks = blk["total_blocks"]
    tail_block_start = blk["tail_block_start"]
    # Off the fully-streamed path every block is the shard's own width, so the
    # tail column-block is not a distinct width and `n_tail` is always 0.
    wt_tail = blk["wt_tail"] if plan["mode"] == MODE_STREAMED else wt_block

    # R7 (A5b), continued: both block widths are read with the same stride rule,
    # so either one being off the alignment grid arms the staged reader for the
    # whole kernel. `stage_stride_max` is what the scratch has to hold per stick
    # (+ one unit of slack, because a CB base is only L1-aligned).
    stage_unaligned = int(src_is_dram and any((w * tile_row_bytes) % dram_align for w in (wt_block, wt_tail)))
    if lv["stage_reads"] is not None:
        stage_unaligned = int(bool(lv["stage_reads"]) and stage_unaligned)
    stage_bytes = tile_height * _round_up(max(wt_block, wt_tail) * tile_row_bytes, dram_align) + dram_align
    if stage_unaligned:
        # The B13/D21 issue-cost levers write straight into the tile-layout
        # destination, which is exactly what is illegal here. They are a
        # low-work-regime optimization, so the correct read wins.
        stateful_reads = fast_addrgen = bank_period = 0

    # ---------- 3. circular buffers ----------
    # A resident CB is ALIASED onto the shard buffer: the shard IS the CB, so
    # the reader/writer move zero bytes on that side. `total_size=0` asks for the
    # tensor's own per-bank size, which must be a whole number of dense shard
    # blocks (a core can hold SEVERAL boxes when there are more shards than
    # cores). Anything else — a padded or cliff shard — is not densely
    # addressable and falls back to streaming.
    def _aliased_cb(index, tensor, page_size, core_ranges, block_bytes):
        try:
            cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor, 0, 0, core_ranges)
        except Exception:  # some nd geometries cannot express a CB at all
            return None, 0
        if block_bytes == 0 or cb.total_size % block_bytes:
            return None, 0
        cb.format_descriptors = [
            ttnn.CBFormatDescriptor(
                buffer_index=index,
                data_format=tensor.dtype,
                page_size=page_size,
                tile=tile_descriptor,
            )
        ]
        return cb, cb.total_size // block_bytes

    def _streamed_cb(index, dtype, page_size, core_ranges, num_pages):
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=core_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index,
                    data_format=dtype,
                    page_size=page_size,
                    tile=tile_descriptor,
                )
            ],
        )

    # ---------- 4. core assignment ----------
    # Sharded: the cores are FIXED by the shard spec (master.md A2) — core k
    # holds shard k, and its blocks are that shard's own contiguous range of the
    # `b = wchunk*nt_h + r` linearization. Interleaved: the Phase-0 split.
    # width_split == 0 is the height-only-split off-arm: cap the core count at
    # nt_h, which is byte-identical to the default when n_wchunks == 1 and
    # collapses a wide-short tensor onto min(nt_h, grid) cores — the exact
    # regression the 2-D linearization exists to prevent.
    if plan["mode"] == MODE_STREAMED:
        cores, all_cores, per_core_blocks = plan_cores(
            device,
            total_blocks,
            use_multicore=use_multicore and bool(lv["multicore"]),
            row_wise=bool(lv["row_wise"]),
            max_cores=None if lv["width_split"] else nt_h,
            min_blocks_per_core=int(_knob("min_blocks_per_core")),
            spread_cores=bool(_knob("spread_cores")),
        )
        assignments = []
        block_start = 0
        for core, num_blocks in zip(cores, per_core_blocks):
            assignments.append((core, block_start, num_blocks))
            block_start += num_blocks
    else:
        sharded_tensor = input_tensor if plan["sharded_side"] == "in" else output_tensor
        cores, all_cores = shard_grid_cores(device, sharded_tensor)

    # ---------- 5. commit the plan against the real buffers ----------
    # The analytic plan says the shard COULD be the block; the buffers decide
    # whether it IS one. A core can hold several shard boxes (more shards than
    # cores), which is still a dense run of blocks; a cliff/padded shard is not,
    # and downgrades to streaming.
    alias_bytes = 0
    boxes_per_core = 1
    cb_input_sticks = cb_output_tiles = None
    if resident_in or resident_out:
        nt_h_shard = plan["shard"]["nt_h_shard"]
        boxes_in = boxes_out = None
        if resident_in:
            cb_input_sticks, boxes_in = _aliased_cb(
                CB_INPUT_STICKS, input_tensor, in_tile_bytes, all_cores, nt_h_shard * wt_block * in_tile_bytes
            )
        if resident_out:
            cb_output_tiles, boxes_out = _aliased_cb(
                CB_OUTPUT_TILES, output_tensor, out_tile_bytes, all_cores, nt_h_shard * wt_block * out_tile_bytes
            )
        boxes = boxes_in if resident_in else boxes_out
        aliased_ok = (not resident_in or cb_input_sticks is not None) and (
            not resident_out or cb_output_tiles is not None
        )
        if resident_in and resident_out and boxes_in != boxes_out:
            aliased_ok = False  # the two sides disagree about the block run
        if plan["mode"] != MODE_RESIDENT and boxes != 1:
            # A crossover core's blocks must be ONE contiguous range of the
            # linearization; several boxes per core is Refinement 4's territory.
            aliased_ok = False
        if not aliased_ok:
            return create_program_descriptor(
                input_tensor,
                output_tensor,
                use_multicore=use_multicore,
                use_double_buffer=use_double_buffer,
                tile_height=tile_height,
                pad=pad,
                levers=dict(lv, force_streamed=1),
            )
        boxes_per_core = boxes
        alias_bytes = (cb_input_sticks.total_size if resident_in else 0) + (
            cb_output_tiles.total_size if resident_out else 0
        )

    if plan["mode"] != MODE_STREAMED:
        nt_h_shard = plan["shard"]["nt_h_shard"]
        if plan["mode"] == MODE_RESIDENT:
            # Every core tilizes the blocks sitting in its own L1 — it never needs
            # to know WHICH shards those are, which is why orientation is a
            # non-issue on this path.
            assignments = [(core, 0, boxes_per_core * nt_h_shard) for core in cores]
        else:
            n_sh_cols = plan["shard"]["n_sh_cols"]
            assignments = [
                (core, (k % n_sh_cols) * nt_h + (k // n_sh_cols) * nt_h_shard, nt_h_shard)
                for k, core in enumerate(cores)
            ]

    streamed_page_bytes = (0 if resident_in else in_tile_bytes) + (0 if resident_out else out_tile_bytes)
    cb_depth = cb_depth_for(
        want_depth2=use_double_buffer and bool(lv["double_buffer"]),
        depth2_bytes=2 * wt_block * streamed_page_bytes,
        budget_bytes=cb_budget_bytes(unreserved_bytes, interleaved_l1_bytes + alias_bytes),
    )
    cb_pages = cb_depth * wt_block  # >= wt_block >= wt_tail: no reader deadlock
    if cb_input_sticks is None:
        cb_input_sticks = _streamed_cb(CB_INPUT_STICKS, input_tensor.dtype, in_tile_bytes, all_cores, cb_pages)
    if cb_output_tiles is None:
        cb_output_tiles = _streamed_cb(CB_OUTPUT_TILES, output_tensor.dtype, out_tile_bytes, all_cores, cb_pages)

    # R7 (A5b): the narrow-stick scratch. READER-PRIVATE — it is never pushed or
    # popped, so it has no producer/consumer pair to synchronize; it is a named
    # L1 region the reader stages one block into. Allocated ONLY on the path that
    # uses it, so no other dtype pays L1 for it. Bounded by a constant, like
    # every other CB here: `wt_block <= WT_BLOCK_MAX` is a byte target, so
    # `stage_bytes <= tile_h * (TARGET_READ_BYTES + dram_align) + dram_align`
    # regardless of W.
    # No tile descriptor: this CB holds raw staged sticks, not tiles, so its one
    # page is the whole staging area.
    cb_read_stage = (
        ttnn.CBDescriptor(
            total_size=stage_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_READ_STAGE,
                    data_format=input_tensor.dtype,
                    page_size=stage_bytes,
                )
            ],
        )
        if stage_unaligned
        else None
    )

    # ---------- 6. kernels ----------
    # CT args: scalar args first, TensorAccessorArgs appended LAST (master.md
    # D18). RT args carry only buffer addresses + the per-core block range
    # (D19), so a second call with the same spec hits the program cache.
    reader_ct_args = [
        CB_INPUT_STICKS,
        nt_h,
        n_wchunks,
        tile_height,
        tile_row_bytes,
        wt_block,
        wt_tail,
        lv["barrier_per_block"],
        lv["stub_read"],
        int(resident_in),
        int(_knob("stagger_reads")),
        n_bands,  # R4: source pages per row (1 = whole-row pages, the Phase-0 path)
        # (R6's two CT args are appended after the R5 block below, so the pad
        # geometry terms keep the indices Refinement 5 pinned.)
        band_bytes,  # R4: bytes per source page
        # R5 (P1/P2/P4/P5): the pad body's CT selector and its four geometry
        # terms. `pad_enabled == 0` erases the whole body, so Track A is
        # unchanged structurally rather than by convention.
        int(pad_enabled),
        pad["pad_word"] if pad_enabled else 0,
        pad["hp"] if pad_enabled else tile_height,  # padded rows per image
        pad["h_real"] if pad_enabled else tile_height,  # real rows per image
        pad["nimg_real"] if pad_enabled else 1,  # real images
        pad["real_row_bytes"] if pad_enabled else tile_row_bytes,  # real bytes per stick
        # R6 (B13/D21): the stateful-read arm and the bank period it groups by.
        int(stateful_reads),
        bank_period,
        int(fast_addrgen),
        # R7 (A5b): the narrow-stick staging path and its scratch.
        stage_unaligned,
        CB_READ_STAGE,
        dram_align,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct_args = [
        CB_OUTPUT_TILES,
        nt_h,
        n_wchunks,
        Wt,
        wt_block,
        wt_tail,
        out_tile_bytes,
        lv["coalesce_writes"],
        lv["stub_write"],
        int(resident_out),
        # R7: the output-format pad rewrite. `pad` is None off the padded path,
        # so the geometry terms fall back to values that make the (erased) body
        # trivially interior.
        out_pad_fix,
        out_pad_word,
        pad["hp"] if pad_enabled else tile_height,
        pad["h_real"] if pad_enabled else tile_height,
        pad["nimg_real"] if pad_enabled else 1,
        (pad["real_row_bytes"] // elem_size) if pad_enabled else TILE_WIDTH,
        tile_height,
        out_elem_size,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # R6 (C14 second degree): the fold is only meaningful where BOTH dataflow
    # kernels are pure CB handshakes — i.e. the same-spec resident path, where
    # neither side moves a byte. A crossover or a reshard still has one real
    # dataflow kernel, so folding it would serialize genuine NoC work onto the
    # compute thread rather than remove an empty kernel.
    fold_knob = FOLD_RESIDENT if lv["fold_resident"] is None else lv["fold_resident"]
    fold_resident = int(bool(fold_knob) and plan["mode"] == MODE_RESIDENT)

    # R6: the upfront input wait is legal exactly where the input CB is the shard
    # and the reader pushes all of it at once — i.e. on a resident INPUT. The
    # knob can only turn it OFF; residency is the gate, never the caller.
    wait_upfront_knob = WAIT_UPFRONT_RESIDENT if lv["wait_upfront"] is None else lv["wait_upfront"]
    wait_upfront = int(bool(wait_upfront_knob) and resident_in)

    compute_ct_args = [
        CB_INPUT_STICKS,
        CB_OUTPUT_TILES,
        wt_block,
        wt_tail,
        needs_cast,
        lv["stub_compute"],
        fold_resident,
        int(TILIZE_UNINIT if lv["tilize_uninit"] is None else lv["tilize_uninit"]),
        wait_upfront,
        numeric["fp32_lossless"],  # R7 (A4): Fp32Mode::Lossless on the fp32 identity
    ]

    # R7: the op's ONLY ComputeConfigDescriptor. Every field is derived from the
    # dtype pair (`numeric_policy`); at the pre-R7 dtypes all three are their
    # defaults, so the bf16 program is byte-identical to Refinement 6's.
    # `unpack_to_dest_mode` is a per-CB vector — sized to the arch's CB count and
    # indexed by CB slot, never by allocation order.
    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = numeric["fp32_dest_acc_en"]
    compute_config.bfp8_pack_precise = numeric["bfp8_pack_precise"]
    if numeric["unpack_to_dest_fp32"]:
        unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
        unpack_modes[CB_INPUT_STICKS] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = unpack_modes

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()

    for core, block_start, num_blocks in assignments:
        # A core's contiguous [b0, b0+nb) crosses the full/tail column-block
        # boundary at most once, because the tail column-block occupies the
        # contiguous suffix [tail_block_start, total_blocks) of the linear space.
        # Off the streamed path every block is the shard's width, so there is no
        # tail width and n_tail is 0.
        if plan["mode"] == MODE_STREAMED:
            n_full = min(max(tail_block_start - block_start, 0), num_blocks)
        else:
            n_full = num_blocks
        n_tail = num_blocks - n_full

        reader_rt[core.x][core.y] = [in_addr, block_start, num_blocks]
        writer_rt[core.x][core.y] = [out_addr, block_start, num_blocks]
        # `num_blocks * wt_block` is the page count BOTH dataflow kernels use on
        # the resident path — stated once here so the fold cannot disagree with
        # the kernels it replaces.
        compute_rt[core.x][core.y] = [n_full, n_tail, num_blocks * wt_block]

    # B9 off-arm: swap the two configs so the read stream lands on the writer's
    # RISC/NoC and vice versa.
    reader_config = ttnn.ReaderConfigDescriptor() if lv["noc_split"] else ttnn.WriterConfigDescriptor()
    writer_config = ttnn.WriterConfigDescriptor() if lv["noc_split"] else ttnn.ReaderConfigDescriptor()

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=reader_config,  # NCRISC / NoC0 by default (master.md B9)
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=writer_config,  # BRISC / NoC1 by default (master.md B9)
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=compute_config,
    )

    # R6 / C14 second degree: on the folded resident path the two dataflow
    # kernels are not merely idle, they are ABSENT — the compute kernel owns both
    # ends of the handshake, so the program launches one kernel per core.
    kernels = [compute_kernel] if fold_resident else [reader_kernel, writer_kernel, compute_kernel]

    return ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=[],
        cbs=[cb_input_sticks, cb_output_tiles] + ([cb_read_stage] if cb_read_stage is not None else []),
    )
