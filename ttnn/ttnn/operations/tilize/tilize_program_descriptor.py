# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor (CBs, kernels, work split).

Implements the ``grid2d_full_width`` / ``grid2d_width_chunked`` regimes of
``op_design.md``: the output tile grid ``R x C`` is blocked on BOTH axes and the
linearized block id ``row_group * num_w_chunks + w_chunk`` is what
``split_work_to_cores`` hands to the cores. ``num_w_chunks == 1`` is the
full-width parameterization; nothing else distinguishes the two regimes.

Every block knob below is a named host constant or a derived quantity with a
single source. CB page counts, loop trip counts and grid sizing are computed
FROM those knobs — never from a whole-op dimension and never as a literal.

``grid2d_sharded`` (Refinement 1) is the SAME schedule with the block grid read
off a shard spec instead of solved: ``num_row_groups`` / ``num_w_chunks`` /
``block_width_tiles`` come from the shard's own geometry, the shard's linear index
IS the block id the three kernels already derive ``(row_group, w_chunk)`` from, and
each core is handed exactly the block(s) whose bytes are resident in its own L1
(``start_block_id`` = the core's index in the shard's core order, ``block_stride`` =
the number of cores — which is round-robin for an ND spec and the identity for a
legacy 2-D one). The sharded side's CB is then **zero-copy over the shard buffer**
(``ttnn.cb_descriptor_from_sharded_tensor``) — a core's own shard is never re-read
through a ``TensorAccessor``. The accessor keeps the interleaved leg and the
genuinely non-local cross-spec leg.

RAGGED COLUMN TAIL (Refinement 6). ``block_width_tiles`` is
``ceil(C / num_w_chunks_target)`` — the design's own extent — and the leftover
``C % block_width_tiles`` columns become ONE tail chunk of their own width.

The mechanism cap that used to force a divisor is real and unchanged: neither CB
endpoint may wrap mid-transfer (``llk_push_tiles``'s
``LLK_ASSERT(remaining >= num_words)`` in ``llk_io_pack.h``, and
``cb_pop_front``'s ``ASSERT(fifo_rd_ptr <= fifo_limit)`` at
``dataflow_api.h:269``, commented "consumer always reads from contiguous memory,
it cannot wrap"), so a CB's page count must be an exact multiple of every
push/pop quantum used ON THAT CORE. The escape is that the mix was only ever a
problem *within* one core: the plan carries TWO DISJOINT CORE RANGES — the
full-width cores and the tail cores — each with its own ``block_width_tiles``
compile-time arg, its own ``col_tile_offset`` and its own CB sizing. Every core
still sees exactly one quantum, so the invariant holds by construction while the
column extent goes back to the design's value.

Smooth ``C`` is byte-identical to the divisor rule (``C % bw == 0`` -> no tail
group, one core range, the same numbers: ``[1,1,32,16384]`` -> 8,
``[1,1,1024,1024]`` -> 16, ``[1,1,2048,2048]`` -> 64). Rough ``C`` is where it
pays: ``[1,1,1,50304]`` (``C = 1572 = 2^2*3*131``) went from 131 chunks of 12
(768 B reads, and 3 blocks on the busiest core against a 2.05 average) to 62
chunks of 25 plus one of 22 (1600 B reads, one block per core).

The one place the divisor rule survives is a SUB-ROW-PAGED source
(``input_pages_per_row > 1``): there the block's row segment must additionally
sit inside a single source page, which is a constraint on the width itself and
not on the per-core mix.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- CB slots (semantic names; the numeric slot is just the buffer index) ---
CB_INPUT_ROWS = 0  # reader -> compute: row-major sub-block, one page per tile
CB_OUTPUT_TILES = 1  # compute -> writer: whole output tile pages
# reader-local scratch: ONE block row pre-filled with the pad value, so a whole
# pad row costs one DM-engine transfer instead of a RISC store loop. Allocated
# only on the padded path (`plan.pad_active`).
CB_PAD_ROW = 2
# writer -> compute: the TRAILING half of each block's row-major sub-block, on
# the split-reader path (Refinement 6). A CB of its own rather than a second
# producer on CB_INPUT_ROWS — one producer and one consumer per CB is a hard CB
# invariant, and two producers on one CB is silent UB.
CB_INPUT_ROWS_SPLIT = 3
# Length of the per-CB `unpack_to_dest_mode` vector. NOT the number of slots this
# op allocates: `get_unpack_dst_formats` (jit_build/data_format.cpp) indexes it
# against the FULL per-core CB table and TT_FATALs on anything shorter
# ("unpack_to_dest_mode vector must have 32 elements"), so it is the hardware's
# CB-slot count and every entry this op does not own stays `Default`.
NUM_CB_SLOTS = 32
assert CB_PAD_ROW < NUM_CB_SLOTS, "a CB slot must be addressable in the unpack_to_dest_mode vector"

# ---------------------------------------------------------------------------
# Named block knobs — the single source of each. None is a tensor dimension.
# ---------------------------------------------------------------------------
TILE_WIDTH = 32
# Output tile-page writes to keep in flight behind one write barrier. Sets
# `write_rows_per_barrier`, which is inert (== 1) once the block is >= this wide.
# MEASURED, not assumed: op_design.md's "Write batch depth" lamp asked whether 8
# sits past the knee, and on `[1,1,16384,32]` (C == 1, so this constant IS the
# whole knob) it does. Median device kernel ns over 3 fresh-cache runs at 64/64
# cores: wb=1 24430 (the one-write-per-barrier trap), wb=2 22519, wb=4 20723,
# wb=8 22568, wb=16 22403. 4 is the plateau — 1.18x over the trap and 1.09x over
# 8 — which is where the `double_buffer` catalog entry put it. Inert on every
# block already >= 4 wide (the perf-focus shape's bw=8 gives wrpb=1 either way),
# and it costs LESS L1 than 8. See
# tests/.../tilize/test_tilize_lever_write_batch.py for the harness.
WRITE_BATCH_MIN_TILES = 4
# can_use_fast_tilize requires block_width_tiles < 256 (tilize_helpers.inl:77).
FAST_TILIZE_WIDTH_CAP = 255
# low_l1=True cap on the column extent. A host constant, independent of EVERY
# tensor dimension — that independence is the low_l1 contract, structurally.
LOW_L1_WIDTH_CAP = 4
# cb_input_rows depth, in tile-rows of the block (reader/compute overlap).
# MEASURED and KEPT AT ITS DEFAULT, with the flat result recorded rather than
# hidden: op_design.md's "Overlap (input depth)" lamp asked for {2,3,4} on the
# two wide-block shapes, and the answer is that the knob does not move either of
# them. Device kernel ns, 2 fresh-cache runs each, all at 64/64 cores:
#   [1,1,32,32768]  (bw=16, 1 KiB reads)  d1 25358/25582  d2 25863/24653
#                                         d3 27056/25478  d4 26285/25509
#   [1,1,2048,2048] (bw=64, 4 KiB reads)  d1 91377/95444  d2 92394/91515
#                                         d3 92520/93470  d4 93191/93780
# Depth 1 being flat too is the informative part: it says the READS are the wall
# on these shapes, so the overlap slot is not what is missing. `[1,1,2048,2048]`
# moves 16 MiB in ~91.5 us = ~183 GB/s, ~64% of this part's peak on a
# simultaneous read+write stream — near the practical DRAM roofline, which no CB
# depth can move. So 2 is kept: it is the smallest value that overlaps at all,
# it is byte-identical in output, and it costs the least L1 of the values that do
# (d3 would cost 640 KB on square_large vs 512 KB). It stays a LIVE knob, not an
# inlined constant, because what would have to change for depth > 2 to pay is a
# shape where reads are not the wall — the fp32 refinement (which drops off the
# fast-tilize path, making compute the expensive stage) is the obvious candidate.
# Harness: tests/.../tilize/test_tilize_lever_input_depth.py.
INPUT_DEPTH_ROWS = 2
# cb_output_tiles depth, in WRITE BATCHES (one batch = write_rows_per_barrier
# tile-rows). Depth 2 buys the writer a full batch in flight while compute
# fills the next one, and keeps the capacity an exact multiple of the batch so
# a full batch never straddles the FIFO wrap.
OUTPUT_DEPTH_BATCHES = 2
# Pipeline waves a core should own, where ONE WAVE is one tile-row of the block —
# the reader's push quantum, the compute helper's per-call unit and the writer's
# minimum wait. A core overlaps its DRAM reads against its DRAM writes only if it
# owns MORE THAN ONE wave; with exactly one it reads, then computes, then writes,
# strictly in series, and with every core in lockstep the device alternates a
# read-only phase with a write-only phase instead of sustaining both.
#
# The wave count is fully determined by the column cut, because the tensor holds
# `R * num_w_chunks` tile-rows in total:
#     waves_per_core = R * num_w_chunks / num_cores
# so `w_chunks_for_occupancy` (fill the grid) and `w_chunks_for_waves` (fill the
# pipe) are the SAME expression, this constant being the factor between them —
# which is why there is one knob here and not two. At 1 it is byte-identical to
# the occupancy-only cut.
#
# This constant is the CAP on how deep the pipe is allowed to get; the value
# actually taken is the deepest one whose read transaction still clears
# `MIN_BLOCK_ROW_BYTES` below, because a wave is bought by HALVING the block
# width. Past 4 the pipe is full and the extra waves only shrink the read.
# MEASURED on the `attention:` LOOSE_CASE `[1,1,32,16384]` (R = 1, so a core
# owns exactly one wave at 1) and on `[1,1,2048,2048]` — see
# tests/.../tilize/test_tilize_lever_pipeline_waves.py and the ablation in
# changelog.md Refinement 3.
PIPELINE_WAVES_PER_CORE = 4
# The read-transaction floor that stops the trade above. A wave is bought by
# HALVING the block width, so it is only worth buying while the resulting read
# stays large enough that the NoC/DRAM per-transaction cost is still amortized.
# MEASURED across five geometries (device kernel ns, 64/64 cores throughout,
# medians where the numbers were close; harness
# tests/.../tilize/test_tilize_lever_pipeline_waves.py). Read size per wave
# setting, then the wall:
#   [1,1,32,16384]  512/256/128/64 B  -> 13759 / 13620 / 15204 / 26947
#   [1,1,32,32768] 1024/512/256/128 B -> 25267 / 24077 / 23757 / 28964
#   [1,1,1024,1024]1024/512/256/128 B -> 23322 / 23555 / 24436 / 23895
#   [1,1,2048,2048]4096/2048/1024/512 -> 92722 / 89365 / 86085 / 85380
#   [1,1,2048,64]   128/64 B          ->  4797 /  6146
# 512 B is the largest value that is never the wrong call: every geometry whose
# read stays at or above it either improves (1.08x on `[1,1,2048,2048]`, 1.05x
# on `[1,1,32,32768]`) or lands inside the +-3% run-to-run band, and every
# geometry that would have to go BELOW it to buy a wave loses (128 B on
# `[1,1,32,16384]`, 64 B on `[1,1,2048,64]`, 256 B on `[1,1,1024,1024]`).
# In BYTES rather than tiles so it means the same thing at every element size.
MIN_BLOCK_ROW_BYTES = 512
# Drop the tilize helper's per-call unpack+pack data-format reconfig
# (`ReconfigureRegisterDatatypeMode::NoReconfigure`). Correct in THIS kernel
# because `compute_kernel_hw_startup(cb_in, cb_out)` programs srcA/srcB and the
# pack format once and nothing else runs on the TRISCs, so every per-call
# reconfig rewrites the value already in the register — including on the casting
# diagonal, where the cast is carried by the two CBs' formats and not by the
# reconfig. Exposed as a knob rather than hardcoded because a refinement that
# adds a SECOND compute phase to this kernel would have to turn it back on.
COMPUTE_SKIP_FORMAT_RECONFIG = True
# Pay the tilize LLK init + uninit once per core instead of once per block
# (`InitUninitMode::InitOnly / Neither / UninitOnly`). Gated at emission time on
# a core actually owning more than one block — see the compute kernel's
# `amortize_init` comment for why the dead instantiations are not free.
COMPUTE_AMORTIZE_INIT = True

# --- split reader (Refinement 6) -------------------------------------------
# Read transaction size (BYTES) at or below which the block's stick reads are
# SPLIT across both data-movement RISC-Vs: the reader (NoC0) takes the leading
# tile-rows of every block, the writer (NoC1) takes the trailing ones into its
# own input CB, and compute consumes the block as two back-to-back sub-blocks.
#
# MEASURED gate, not a guess. The `split_reader` catalog entry
# (ttnn/ttnn/operations/examples/master.md) is explicit that this does nothing
# unless a data-movement RISC-V is itself the bottleneck, so the whole-op
# ablation came first — `[1,1,16384,32]`, 64/64 cores, device kernel ns:
#     all payloads stubbed  6221   (NCRISC 5921, BRISC  598)
#     + read payload       14362   (NCRISC 14054, BRISC 630)
#     + write payload       9220   (NCRISC 5937, BRISC 8909)
#     + compute payload     6479
#     full op              20993   (NCRISC 17448, BRISC 20688)
# The reader RISC-V is on the critical path for the WHOLE kernel and 6221 ns of
# the 20993 is its per-stick loop with NO NoC payload at all (256 sticks/core at
# 64 B), while the writer's own payload is 2999 ns. That is the catalog's
# signature exactly. A 512 B read (the `attention` twin) is 8x fewer commands
# for the same bytes and is NOT issue-bound, which is why this is a byte
# threshold and not "always on": 256 B is one tile-row of 4 bf16 tiles, the
# widest block whose reads were measured issue-dominated.
SPLIT_READER_MAX_ROW_BYTES = 256
# Share of each block's tile-rows the WRITER reads, in percent. NOT 50: the two
# data-movement RISC-Vs are not interchangeable — the writer also carries the
# block's stores, and its reads share NoC1 with them — so the balance point is a
# measurement. On `[1,1,16384,32]` (8 tile-rows per block, so the reachable
# splits are 0..6 writer rows), device kernel ns at 64/64 cores:
#     writer rows  0      1      2      3      4      6
#     device ns  20770  19923  18864  17945  20797  24716
#     NCRISC     17278  15751  14721  10402   8417   4262
# 3 of 8 (38%) is the floor of the curve and 1.16x over the unsplit baseline.
# Past it the writer becomes the wall faster than the reader is relieved: at an
# even split NCRISC is down to 8417 but BRISC is still ~20500, i.e. BRISC costs
# roughly 2.3x per stick what NCRISC does once its stores are counted.
SPLIT_READER_WRITER_SHARE_PCT = 38

# Let the column cut leave a RAGGED TAIL chunk (`C % block_width_tiles` columns)
# carried by its own core range, instead of forcing `block_width_tiles` to be a
# divisor of `C`. See the module docstring for the CB-wrap mechanism this
# respects and how two core ranges respect it. A live knob: at False the plan
# falls back to the divisor rule and is byte-identical to Refinement 5.
# Inert on smooth `C` either way (`C % bw == 0` emits no tail group).
RAGGED_COLUMN_TAIL = True

# Legal output tile heights (power-of-two fractions of 32).
LEGAL_TILE_HEIGHTS = (1, 2, 4, 8, 16, 32)
# A tile's FACE is 16 wide at every legal tile height, and `min(tile_h, 16)`
# tall (`tt_metal/impl/data_format/tile.cpp:TILE_FACE_HW_CHOICES`). A tile of
# height `h` is therefore `h / min(h,16)` face-ROWS of two faces each, laid out
# face-row-major, elements row-major inside a face. Everything the retile block
# operation does is derived from that one fact.
FACE_WIDTH = 16


def retile_copy_unit(in_tile_h: int, out_tile_h: int) -> tuple:
    """(rows, cols) of the largest run that is CONTIGUOUS IN BOTH tile layouts.

    This is the whole retile algorithm in one expression, and it is what makes
    the re-tile a pure NoC gather/scatter with no compute stage at all: the
    output tile's bytes are assembled directly out of the input tiles' faces.

    Let `a = min(h, 16)` be a layout's face height.

      * `a_in == a_out`: face `(fr, 0)` and face `(fr, 1)` are ADJACENT in both
        layouts, so a whole face-PAIR SLAB (`a` consecutive rows x all 32
        columns, stored left-face-then-right-face — NOT row-major) is one byte
        run with the same internal order on both sides. Consecutive slabs stay
        contiguous while they remain inside one input tile AND one output tile,
        so the run extends to `min(in_tile_h, out_tile_h)` rows.
      * `a_in != a_out`: the slabs interleave differently, and the largest
        common run is a single face FRAGMENT — `min(in_tile_h, out_tile_h)`
        rows x 16 columns. (When the face heights differ, that minimum is
        always < 16 and divides both face heights, both being powers of two.)

    Verified exhaustively against the layout formula for all 36 legal
    (in_tile_h, out_tile_h) pairs; see probes/probe_024.py.
    """
    a_in = min(int(in_tile_h), FACE_WIDTH)
    a_out = min(int(out_tile_h), FACE_WIDTH)
    rows = min(int(in_tile_h), int(out_tile_h))
    return rows, (TILE_WIDTH if a_in == a_out else FACE_WIDTH)


def _bfloat16_bits(value: float) -> int:
    """`value` as a bfloat16 bit pattern, rounded to nearest EVEN.

    bfloat16 is fp32's top 16 bits, so the encoding is a truncation plus the
    round the hardware and torch both apply — a plain truncation would put a
    different number in the pad region than the oracle's
    `torch.nn.functional.pad(x.bfloat16(), value=v)` for any fill that is not
    exactly representable. The `+1` carries into the exponent naturally, which
    is also what makes an overflowing fill saturate to inf rather than wrap.
    """
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    upper, lower = (bits >> 16) & 0xFFFF, bits & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and (upper & 1)):
        upper = (upper + 1) & 0xFFFF
    return upper


def _fp8_e4m3_bits(value: float) -> int:
    """`value` as an OCP fp8-e4m3 (1-4-3, bias 7, no inf) bit pattern.

    Written as a nearest-value search over the 256 encodings rather than as bit
    surgery: the fill is a host-side constant computed once per program build,
    so the exhaustive form costs nothing and cannot get the subnormal /
    saturation edges wrong. Ties resolve to the even mantissa, matching the
    round-to-nearest-even the packer applies to a real element.
    """
    value = float(value)
    if value != value:  # NaN
        return 0x7F
    best, best_err = 0, None
    for bits in range(256):
        exp, man = (bits >> 3) & 0xF, bits & 0x7
        if exp == 0xF and man == 0x7:
            continue  # the NaN encoding is not a value
        magnitude = (man / 8.0) * 2.0**-6 if exp == 0 else (1.0 + man / 8.0) * 2.0 ** (exp - 7)
        candidate = -magnitude if bits & 0x80 else magnitude
        err = abs(candidate - value)
        if best_err is None or err < best_err or (err == best_err and (man & 1) == 0):
            best, best_err = bits, err
    return best


def pad_fill_word(dtype, value, elem_size: int) -> int:
    """The fill as an `elem_size`-wide bit pattern in the INPUT's format.

    The fill is written into `cb_input_rows`, whose data_format is the INPUT
    dtype, so the encoding is the input's — the output cast happens later, at
    pack time, exactly as it does for a real element.

    A NEGATIVE fill on an integer dtype is a two's-complement bit_cast at the
    element width and must not truncate: masking to `elem_size * 8` bits is the
    whole rule, and it is written width-generically here so the integer dtypes
    Refinement 5 adds need no second implementation.
    """
    if value is None:
        value = 0
    if dtype == ttnn.bfloat16:
        return _bfloat16_bits(value)
    if dtype == ttnn.float32:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if dtype == _FP8_E4M3:
        return _fp8_e4m3_bits(value)
    if dtype in (ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8):
        return int(value) & ((1 << (elem_size * 8)) - 1)
    raise RuntimeError(f"tilize: no pad-fill encoding for dtype {dtype}")


# ---------------------------------------------------------------------------
# Numerical policy — the three dtype-driven compute-config decisions
# ---------------------------------------------------------------------------
#
# All three are DERIVED from (input dtype, output dtype) and none of them adds a
# code path: the cast itself is carried by the two CBs' `data_format`s and
# happens at pack time, exactly as it did when both sides were bfloat16. What
# these select is the compute config the datapath needs in order to be
# VALUE-PRESERVING at that pair.

# fp8_e4m3 exists only on architectures that have it; `getattr` keeps this
# module importable where the enum is absent rather than making fp8 a build-time
# dependency of the whole op.
_FP8_E4M3 = getattr(ttnn, "fp8_e4m3", None)


def is_lossless_fp32_relay(in_dtype, out_dtype) -> bool:
    """True when the tilize must take the SLOW (bit-exact) fp32 path.

    `can_use_fast_tilize` already refuses a Float32 OUTPUT (its pack stage steps
    DEST at bf16 stride — `tilize_helpers.inl`), so an fp32 -> fp32 relay lands
    on the regular `tilize_init`/`tilize_block` path either way. What is NOT
    automatic is that that path stays bit-exact: without
    `UnpackToDestMode::UnpackToDestFp32` on the input CB the unpacker still
    routes the datum through SrcA, which is tf32 (10 mantissa bits), and the
    output comes back off by ~1e-3 relative. So all three of `Fp32Mode::Lossless`,
    `fp32_dest_acc_en=true` and the `UnpackToDestFp32` tag are required together,
    and this predicate is their single source.

    This is the one case where the helper's own "prefer Fast, the downstream FPU
    truncates anyway" advice does not apply: there IS no downstream FPU op here.
    The tiled output IS the product, and for a byte re-lay the oracle is
    bit-identity.

    Deliberately NOT true for fp32 -> bf16/bfp: there fast tilize is available
    and its static_assert REQUIRES `UnpackToDestMode::Default` on the input CB,
    so tagging that cell would be a compile error (and the tf32 step is below the
    output format's own precision anyway).
    """
    return in_dtype == ttnn.float32 and out_dtype == ttnn.float32


def needs_srcb_alu_format_repair(in_dtype) -> bool:
    """True when the math thread's SrcB ALU-format field must be re-written.

    Wormhole B0 defect, mechanism named so this is not a magic flag:
    `_llk_math_hw_configure_` (tt_llk_wormhole_b0/llk_lib/llk_math_common.h) builds
    ONE config word as
        (srca_format << SrcA_SHAMT) | (srcb_format << SrcB_SHAMT) | int8_math
    and writes it under `SrcA_MASK | SrcB_MASK`, WITHOUT applying
    `masked_data_format()`. Both fields are 4 bits. `DataFormat::UInt8` is 30, so
    bit 4 of the SrcA value lands at the SrcB field's low bit, which the mask does
    not confine: SrcA comes out correct (14 = Int8) but SrcB comes out 15 instead
    of 14. The UInt8 datacopy MOP is ELWADD (it reads SrcB, which the tilize
    unpack MOP zero-fills), so a mis-typed SrcB is enough to zero the result —
    which is exactly what an unrepaired uint8 tilize produces.

    `reconfig_data_format_srcb(cb)` re-writes the SAME field under
    `SrcB_MASK | INT8_MASK` only, so the spill bit is masked away and the field
    lands at 14. That is the repair, and it is a public compute-API call, not raw
    LLK.

    Scoped as narrowly as the mechanism: a format whose enum exceeds the 4-bit
    field AND that reaches SrcA/SrcB at all. UInt32 (24) also spills, but
    `_llk_unpack_tilize_init_` routes UInt32/Int32 straight to DEST — SrcB is
    never read there, the corrupt field is inert, and those pairs are already
    bit-exact, so they are deliberately left alone.
    """
    return in_dtype == ttnn.uint8


def requires_fp32_dest_acc(in_dtype, out_dtype) -> bool:
    """Whether the datapath REQUIRES fp32 DEST at this dtype pair.

    Three sources, all of them the datapath's and none of them a preference:
      * fp32 input — the value only survives DEST at 32 bits (pre-existing rule).
      * fp32 -> fp32 — `Fp32Mode::Lossless` static_asserts `DST_ACCUM_MODE`.
      * an Int8/UInt8 format on a Src register — the LLK asserts
        "Reconfiguring math to/from Int8/UInt8/Int32 formats requires FP32 Dest
        mode enabled" (`llk_math_common.h`), which the SrcB repair above trips.

    A user `compute_kernel_config` may turn fp32 DEST ON but never OFF (see
    `create_program_descriptor`): tilize is value-preserving, so a config that
    would silently truncate is a wrong answer rather than a speed/precision
    trade the caller is entitled to make.
    """
    return (
        in_dtype == ttnn.float32
        or is_lossless_fp32_relay(in_dtype, out_dtype)
        or needs_srcb_alu_format_repair(in_dtype)
    )


def _largest_divisor_at_most(n: int, limit: int) -> int:
    """Coarsest divisor of `n` that is <= `limit`. Always exists (1 | n)."""
    limit = max(1, min(int(limit), int(n)))
    for d in range(limit, 0, -1):
        if n % d == 0:
            return d
    return 1


def _cores_to_range_set(cores) -> "ttnn.CoreRangeSet":
    """A CoreRangeSet over `cores` (a row-wise-ordered list), merging runs."""
    ranges = set()
    start = prev = None
    for c in cores:
        if start is not None and int(c.y) == int(prev.y) and int(c.x) == int(prev.x) + 1:
            prev = c
            continue
        if start is not None:
            ranges.add(ttnn.CoreRange(start, prev))
        start = prev = c
    if start is not None:
        ranges.add(ttnn.CoreRange(start, prev))
    return ttnn.CoreRangeSet(ranges)


def _contiguous_assignment(cores, num_blocks: int, first_block_id: int = 0):
    """`num_blocks` split into contiguous per-core ranges over `cores`.

    The same balanced rule `split_work_to_cores` uses (the first `n % k` cores
    take one extra), written out because the two column families are handed
    DISJOINT SUBSETS of the grid rather than a whole `CoreCoord` grid.
    Returns `[(core, start_block_id, num_blocks, block_stride=1), ...]` for the
    cores that actually got work, and the CoreRangeSet over exactly those.
    """
    used = min(len(cores), num_blocks)
    if used == 0:
        return [], _cores_to_range_set([])
    base, rem = divmod(num_blocks, used)
    assignment = []
    start = first_block_id
    for i in range(used):
        per_core = base + (1 if i < rem else 0)
        assignment.append((cores[i], start, per_core, 1))
        start += per_core
    assert start == first_block_id + num_blocks
    return assignment, _cores_to_range_set(cores[:used])


class ColumnGroup:
    """One core range's view of the column axis — the ragged tail's carrier.

    A plan has ONE of these when `C % block_width_tiles == 0` (the full-width
    group, `col_tile_offset == 0`) and TWO when it does not: the full-width
    cores and the tail cores. Every field is exactly the per-core-range slice of
    what used to be a single global value, which is what lets one core see one
    CB push/pop quantum (see the module docstring's wrap invariant).
    """

    __slots__ = (
        "block_width_tiles",
        "num_w_chunks",
        "col_tile_offset",
        "block_row_bytes",
        "write_rows_per_barrier",
        "num_blocks",
        "cores",
        "assignment",
        "input_depth_rows",
        "output_depth_batches",
        "is_retile",
        "pad_active",
        "split_reader",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def input_cb_pages(self) -> int:
        if self.is_retile:
            return 1
        return self.input_depth_rows * self.block_width_tiles

    @property
    def split_cb_pages(self) -> int:
        """cb_input_rows_split: the writer-side half-block, whole (Refinement 6)."""
        return self.split_reader * self.block_width_tiles

    @property
    def output_cb_pages(self) -> int:
        return self.output_depth_batches * self.write_rows_per_barrier * self.block_width_tiles

    @property
    def pad_row_bytes(self) -> int:
        return self.block_row_bytes if self.pad_active else 0


def _split_reader_rows(
    *,
    block_width_tiles: int,
    block_row_bytes: int,
    tensor_row_blocks: int,
    num_row_groups: int,
    num_blocks_total: int,
    is_retile: bool,
    pad_active: bool,
    input_native: bool,
    output_native: bool,
    input_pages_per_row: int,
    in_page_bytes: int,
    out_page_bytes: int,
    output_depth_batches: int,
    write_rows_per_barrier: int,
    input_depth_rows: int,
    budget: int,
) -> int:
    """Tile-rows of the block the WRITER reads, or 0 for "do not split".

    The return value IS `cb_input_rows_split`'s depth in tile-rows, and it is
    the writer's WHOLE half-block on purpose: a shallower split CB would let the
    writer block in `cb_reserve_back` while compute blocks in the output CB's
    `cb_reserve_back`, which is a cycle. Sized to the whole half, the writer
    never waits for compute before it starts storing, so there is no cycle to
    close.

    Every condition below is a correctness or a payoff precondition:
      * only the plain stick path — the padded, retile, native-shard and
        sub-row-paged legs are different block operations, and duplicating one
        of them into the writer would be a second implementation, not a knob;
      * a NON-native output, because the producer of `cb_input_rows_split` IS
        the writer kernel — and a natively sharded output emits no writer kernel
        at all (the packer has already written the shard), so the split would
        leave compute waiting forever on a CB with no producer. That leg is a
        HANG, not a slowdown;
      * the TALLEST block at least two tile-rows tall, so the writer's half is
        bounded by `floor(max_extent * pct / 100)` and the split CB can be sized
        from it. A SHORTER block in the same plan may still round its own share
        to zero, which all three kernels handle identically (the writer skips its
        read, compute skips its second sub-block, the reader takes the whole
        block) — the split point is derived from each block's own extent;
      * a read transaction small enough to be ISSUE-dominated (see the constant);
      * the extra CB fits the same L1 budget the column extent was solved against.
    """
    if SPLIT_READER_MAX_ROW_BYTES <= 0 or num_blocks_total == 0:
        return 0
    if is_retile or pad_active or input_native or output_native or input_pages_per_row > 1:
        return 0
    if block_row_bytes > SPLIT_READER_MAX_ROW_BYTES:
        return 0
    max_extent = math.ceil(tensor_row_blocks / num_row_groups)
    if max_extent < 2:
        return 0
    rows = min(max_extent - 1, (max_extent * SPLIT_READER_WRITER_SHARE_PCT) // 100)
    if rows < 1:
        return 0
    footprint = (
        (input_depth_rows + rows) * block_width_tiles * in_page_bytes
        + output_depth_batches * write_rows_per_barrier * block_width_tiles * out_page_bytes
    )
    return rows if footprint <= budget else 0


class TilizePlan:
    """The derived block plan: every knob, plus the per-core block assignment.

    Memoized (see `_PLAN_CACHE`) so a repeat call at the same shape / dtype /
    memory_config / tile / low_l1 / grid does NOT re-enter the W_FIT solve or
    `split_work_to_cores` — that derivation is exactly what the program-cache
    rule says must not be redone per call.
    """

    __slots__ = (
        "tile_h",
        "tensor_row_blocks",
        "tensor_col_tiles",
        "block_width_tiles",
        "num_w_chunks",
        "num_row_groups",
        "num_blocks_total",
        "write_rows_per_barrier",
        "input_depth_rows",
        "output_depth_batches",
        "in_page_bytes",
        "out_page_bytes",
        "block_row_bytes",
        # --- ragged column tail (Refinement 6). `tail_group` is None on every
        # plan whose `C` divides by `block_width_tiles` (which is every smooth
        # geometry, and every sharded / sub-row-paged one by construction), so
        # the single-core-range program is unchanged. When it is present the
        # program carries a SECOND core range, running the same three kernels at
        # the tail's own `block_width_tiles` / `col_tile_offset` / CB sizes.
        "tail_group",
        # --- split reader (Refinement 6). The number of the block's tile-rows
        # whose sticks the WRITER kernel reads (into cb_input_rows_split) so the
        # two data-movement RISC-Vs share the NoC issue cost. 0 = off, and every
        # split-related CB, CT arg and kernel branch compiles out.
        "split_reader",
        "all_cores",
        # [(core, start_block_id, num_blocks, block_stride), ...]. `block_stride`
        # is 1 for the solved plan (contiguous ranges) and `num_cores` for the
        # shard-driven plan, which is exactly the ROUND_ROBIN_1D placement an ND
        # spec uses and degenerates to "one block per core" for a legacy 2-D one.
        "assignment",
        # --- numerical policy (Refinement 5). All three are derived from the
        # (input dtype, output dtype) pair by the predicates above; none of them
        # selects a different code path, only the compute config the datapath
        # needs to stay value-preserving at that pair.
        "fp32_dest_acc_en",
        "lossless_fp32",
        "repair_srcb_alu_format",
        # Zero-copy: the named side's CB is placed ON the tensor's own shard
        # buffer, so that side does no NoC access at all.
        "input_native",
        "output_native",
        # How the ROW_MAJOR source pages a row (1 = a page IS a row). > 1 puts
        # the reader on its strided branch.
        "input_pages_per_row",
        "in_page_width_bytes",
        # --- retile (Refinement 4): a TILE input re-laid at another tile
        # height. `in_tile_h` is 0 on the ROW_MAJOR path, where there is no
        # input tile geometry at all.
        "is_retile",
        "in_tile_h",
        "in_tile_rows_per_image",
        # --- padding (Refinement 2). All inert when `pad_active` is False, and
        # the reader's non-padded branches are then byte-identical to Phase 0. ---
        "pad_active",
        "pad_word",
        "elem_size",
        "in_num_images",
        "in_rows_per_image",
        "in_row_bytes",
        "rows_per_image_out",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    # --- derived L1 footprint, in pages and bytes --------------------------
    @property
    def full_group(self) -> "ColumnGroup":
        """The full-width core range, as the same object shape the tail uses."""
        return ColumnGroup(
            block_width_tiles=self.block_width_tiles,
            num_w_chunks=self.num_w_chunks,
            col_tile_offset=0,
            block_row_bytes=self.block_row_bytes,
            write_rows_per_barrier=self.write_rows_per_barrier,
            num_blocks=self.num_blocks_total,
            cores=self.all_cores,
            assignment=self.assignment,
            input_depth_rows=self.input_depth_rows,
            output_depth_batches=self.output_depth_batches,
            is_retile=self.is_retile,
            pad_active=self.pad_active,
            split_reader=self.split_reader,
        )

    @property
    def groups(self) -> list:
        """Every core range this program covers — one, or two with a ragged tail."""
        return [self.full_group] + ([self.tail_group] if self.tail_group is not None else [])

    @property
    def num_cores_used(self) -> int:
        return sum(len(g.assignment) for g in self.groups)

    @property
    def input_cb_pages(self) -> int:
        # The retile path never reads through cb_input_rows (the reader
        # assembles output tiles straight into cb_output_tiles), so the CB
        # shrinks to the ONE page that keeps its JIT tile/format descriptor
        # well-formed for the reader's other, compile-time-discarded branches.
        if self.is_retile:
            return 1
        return self.input_depth_rows * self.block_width_tiles

    @property
    def split_cb_pages(self) -> int:
        return self.split_reader * self.block_width_tiles

    @property
    def output_cb_pages(self) -> int:
        return self.output_depth_batches * self.write_rows_per_barrier * self.block_width_tiles

    @property
    def pad_row_bytes(self) -> int:
        """cb_pad_row's single page: ONE block row of the fill (0 when unpadded)."""
        return self.block_row_bytes if self.pad_active else 0

    @property
    def l1_per_core_bytes(self) -> int:
        """Worst per-core footprint over the plan's core ranges (the tail's is
        never larger — it is the same depths at a narrower block)."""
        return (
            (self.input_cb_pages + self.split_cb_pages) * self.in_page_bytes
            + self.output_cb_pages * self.out_page_bytes
            + self.pad_row_bytes
        )


_PLAN_CACHE: dict = {}


# ---------------------------------------------------------------------------
# The shard partition — the block grid a sharded operand has ALREADY fixed
# ---------------------------------------------------------------------------


class ShardPartition:
    """A shard spec re-expressed as this op's block grid, in TILE units.

    tilize has no dependent axis: an output tile depends only on its own
    32-element column slice of its own `tile_h` sticks. So a shard IS a block —
    a HEIGHT shard is a `row_group`, a WIDTH shard a `w_chunk`, a BLOCK shard the
    2-D block the op already builds — and there is no cross-core combine, no
    mcast and no semaphore to design. What changes versus the solved plan is only
    WHERE the numbers come from.

    `cores[i]` is the core the i-th shard lives on, in the shard's own linear
    order (`shard_row * num_shard_cols + shard_col`, which is the order
    `buffer.cpp:core_to_host_pages` lays shards out in and therefore the order
    the L1 bytes are in). That linear index IS the `block_id` the three kernels
    already derive `(row_group, w_chunk)` from.
    """

    __slots__ = (
        "cores",
        "grid",
        "shard_rows_tiles",
        "shard_cols_tiles",
        "num_shard_rows",
        "num_shard_cols",
        "num_shards",
        "is_l1",
        "per_core_bytes",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def key(self) -> tuple:
        """Identity of the PARTITION (not of the spec that expressed it), so an
        ND spec and the legacy 2-D spec describing the same cut compare equal."""
        return (
            tuple((int(c.x), int(c.y)) for c in self.cores),
            self.shard_rows_tiles,
            self.shard_cols_tiles,
            self.num_shard_rows,
            self.num_shard_cols,
        )


def _folded_2d(padded_shape) -> tuple:
    """The (height, width) 2-D view a shard spec is written against: every
    leading dim folds into the height, exactly as R folds them into tile-rows."""
    height = 1
    for d in padded_shape[:-1]:
        height *= int(d)
    return height, int(padded_shape[-1])


def _shard_geometry(mem_config):
    """(shard_h, shard_w, grid, orientation) for either shard API, or None.

    A LIVE tensor normalizes an ND spec down to the equivalent legacy 2-D one
    whenever it has one (measured: `probes/probe_010.py`), so reading the live
    tensor is what makes `nd_same_spec` and `nd_in_legacy_out` compare equal
    without an API-shaped special case. The ND branch below is the residue: a
    spec that keeps a genuine ND shape.
    """
    shard_spec = getattr(mem_config, "shard_spec", None)
    if shard_spec is not None:
        shape = list(shard_spec.shape)
        return int(shape[-2]), int(shape[-1]), shard_spec.grid, shard_spec.orientation
    nd_spec = getattr(mem_config, "nd_shard_spec", None)
    if nd_spec is None:
        return None
    shape = [int(d) for d in list(nd_spec.shard_shape)]
    if len(shape) < 2 or any(d != 1 for d in shape[:-2]):
        # A shard that spans more than one leading-dim slice is not a contiguous
        # run of folded rows, so the 2-D block view does not hold.
        return None
    return shape[-2], shape[-1], nd_spec.grid, nd_spec.orientation


def shard_partition(tensor, tile_h: int):
    """`ShardPartition` for `tensor`, or None when its shards are not blocks.

    None means "this side cannot drive the block grid" — not sharded, a ragged
    partition (the last shard short, which would make `block_width_tiles` vary
    per core and it is a COMPILE-TIME template argument), or a shard that is not
    a whole number of tiles. The caller then falls back to the solved plan, which
    is correct for every placement because the accessor addresses a sharded
    tensor as readily as an interleaved one — just not natively.
    """
    if not tensor.is_sharded():
        return None
    mem_config = tensor.memory_config()
    geometry = _shard_geometry(mem_config)
    if geometry is None:
        return None
    shard_h, shard_w, grid, orientation = geometry

    height, width = _folded_2d(list(tensor.padded_shape))
    if shard_h <= 0 or shard_w <= 0:
        return None
    if shard_h % tile_h or shard_w % TILE_WIDTH:
        return None
    if height % shard_h or width % shard_w:
        return None  # ragged: block_width_tiles / block_row_extent would vary

    num_shard_rows = height // shard_h
    num_shard_cols = width // shard_w
    num_shards = num_shard_rows * num_shard_cols

    num_cores = grid.num_cores()
    if num_cores == 0 or num_shards < 1:
        return None
    # buffer.cpp: shards are placed on `corerange_to_cores(grid, n, row_major)`
    # in their linear order, row_wise iff the orientation is ROW_MAJOR.
    core_order = ttnn.corerange_to_cores(grid, num_cores, orientation == ttnn.ShardOrientation.ROW_MAJOR)
    cores = [core_order[i % num_cores] for i in range(num_shards)]

    total_bytes = int(tensor.buffer_num_pages()) * int(tensor.buffer_aligned_page_size())
    shards_per_core = math.ceil(num_shards / num_cores)
    return ShardPartition(
        cores=cores,
        grid=grid,
        shard_rows_tiles=shard_h // tile_h,
        shard_cols_tiles=shard_w // TILE_WIDTH,
        num_shard_rows=num_shard_rows,
        num_shard_cols=num_shard_cols,
        num_shards=num_shards,
        is_l1=(mem_config.buffer_type == ttnn.BufferType.L1),
        per_core_bytes=(total_bytes // num_shards) * shards_per_core,
    )


def _resident_l1_bytes(tensor, partition, num_cores: int) -> int:
    """Per-core L1 the tensor ITSELF holds, which the CB budget must not spend.

    `ttnn.get_max_worker_l1_unreserved_size()` reports the worker L1 arena, not
    what is free in it, and an L1-resident operand (every sharded one, by
    definition) is already sitting in that arena. l1_ledger.md finding, folded in
    here because a sharded refinement is where it starts to bite.
    """
    if tensor.memory_config().buffer_type != ttnn.BufferType.L1:
        return 0
    if partition is not None:
        return partition.per_core_bytes
    total = int(tensor.buffer_num_pages()) * int(tensor.buffer_aligned_page_size())
    return math.ceil(total / max(1, num_cores))


def _input_tile_height(input_tensor) -> int:
    """The input's own tile height, or 0 for a ROW_MAJOR input (no tile geometry).

    Non-zero IS the retile predicate: a TILE input is the only thing that puts
    the reader on its face-walking block operation.
    """
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        return 0
    tile = getattr(input_tensor, "tile", None)
    return int(tile.tile_shape[0]) if tile is not None else TILE_WIDTH


def plan_cache_key(input_tensor, output_tensor, *, low_l1: bool, grid, pad_value) -> tuple:
    """Hashable identity of everything the plan derivation reads."""
    return (
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        str(input_tensor.memory_config()),
        tuple(output_tensor.padded_shape),
        output_tensor.dtype,
        str(output_tensor.memory_config()),
        output_tensor.buffer_page_size(),
        (output_tensor.tile.tile_shape[0], output_tensor.tile.tile_shape[1]),
        # The INPUT's tile height selects the reader's block operation on the
        # retile path, so it belongs in the plan's identity.
        input_tensor.buffer_page_size(),
        _input_tile_height(input_tensor),
        bool(low_l1),
        (grid.x, grid.y),
        None if pad_value is None else float(pad_value),
    )


def derive_plan(input_tensor, output_tensor, *, low_l1: bool, grid, pad_value=None) -> TilizePlan:
    """Rule 2, in its two ordered steps: (1) fill the grid, (2) take the
    coarsest block that fits L1. Nothing here is restated from a literal."""
    key = plan_cache_key(input_tensor, output_tensor, low_l1=low_l1, grid=grid, pad_value=pad_value)
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    num_cores = int(grid.x) * int(grid.y)

    tile_h = int(output_tensor.tile.tile_shape[0])
    # `shape` is the INPUT's page grid (what the reader addresses); `out_shape`
    # is the OUTPUT's tile grid (what the block plan is cut on).
    shape = list(input_tensor.padded_shape)
    out_shape = list(output_tensor.padded_shape)

    # Geometry: R is per-image and folds the leading dims. NEVER
    # floor(num_images * H / tile_h) -- [8,1,249,2048] is the case that differs.
    #
    # Read off the OUTPUT's padded shape, not the input's. The tile grid IS the
    # output, and the two shapes are NOT interchangeable once a shard is in play:
    # a ROW_MAJOR tensor's padded shape rounds its last dim up to its PAGE width,
    # which a width-cutting shard sets (`[3,160,160]` sharded 64 wide reports
    # `[3,160,192]`), while a TILE tensor rounds to the tile. Taking C from the
    # input there yields 6 columns for a 5-column output — measured as the
    # `test_tilize_nd_sharded` value mismatch.
    num_images = 1
    for d in out_shape[:-2]:
        num_images *= int(d)
    rows_per_image = math.ceil(int(out_shape[-2]) / tile_h)
    tensor_row_blocks = num_images * rows_per_image  # R
    tensor_col_tiles = math.ceil(int(out_shape[-1]) / TILE_WIDTH)  # C

    # Page bytes. tb_in is one tile's worth of ROW-MAJOR bytes; tb_out is one
    # whole output tile page, read off the buffer so block-float / tiny-tile
    # page sizes are exact rather than computed.
    in_page_bytes = tile_h * TILE_WIDTH * input_tensor.element_size()
    out_page_bytes = output_tensor.buffer_page_size()

    # --- how the ROW_MAJOR source is PAGED --------------------------------
    # A stick-indexed read (`start_page + row`) is only a stick index when a page
    # IS a whole row — true for interleaved ROW_MAJOR and for a HEIGHT-sharded
    # one. A shard that cuts the width pages by SHARD width instead, so one row
    # spans `input_pages_per_row` pages and consecutive sticks are that far
    # apart. Native consumption sidesteps it entirely (no read at all); where
    # native is unavailable the reader takes its strided branch, for which the
    # block's row segment must sit inside a single source page.
    in_page_width_elems = int(input_tensor.buffer_page_size()) // input_tensor.element_size()
    input_pages_per_row = max(1, math.ceil(int(shape[-1]) / max(1, in_page_width_elems)))

    # --- retile: a TILE input, re-laid at another tile height --------------
    # The input's pages are whole TILES, not sticks, so `input_pages_per_row`
    # (a stick-paging quantity) is meaningless here and every stick-indexed
    # branch of the reader is off. What replaces them is one derived unit: the
    # largest byte run contiguous in both tile layouts (`retile_copy_unit`),
    # which the reader gathers straight into the OUTPUT tile — no row-major
    # intermediate, no compute stage, and never an untilize/tilize round trip
    # (op_design.md ranks that `rejected` at 2x the minimum DRAM traffic).
    in_tile_h = _input_tile_height(input_tensor)
    is_retile = in_tile_h > 0
    if is_retile:
        input_pages_per_row = 1

    # --- the pad region: what the output grid covers MINUS the input -------
    # Padding is `grid2d_padded` in op_design.md, and it is ADDITIVE on the
    # block that already exists: the block grid, the core assignment, the CBs
    # and the compute call are untouched; only `load_block` gains a fill.
    #
    # The pad extent is derived from the two shapes rather than from the
    # request, and the two agree by construction: `validate()` refuses a
    # non-tile-aligned input with no padding argument, so a pad REGION exists iff
    # the output's padded grid reaches past the input's LOGICAL extent. Reading
    # it off the geometry is what keeps `pad_active` False — and the Phase 0
    # reader branches byte-identical — on every already-aligned call, padded or
    # not (`explicit` at exactly the tile round asks for no fill and gets none).
    #
    # LOGICAL, not padded, on the input side: the pad boundary is where the
    # caller's data ends, and a ROW_MAJOR tensor's padded shape rounds its last
    # dim up to its PAGE width (a width-cutting shard sets that), which would
    # claim page padding as real data.
    elem_size = int(input_tensor.element_size())
    in_logical = [int(d) for d in list(input_tensor.shape)]
    in_logical = [1] * (2 - len(in_logical)) + in_logical  # rank 0/1: the pad SYNTHESIZES both tile dims
    in_num_images = 1
    for d in in_logical[:-2]:
        in_num_images *= d
    in_rows_per_image = in_logical[-2]
    in_row_bytes = in_logical[-1] * elem_size
    pad_active = (
        in_num_images < num_images
        or in_rows_per_image < rows_per_image * tile_h
        or in_row_bytes < tensor_col_tiles * TILE_WIDTH * elem_size
    )
    pad_word = pad_fill_word(input_tensor.dtype, pad_value, elem_size) if pad_active else 0

    # The input's tile-rows PER IMAGE. The output's tile grid and the input's do
    # NOT have to agree on the padded per-image height (`H=20` at in_tile_h=8 is
    # three input tile-rows, at out tile_h=4 five output tile-rows), so the
    # reader splits a folded output tile-row per IMAGE before turning it into an
    # input tile-row — the same segmentation the padded branch does, and the
    # reason no host guard on "the two padded heights must match" is needed.
    in_tile_rows_per_image = math.ceil(in_rows_per_image / in_tile_h) if is_retile else 0
    if is_retile:
        # A pure byte re-lay assembles the output tile's bytes from the input
        # tiles' faces; both preconditions below are what make "byte" the right
        # unit. They cannot be reached from SUPPORTED today (dtype and
        # output_dtype are both bfloat16, and retile x padding is EXCLUDED), and
        # they are asserted rather than assumed so the dtype refinement trips
        # here instead of producing wrong bytes.
        if input_tensor.dtype != output_tensor.dtype:
            raise RuntimeError(
                f"tilize: re-tiling {in_tile_h}->{tile_h} with a dtype cast "
                f"({input_tensor.dtype} -> {output_tensor.dtype}) is not implemented; the retile "
                "path is a pure byte re-lay and carries no pack-time conversion stage"
            )
        if (
            int(input_tensor.buffer_page_size()) != in_tile_h * TILE_WIDTH * elem_size
            or int(out_page_bytes) != tile_h * TILE_WIDTH * elem_size
        ):
            raise RuntimeError(
                "tilize: re-tiling needs densely packed tile pages on both sides "
                f"(in {input_tensor.buffer_page_size()} B vs {in_tile_h * TILE_WIDTH * elem_size}, "
                f"out {out_page_bytes} B vs {tile_h * TILE_WIDTH * elem_size}); a block-float or "
                "otherwise padded tile page carries per-face exponents the face walk does not move"
            )
        if pad_active:
            raise RuntimeError(
                "tilize: re-tiling a padded region is not implemented (EXCLUSIONS covers the cell); "
                "the fill would have to be written into output faces the face walk never sources"
            )

    # --- who fixes the block grid: a shard spec, or the L1 solve? ---------
    # tilize has no dependent axis, so a shard is already a block. When one side
    # is L1-sharded its OWN partition is used verbatim and its CB is placed on
    # its buffer (zero-copy); the other side falls back to the accessor. Both
    # sides go native only when the two partitions are the SAME cut, which is
    # what makes a same-spec call touch the NoC not at all.
    def _partition_tiles_the_grid(part):
        """A partition can only DRIVE the plan if its shards tile the output grid
        exactly. A shard whose last column is part page-padding (a ROW_MAJOR
        tensor cut 64 wide across a 160-wide row) covers more tile columns than
        the output has, so it is a placement the accessor still serves but the
        block grid cannot be read off."""
        return part is not None and (
            part.num_shard_rows * part.shard_rows_tiles == tensor_row_blocks
            and part.num_shard_cols * part.shard_cols_tiles == tensor_col_tiles
        )

    in_partition = shard_partition(input_tensor, tile_h)
    out_partition = shard_partition(output_tensor, tile_h)
    if not _partition_tiles_the_grid(in_partition):
        in_partition = None
    if not _partition_tiles_the_grid(out_partition):
        out_partition = None
    partition = None
    input_native = False
    output_native = False
    if is_retile:
        # Retile x sharded is EXCLUDED (see tilize.EXCLUSIONS): the face walk is
        # written against the interleaved TILE page index (`tile_row * C + col`)
        # and a zero-copy CB over a resident TILE shard would need the shard's
        # own page map on both sides. Forcing both partitions off keeps the
        # solved plan — which is correct for every placement — as the only
        # retile path, rather than half-wiring a native one no test can reach.
        in_partition = out_partition = None
    if pad_active:
        # A zero-copy input CB IS the resident shard, and the shard holds only
        # the caller's own bytes — there is nowhere in it to put the fill, and
        # nothing that could put it there without corrupting the input tensor.
        # So a padded call reads its input through the accessor and fills a
        # scratch CB. The OUTPUT side is unaffected: the packer writes whole
        # tiles, pad positions included, straight into the output shard.
        in_partition = None
    if in_partition is not None and in_partition.is_l1:
        partition, input_native = in_partition, True
        output_native = out_partition is not None and out_partition.is_l1 and out_partition.key == in_partition.key
    elif out_partition is not None and out_partition.is_l1:
        partition, output_native = out_partition, True

    if partition is not None and partition is out_partition and input_pages_per_row > 1:
        # The output shard would fix a block width the strided read cannot honour
        # (see below): a block row segment has to sit inside ONE source page, and
        # only the solved plan is free to choose a width that does. Correctness
        # over the output's zero-copy, in a mix (sub-row-paged non-native input +
        # L1-sharded output on a different cut) no test in the suite reaches.
        partition, output_native = None, False

    if not input_native and input_pages_per_row > 1 and in_page_width_elems % TILE_WIDTH:
        raise RuntimeError(
            f"tilize: input's ROW_MAJOR page is {in_page_width_elems} elements, which is neither a "
            f"whole {shape[-1]}-element row nor a whole number of {TILE_WIDTH}-element tiles; the "
            "block's row segment cannot be addressed within one source page"
        )

    # --- L1 budget. The arena minus what the OPERANDS already hold ---------
    # An L1-resident tensor sits in the same worker arena the CBs are cut from,
    # and every sharded operand is L1-resident by definition, so a budget that
    # ignores it over-promises exactly where a sharded call needs it most.
    budget = ttnn.get_max_worker_l1_unreserved_size()
    budget -= _resident_l1_bytes(input_tensor, in_partition, num_cores)
    budget -= _resident_l1_bytes(output_tensor, out_partition, num_cores)
    budget = max(budget, in_page_bytes + out_page_bytes)

    input_depth_rows = INPUT_DEPTH_ROWS
    output_depth_batches = OUTPUT_DEPTH_BATCHES
    # Ragged column tail (Refinement 6): only the solved branch can produce one.
    # A shard IS the block grid (its chunks are uniform by construction) and the
    # empty grid has no columns at all.
    tail_width_tiles = 0
    num_tail_blocks = 0
    tail_assignment: list = []
    tail_cores = None

    if tensor_row_blocks == 0 or tensor_col_tiles == 0:
        # --- the empty tensor: a grid with NO output tiles -----------------
        # `torch.rand(0)` tilized is a legal, if degenerate, request: the output
        # is a TILE tensor with zero elements. It has no blocks, so there is
        # nothing to solve and nothing to distribute — and the column solve
        # below divides by a target derived from `tensor_col_tiles`, which is
        # where an unguarded empty grid turns into a host-side ZeroDivisionError.
        #
        # The op still DISPATCHES: one core is handed `num_blocks = 0`, all
        # three kernels' block loops run zero iterations, and no NoC access is
        # ever issued against the zero-page buffers. That keeps "exactly one
        # native dispatch per invocation" true for the empty case as well, and
        # keeps every extent at its minimum legal value (1) so the CB
        # descriptors and the kernels' compile-time divisors stay well-formed.
        block_width_tiles = 1
        num_w_chunks = 1
        num_row_groups = 1
        num_blocks_total = 0
        write_rows_per_barrier = 1
        # A zero-page buffer has no resident shard to place a CB on, whatever
        # its memory_config says.
        input_native = output_native = False
        all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        assignment = [(ttnn.CoreCoord(0, 0), 0, 0, 1)]
    elif partition is not None:
        # --- grid2d_sharded: every number below is READ, not solved. -------
        block_width_tiles = partition.shard_cols_tiles
        num_w_chunks = partition.num_shard_cols
        num_row_groups = partition.num_shard_rows
        num_blocks_total = partition.num_shards
        write_rows_per_barrier = max(1, math.ceil(WRITE_BATCH_MIN_TILES / block_width_tiles))

        # Only the NON-native side costs CB L1; the native side's CB is the
        # tensor. Shrink the depth knobs (never the block) if the pair overruns.
        def _scratch_bytes(depth_in, depth_out):
            total = 0
            if pad_active:
                total += block_width_tiles * (in_page_bytes // tile_h)  # cb_pad_row
            if not input_native:
                total += depth_in * block_width_tiles * in_page_bytes
            if not output_native:
                total += depth_out * write_rows_per_barrier * block_width_tiles * out_page_bytes
            return total

        while _scratch_bytes(input_depth_rows, output_depth_batches) > budget and (
            input_depth_rows > 1 or output_depth_batches > 1
        ):
            if input_depth_rows >= output_depth_batches and input_depth_rows > 1:
                input_depth_rows -= 1
            else:
                output_depth_batches -= 1

        # One block per shard, handed to the core the shard is resident on. The
        # shard's linear index IS the block id, and the ROUND_ROBIN_1D placement
        # both shard APIs use makes core i own shards {i, i+N, i+2N, ...} — a
        # start plus a stride, which is also the identity for a legacy 2-D spec
        # (N == num_shards, so each core owns exactly one).
        all_cores = partition.grid
        stride = partition.grid.num_cores()
        assignment = []
        for core_index, core in enumerate(ttnn.corerange_to_cores(all_cores, stride, True)):
            blocks_here = len(range(core_index, num_blocks_total, stride)) if core_index < num_blocks_total else 0
            assignment.append((core, core_index, blocks_here, stride))
        assert (
            sum(a[2] for a in assignment) == num_blocks_total
        ), f"tilize: shard assignment covered {sum(a[2] for a in assignment)} of {num_blocks_total} shards"
    else:
        # --- the column-extent L1 bound, in closed form -------------------
        # footprint = INPUT_DEPTH_ROWS*W*tb_in + OUTPUT_DEPTH_BATCHES*wrpb*W*tb_out
        # and wrpb*W <= WRITE_BATCH_MIN_TILES + W, so
        #   footprint <= W*(2*tb_in + 2*tb_out) + 2*WRITE_BATCH_MIN_TILES*tb_out
        # which inverts to W_FIT below. Budget is read from the device, never a
        # literal; no tensor dimension appears.
        # On the retile path cb_input_rows is a one-page stub (see
        # TilizePlan.input_cb_pages), so only the output CB scales with the
        # column extent and the solve gets the whole budget for it.
        denom = OUTPUT_DEPTH_BATCHES * out_page_bytes
        if not is_retile:
            denom += INPUT_DEPTH_ROWS * in_page_bytes
        if pad_active:
            # cb_pad_row is ONE block row: `block_width_tiles * TILE_WIDTH * elem`
            # bytes, i.e. `in_page_bytes / tile_h` per tile of block width. Small,
            # but it scales with the same knob, so it belongs in the same solve
            # rather than being spent behind the budget's back.
            denom += in_page_bytes // tile_h
        headroom = budget - OUTPUT_DEPTH_BATCHES * WRITE_BATCH_MIN_TILES * out_page_bytes
        w_fit = max(1, min(headroom // denom, FAST_TILIZE_WIDTH_CAP))
        w_cap = min(w_fit, LOW_L1_WIDTH_CAP) if low_l1 else w_fit

        # Step 1 (occupancy + pipeline waves) and step 2 (L1 fit) -> the smallest
        # column split that satisfies both. num_w_chunks is MINIMIZED: a column cut
        # multiplies the read transaction count, so it is spent only where it is
        # required. The tensor holds `R * num_w_chunks` tile-rows, so ONE
        # expression covers both demands on the column axis — fill the grid
        # (`waves == 1`) and fill each core's pipe (`waves > 1`, which is what
        # buys read/write overlap on a geometry whose R cannot supply a second
        # wave by itself).
        w_chunks_for_l1 = math.ceil(tensor_col_tiles / w_cap)
        # On a sub-row-paged source the width must ALSO divide the page width in
        # tiles, so `w_chunk * block_row_bytes` lands at a page boundary plus an
        # offset that leaves the whole segment inside that page. Expressed by
        # narrowing what the divisor is taken OF — gcd(C, page width) — so the
        # search itself stays the single `_largest_divisor_at_most` knob, and
        # degenerates to C exactly when a page is a whole row.
        page_width_tiles = in_page_width_elems // TILE_WIDTH if input_pages_per_row > 1 else tensor_col_tiles
        width_source = math.gcd(tensor_col_tiles, page_width_tiles)

        # A RAGGED TAIL is only legal where the width faces no constraint of its
        # own: a sub-row-paged source additionally needs the block's row segment
        # inside ONE page, which is a property of the width and not of the
        # per-core mix, so that leg keeps the divisor rule.
        ragged_tail_ok = RAGGED_COLUMN_TAIL and input_pages_per_row == 1

        def _width_at(waves: int) -> int:
            """Coarsest column extent that fits L1 and yields >= the target chunks.

            `ceil(C / target)` is the design's own extent. Where the tail it
            leaves cannot be carried (a sub-row-paged source, or the knob off)
            this falls back to the coarsest DIVISOR of `C` — see this module's
            docstring for the CB-wrap mechanism and how two core ranges satisfy
            it without weakening it.
            """
            w_chunks_for_waves = min(tensor_col_tiles, math.ceil(num_cores * waves / tensor_row_blocks))
            target = max(w_chunks_for_l1, w_chunks_for_waves)
            if ragged_tail_ok:
                return max(1, min(w_cap, tensor_col_tiles, math.ceil(tensor_col_tiles / target)))
            return _largest_divisor_at_most(width_source, min(w_cap, tensor_col_tiles // target))

        # The wave count is bought FROM the column axis, so each doubling halves
        # the read transaction — the trade this knob makes. `MIN_BLOCK_ROW_BYTES`
        # is where the trade stops paying; take the deepest pipe whose read still
        # clears it, and fall back to the occupancy-only cut when none does.
        #
        # The ladder is HALVED, not decremented: a wave is bought by halving the
        # block width, so `{cap, cap/2, ..., 2}` are the settings
        # `MIN_BLOCK_ROW_BYTES` was calibrated against (Refinement 3 measured
        # w1/w2/w4/w8). Under the divisor rule a non-power-of-two wave count
        # almost always collapsed onto its neighbour's divisor, so walking every
        # integer was harmless; under the ragged rule it is a genuinely new
        # candidate, and the one it produces on `C = 1572` (waves 3 -> a 576 B
        # read) measured 29058 ns against 27035 for the power-of-two neighbour's
        # 832 B — i.e. the uncalibrated rung is the worst of the three. Halving
        # keeps the ladder and the floor on the same footing.
        block_width_tiles = _width_at(1)
        waves = PIPELINE_WAVES_PER_CORE
        while waves > 1:
            candidate = _width_at(waves)
            if candidate * TILE_WIDTH * elem_size >= MIN_BLOCK_ROW_BYTES:
                block_width_tiles = candidate
                break
            waves //= 2
        # The column cut, now in two families. `tail_width_tiles == 0` is the
        # smooth case and is byte-identical to the divisor rule.
        num_w_chunks = tensor_col_tiles // block_width_tiles
        tail_width_tiles = tensor_col_tiles - num_w_chunks * block_width_tiles

        # Occupancy counts BOTH families' chunks — the tail chunk is a column of
        # the grid like any other, it is only carried by different cores.
        total_w_chunks = num_w_chunks + (1 if tail_width_tiles else 0)
        num_row_groups = min(tensor_row_blocks, max(1, math.ceil(num_cores / total_w_chunks)))
        num_blocks_total = num_row_groups * num_w_chunks
        num_tail_blocks = num_row_groups if tail_width_tiles else 0

        # Transactions-in-flight knob for the writer. Inert (1) once the block is
        # at least WRITE_BATCH_MIN_TILES wide; the whole knob on a C == 1 tensor.
        write_rows_per_barrier = max(1, math.ceil(WRITE_BATCH_MIN_TILES / block_width_tiles))

        # --- core assignment over the linearized block id -----------------
        # row_wise=True explicitly: a column line of cores measured 2.91x worse
        # than a row line on an interleaved DRAM->DRAM copy, and row_wise=False is
        # the default that hands you the column.
        if num_tail_blocks == 0:
            (
                _num_cores_used,
                all_cores,
                core_group_1,
                core_group_2,
                blocks_per_core_g1,
                blocks_per_core_g2,
            ) = ttnn.split_work_to_cores(grid, num_blocks_total, row_wise=True)

            assignment = []
            start = 0
            for group, per_core in ((core_group_1, blocks_per_core_g1), (core_group_2, blocks_per_core_g2)):
                if per_core == 0:
                    continue
                for core in ttnn.corerange_to_cores(group, None, True):
                    assignment.append((core, start, per_core, 1))
                    start += per_core
            assert start == num_blocks_total, f"tilize: block assignment covered {start} of {num_blocks_total} blocks"
        else:
            # TWO DISJOINT CORE RANGES. The grid's row-wise core order is split
            # at one point: a prefix carries the full-width blocks, the suffix
            # carries the tail blocks. The cut is chosen to minimize the busiest
            # core's TILE count — the two families' blocks are different sizes,
            # so an even split of the BLOCK count would not balance the work.
            all_grid_cores = ttnn.corerange_to_cores(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}),
                num_cores,
                True,
            )

            def _busiest_tiles(cores_for_tail: int) -> int:
                cores_for_full = num_cores - cores_for_tail
                full = math.ceil(num_blocks_total / min(cores_for_full, num_blocks_total)) * block_width_tiles
                tail = math.ceil(num_tail_blocks / min(cores_for_tail, num_tail_blocks)) * tail_width_tiles
                return max(full, tail)

            tail_cores_count = min(
                range(1, min(num_tail_blocks, num_cores - 1) + 1),
                key=lambda k: (_busiest_tiles(k), k),
            )
            full_cores_list = all_grid_cores[: num_cores - tail_cores_count]
            tail_cores_list = all_grid_cores[num_cores - tail_cores_count :]

            assignment, all_cores = _contiguous_assignment(full_cores_list, num_blocks_total)
            tail_assignment, tail_cores = _contiguous_assignment(tail_cores_list, num_tail_blocks)

    # --- split reader (Refinement 6) --------------------------------------
    # Both data-movement RISC-Vs issue the block's stick reads instead of one.
    # MEASURED gate, not a guess: it is worth it exactly where ONE reader RISC-V
    # is issue-bound, which for this op means a read transaction small enough
    # that the per-stick issue cost dominates the bytes. See the constant.
    split_reader = _split_reader_rows(
        block_width_tiles=block_width_tiles,
        block_row_bytes=block_width_tiles * TILE_WIDTH * elem_size,
        tensor_row_blocks=tensor_row_blocks,
        num_row_groups=num_row_groups,
        num_blocks_total=num_blocks_total,
        is_retile=is_retile,
        pad_active=pad_active,
        input_native=input_native,
        output_native=output_native,
        input_pages_per_row=input_pages_per_row,
        in_page_bytes=in_page_bytes,
        out_page_bytes=out_page_bytes,
        output_depth_batches=output_depth_batches,
        write_rows_per_barrier=write_rows_per_barrier,
        input_depth_rows=input_depth_rows,
        budget=budget,
    )

    tail_group = None
    if num_tail_blocks:
        tail_group = ColumnGroup(
            block_width_tiles=tail_width_tiles,
            num_w_chunks=1,
            col_tile_offset=num_w_chunks * block_width_tiles,
            block_row_bytes=tail_width_tiles * TILE_WIDTH * elem_size,
            write_rows_per_barrier=max(1, math.ceil(WRITE_BATCH_MIN_TILES / tail_width_tiles)),
            num_blocks=num_tail_blocks,
            cores=tail_cores,
            assignment=tail_assignment,
            input_depth_rows=input_depth_rows,
            output_depth_batches=output_depth_batches,
            is_retile=is_retile,
            pad_active=pad_active,
            # The split is a per-core-range decision like every other column
            # knob; the tail's block is narrower, so it re-runs the same gate.
            split_reader=_split_reader_rows(
                block_width_tiles=tail_width_tiles,
                block_row_bytes=tail_width_tiles * TILE_WIDTH * elem_size,
                tensor_row_blocks=tensor_row_blocks,
                num_row_groups=num_row_groups,
                num_blocks_total=num_tail_blocks,
                is_retile=is_retile,
                pad_active=pad_active,
                input_native=input_native,
                output_native=output_native,
                input_pages_per_row=input_pages_per_row,
                in_page_bytes=in_page_bytes,
                out_page_bytes=out_page_bytes,
                output_depth_batches=output_depth_batches,
                write_rows_per_barrier=max(1, math.ceil(WRITE_BATCH_MIN_TILES / tail_width_tiles)),
                input_depth_rows=input_depth_rows,
                budget=budget,
            ),
        )

    plan = TilizePlan(
        tile_h=tile_h,
        tensor_row_blocks=tensor_row_blocks,
        tensor_col_tiles=tensor_col_tiles,
        block_width_tiles=block_width_tiles,
        num_w_chunks=num_w_chunks,
        num_row_groups=num_row_groups,
        num_blocks_total=num_blocks_total,
        write_rows_per_barrier=write_rows_per_barrier,
        input_depth_rows=input_depth_rows,
        output_depth_batches=output_depth_batches,
        in_page_bytes=in_page_bytes,
        out_page_bytes=out_page_bytes,
        block_row_bytes=block_width_tiles * TILE_WIDTH * input_tensor.element_size(),
        tail_group=tail_group,
        split_reader=split_reader,
        all_cores=all_cores,
        assignment=assignment,
        fp32_dest_acc_en=requires_fp32_dest_acc(input_tensor.dtype, output_tensor.dtype),
        lossless_fp32=is_lossless_fp32_relay(input_tensor.dtype, output_tensor.dtype),
        repair_srcb_alu_format=needs_srcb_alu_format_repair(input_tensor.dtype),
        input_native=input_native,
        output_native=output_native,
        input_pages_per_row=input_pages_per_row,
        in_page_width_bytes=in_page_width_elems * input_tensor.element_size(),
        is_retile=is_retile,
        in_tile_h=in_tile_h,
        in_tile_rows_per_image=in_tile_rows_per_image,
        pad_active=pad_active,
        pad_word=pad_word,
        elem_size=elem_size,
        in_num_images=in_num_images,
        in_rows_per_image=in_rows_per_image,
        in_row_bytes=in_row_bytes,
        rows_per_image_out=rows_per_image,
    )
    _PLAN_CACHE[key] = plan
    return plan


def _cb_on_shard(cb_index, tensor, core_ranges, page_bytes: int, tile_desc):
    """A CB placed ON `tensor`'s resident shard — the native sharded path.

    `cb_descriptor_from_sharded_tensor` inherits the tensor's own page size
    (a stick for ROW_MAJOR, a tile for TILE). This op accounts in whole tiles on
    both sides, so the page size is re-stated as the tile-sized unit; the bytes
    are untouched and sit in L1 exactly as a tile-paged buffer, which is what
    keeps the whole leg zero-copy while the helpers still see clean tile pages.
    """
    cb = ttnn.cb_descriptor_from_sharded_tensor(cb_index, tensor, 0, 0, core_ranges)
    formats = cb.format_descriptors
    formats[0].page_size = page_bytes
    formats[0].tile = tile_desc
    cb.format_descriptors = formats
    assert (
        cb.total_size % page_bytes == 0
    ), f"tilize: shard bank size {cb.total_size} is not a whole number of {page_bytes} B tile pages"
    return cb


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    low_l1: bool = False,
    pad_value=None,
    compute_kernel_config=None,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()  # never a hardcoded core count
    plan = derive_plan(input_tensor, output_tensor, low_l1=low_l1, grid=grid, pad_value=pad_value)

    tile_desc = ttnn.TileDescriptor(plan.tile_h, TILE_WIDTH)

    # ========== Circular buffers ==========
    # Emitted PER COLUMN GROUP. A plan has one group unless the column cut left
    # a ragged tail, in which case the tail's cores get their own CB sizes at
    # their own `block_width_tiles` — which is what keeps the per-core CB
    # push/pop quantum a single constant and the FIFO wrap legal (see the module
    # docstring). On the single-group plan this loop emits exactly the same
    # descriptors as before.
    #
    # cb_input_rows: live set is ONE tile-row of the block (the tilize helper
    # waits/pops exactly block_width_tiles pages per iteration), so
    # block_row_extent does not enter the size. Capacity = depth * live set.
    #
    # ZERO-COPY when the input is consumed natively: the CB is PLACED ON the
    # shard buffer, so the resident bytes are the CB's contents and the reader
    # issues no NoC read at all. The page size is overridden to the tile-sized
    # accounting unit the tilize helper counts in — the shard's own paging is
    # one ROW_MAJOR stick, and `block_width_tiles` sticks-worth of contiguous
    # bytes IS one tile-row page group, because the shard's row width is exactly
    # the block's row width (`block_width_tiles == shard_cols_tiles`).
    #
    # cb_output_tiles: live set is one write batch; capacity is
    # OUTPUT_DEPTH_BATCHES of them. Its data_format is where the
    # value-preserving `dtype=` cast happens, at pack time.
    #
    # ZERO-COPY when the output is produced natively: the packer writes the
    # output shard in place and there is no writer kernel at all. The tile order
    # matches because compute emits tile-rows left to right across the whole
    # shard width, which is the order a TILE shard stores its pages in.
    #
    # cb_pad_row: reader-local scratch holding ONE block row pre-filled with the
    # fill value. Allocated only on the padded path. Its point is that a whole
    # pad ROW then costs one DM-engine transfer from L1 to L1 instead of a RISC
    # store loop over `block_row_bytes` — the reader seeds it once per kernel
    # (32 elements by hand, then doubling local reads) and reads from it for
    # every fully-padded row of every block. It has no producer/consumer pair:
    # nothing is ever pushed or popped, so its `total_size` is exactly its one
    # page and the reader only ever takes `get_write_ptr` of it.
    #
    # cb_input_rows_split: the WRITER's half of the block on the split-reader
    # path. Sized to the whole half so the writer never waits on compute before
    # its store — see tilize_writer.cpp's SPLIT READER note for why a shallower
    # buffer would close a deadlock cycle.
    def _scratch_cb(index, dtype, page_bytes, pages, cores, tile=None):
        return ttnn.CBDescriptor(
            total_size=pages * page_bytes,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index,
                    data_format=dtype,
                    page_size=page_bytes,
                    **({"tile": tile} if tile is not None else {}),
                )
            ],
        )

    cbs = []
    for group in plan.groups:
        if plan.input_native:
            cbs.append(_cb_on_shard(CB_INPUT_ROWS, input_tensor, group.cores, plan.in_page_bytes, tile_desc))
        else:
            cbs.append(
                _scratch_cb(
                    CB_INPUT_ROWS, input_tensor.dtype, plan.in_page_bytes, group.input_cb_pages, group.cores, tile_desc
                )
            )
        if plan.output_native:
            cbs.append(_cb_on_shard(CB_OUTPUT_TILES, output_tensor, group.cores, plan.out_page_bytes, tile_desc))
        else:
            cbs.append(
                _scratch_cb(
                    CB_OUTPUT_TILES,
                    output_tensor.dtype,
                    plan.out_page_bytes,
                    group.output_cb_pages,
                    group.cores,
                    tile_desc,
                )
            )
        if group.pad_active:
            cbs.append(_scratch_cb(CB_PAD_ROW, input_tensor.dtype, group.pad_row_bytes, 1, group.cores))
        if group.split_reader:
            cbs.append(
                _scratch_cb(
                    CB_INPUT_ROWS_SPLIT,
                    input_tensor.dtype,
                    plan.in_page_bytes,
                    group.split_cb_pages,
                    group.cores,
                    tile_desc,
                )
            )

    # ========== Compute config ==========
    # The user-facing `compute_kernel_config` maps 1:1 onto the descriptor for
    # the three knobs that are the caller's to choose. `fp32_dest_acc_en` is the
    # exception and is ORed, never overridden: at the dtype pairs where
    # `requires_fp32_dest_acc` is true, turning DEST back down to 16 bits does
    # not trade precision for speed — it makes a VALUE-PRESERVING op return the
    # wrong bytes (and, on the lossless-fp32 path, fails the helper's own
    # static_assert). Everything else is honoured verbatim.
    #
    # Passing nothing reproduces the pre-refinement descriptor exactly:
    # HiFi4 / math_approx_mode=False / dst_full_sync_en=False, with
    # fp32_dest_acc_en from the plan. `MathFidelity.Invalid` is what a bare
    # `ttnn.WormholeComputeKernelConfig()` carries, so it maps to that default
    # rather than being pushed at the hardware.
    user_fidelity = getattr(compute_kernel_config, "math_fidelity", None)
    if user_fidelity is None or user_fidelity == ttnn.MathFidelity.Invalid:
        user_fidelity = ttnn.MathFidelity.HiFi4
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=user_fidelity,
        math_approx_mode=bool(getattr(compute_kernel_config, "math_approx_mode", False)),
        fp32_dest_acc_en=bool(plan.fp32_dest_acc_en or getattr(compute_kernel_config, "fp32_dest_acc_en", False)),
        dst_full_sync_en=bool(getattr(compute_kernel_config, "dst_full_sync_en", False)),
    )
    # `UnpackToDestFp32` on cb_input_rows is the third leg of the lossless fp32
    # relay (see `is_lossless_fp32_relay`): it takes the input datum straight to
    # DEST in full fp32 instead of through SrcA's tf32. It is set ONLY there —
    # on the fast-tilize path the helper static_asserts that this CB is
    # `Default`, and combining the two silently corrupts the output.
    if plan.lossless_fp32:
        unpack_to_dest_mode = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
        unpack_to_dest_mode[CB_INPUT_ROWS] = ttnn.UnpackToDestMode.UnpackToDestFp32
        # The SPLIT input CB is the same relay's second input — the compute
        # kernel tilizes the block's trailing sub-block straight out of it — so
        # it carries the identical mode. Setting only CB_INPUT_ROWS trips the
        # helper's own `Fp32Mode::Lossless` static_assert on the split call
        # (tilize_helpers.inl:122), which is the guard doing its job.
        unpack_to_dest_mode[CB_INPUT_ROWS_SPLIT] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = unpack_to_dest_mode

    # ========== Kernels ==========
    # The CT "plan" block is identical in all three kernels: each derives its
    # own view of a block from the same numbers, which is why there is no
    # cross-kernel handshake and no coordinator core.
    #
    # Emitted PER COLUMN GROUP, exactly like the CBs: the tail's cores run the
    # same three kernel sources with their own `block_width_tiles`,
    # `num_w_chunks` and `col_tile_offset`. On a single-group plan this is one
    # descriptor per kernel, unchanged.
    in_accessor_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    out_accessor_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()

    kernels = []
    for group in plan.groups:
        reader_ct_args = [
            CB_INPUT_ROWS,
            group.block_width_tiles,
            plan.tile_h,
            plan.tensor_row_blocks,
            plan.num_row_groups,
            group.num_w_chunks,
            group.block_row_bytes,
            int(plan.input_native),
            plan.input_pages_per_row,
            plan.in_page_width_bytes,
            # --- padding (inert, and the branch compiles out, when pad_active is 0)
            int(group.pad_active),
            CB_PAD_ROW,
            plan.elem_size,
            plan.pad_word,
            plan.in_num_images,
            plan.in_rows_per_image,
            plan.in_row_bytes,
            plan.rows_per_image_out,
            # --- retile (inert, and the branch compiles out, when is_retile is 0)
            int(group.is_retile),
            plan.in_tile_h,
            plan.in_tile_rows_per_image,
            CB_OUTPUT_TILES,
            plan.tensor_col_tiles,
            plan.out_page_bytes,
            # --- ragged column tail / split reader (both inert at 0)
            group.col_tile_offset,
            group.split_reader,
            SPLIT_READER_WRITER_SHARE_PCT,
        ]
        reader_ct_args.extend(in_accessor_args)

        writer_ct_args = [
            CB_OUTPUT_TILES,
            group.block_width_tiles,
            plan.tensor_row_blocks,
            plan.tensor_col_tiles,
            plan.num_row_groups,
            group.num_w_chunks,
            group.write_rows_per_barrier,
            plan.out_page_bytes,
            group.col_tile_offset,
            group.split_reader,
            CB_INPUT_ROWS_SPLIT,
            plan.tile_h,
            group.block_row_bytes,
            SPLIT_READER_WRITER_SHARE_PCT,
        ]
        writer_ct_args.extend(out_accessor_args)
        # The writer reads the block's trailing tile-rows on the split path, so
        # it carries the INPUT's accessor too — declared unconditionally and
        # chained off the output accessor's offset so the indices never move.
        writer_ct_args.extend(in_accessor_args)

        compute_ct_args = [
            CB_INPUT_ROWS,
            CB_OUTPUT_TILES,
            group.block_width_tiles,
            plan.tensor_row_blocks,
            plan.num_row_groups,
            group.num_w_chunks,
            int(COMPUTE_SKIP_FORMAT_RECONFIG),
            # The init/uninit amortization is only emitted where a core can
            # actually own more than one block; at one block per core its three
            # extra instantiations are dead code that still costs binary-dispatch
            # time. It is also OFF whenever the reader is split, because the two
            # sub-block calls program the tilize LLK from different input CB
            # indices and the amortized modes would carry one CB's init into the
            # other's call.
            int(
                COMPUTE_AMORTIZE_INIT
                and not group.split_reader
                and max((a[2] for a in group.assignment), default=0) > 1
            ),
            # --- numerical policy (Refinement 5), both derived, both inert (and
            # compiled out) at every dtype pair that does not need them.
            int(plan.lossless_fp32),
            int(plan.repair_srcb_alu_format),
            # --- split reader (inert, and the second tilize call compiles out, at 0)
            group.split_reader,
            CB_INPUT_ROWS_SPLIT,
            SPLIT_READER_WRITER_SHARE_PCT,
        ]

        reader_rt_args = ttnn.RuntimeArgs()
        writer_rt_args = ttnn.RuntimeArgs()
        compute_rt_args = ttnn.RuntimeArgs()
        for core, start_block_id, num_blocks, block_stride in group.assignment:
            reader_rt_args[core.x][core.y] = [in_addr, start_block_id, num_blocks, block_stride]
            writer_rt_args[core.x][core.y] = [out_addr, start_block_id, num_blocks, block_stride, in_addr]
            compute_rt_args[core.x][core.y] = [start_block_id, num_blocks, block_stride]

        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
                core_ranges=group.cores,
                compile_time_args=reader_ct_args,
                runtime_args=reader_rt_args,
                config=ttnn.ReaderConfigDescriptor(),  # reads on NoC0
            )
        )
        # No writer at all on the native output path: the packer has already
        # placed every tile in the output shard, so there is nothing left to
        # move. A writer that re-wrote a core's own shard over the NoC would be
        # the interleaved path wearing a sharded hat.
        if not plan.output_native:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
                    core_ranges=group.cores,
                    compile_time_args=writer_ct_args,
                    runtime_args=writer_rt_args,
                    config=ttnn.WriterConfigDescriptor(),  # writes on NoC1
                )
            )
        # No compute kernel at all on the RETILE path. A re-tile is a pure byte
        # re-lay between two tiled layouts, so `retile_copy_unit`'s runs go from
        # the source tile's faces straight into the destination tile's faces over
        # the NoC — there is no row-major intermediate for a tilize LLK to
        # consume and nothing for the FPU to do. cb_output_tiles then has the
        # READER as its single producer and the writer as its single consumer,
        # which is the same one-producer/one-consumer contract as every other CB.
        if not plan.is_retile:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
                    core_ranges=group.cores,
                    compile_time_args=compute_ct_args,
                    runtime_args=compute_rt_args,
                    config=compute_config,
                )
            )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
