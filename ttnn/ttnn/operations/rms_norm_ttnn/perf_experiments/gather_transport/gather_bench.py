# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ISOLATED BAKE-OFF: the member -> root GATHER TRANSPORT of rms_norm_ttnn's
# cross-core width combine.  Idea slug `gather_transport`.
#
# WHAT IS ISOLATED (per /perf-lab's concept-isolation table).  Everything that is
# NOT the transport is held identical across variants:
#   * the payload SOURCE is a resident L1 shard (one fp32 tile per core), so no
#     reader, no DRAM, no gamma, and therefore NO INTER-CORE SKEW -- every core
#     enters the ship at the same instant.  This is the de-skewed measurement the
#     task asks for in (d); the op's own members are skewed 541 ns by a per-core
#     DRAM read that a sibling agent is removing.
#   * the ROOT's fold is the op's own D22 pairwise-add DEST walk over the same
#     GATHER_SLOTS pages, byte-identical across variants.
#   * the precision contract is the op's, UNCHANGED and never a variant knob:
#     HiFi2, fp32_dest_acc_en=False, math_approx_mode=False, fp32 gather pages.
# The ONLY thing a variant changes is HOW the 32 fp32 partial stats travel from a
# member's L1 into the root's landing page.
#
# THE VARIANTS (all carry the SAME 32 fp32 values):
#   col_2x1024        BASELINE == the shipped op (GATHER_FACES=2): a REDUCE_ROW
#                     partial is a COLUMN vector, so its 32 values live in faces 0
#                     and 2 -> two 1024 B writes, one barrier, one atomic inc.
#   col_1x3072        control: one 3072 B write (faces 0..2 prefix).  The op's
#                     Refinement-1 lever-3 sweep already measured this a LOSS on the
#                     real op; re-running it here is the bench's CALIBRATION against
#                     a known number, not a new question.
#   col_1x4096        control: whole tile (GATHER_FACES=4), also a known loss.
#   row_2x64          THE IDEA: if the partial were a ROW vector its 32 values would
#                     be row 0 of faces 0 and 1 -> two 64 B writes.  Same transaction
#                     count, 16x FEWER BYTES.  (Transport half only; the permute that
#                     would produce a row-shaped partial is priced separately below.)
#   *_flush           the member replaces `noc_async_write_barrier()` (round-trip ACK)
#                     with `noc_async_writes_flushed()` (departed) before the atomic
#                     inc.  Legal by NoC ordering: both the payload write and
#                     Semaphore::up default to NOC_UNICAST_WRITE_VC (VC 1) and travel
#                     the same source->dest path, so a write that has DEPARTED cannot
#                     be overtaken by an atomic issued after it.
#   *_dualnoc         half the members ship from BRISC/NOC_0 and half from
#                     NCRISC/NOC_1, so the root's L1 is filled through BOTH NIUs.
#                     Tests whether the root's ingress is ONE NoC link's bandwidth.
#   *_altnoc          the SAME split, but issued from ONE kernel: odd slots build their
#                     destination with `get_noc_addr(x, y, addr, /*noc=*/1)` and write /
#                     flush / atomic through `Noc(1)`.  Same two NIUs at the root, but the
#                     op-side change is four lines in the writer instead of a second
#                     shipping kernel -- so it is worth knowing whether it costs the same.
#
# CORRECTNESS is the only pass/fail: every variant's folded output is compared
# BIT-EXACTLY against the baseline variant's (same values transported => same sum)
# and the baseline itself against torch.

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics
import zlib

import ttnn

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

CB_X = 0
CB_GATHER = 1
CB_OUT = 2
CB_SHIP = 3
CB_FOLD = 4
SEM_GATHER = 0
TILE = 32

# variant -> (ship_shape, use_flush, noc_split, xpose_send, xpose_back)
#   ship_shape 0 = 2 x 1024 B at face 0 / face 2   (the shipped op, GATHER_FACES=2)
#              1 = 1 x 3072 B (faces 0..2 prefix, GATHER_FACES=3)
#              2 = 1 x 4096 B (whole tile, GATHER_FACES=4)
#              3 = 2 x   64 B at row 0 of faces 0 / 1 (a ROW-shaped partial)
#              4 = NO PAYLOAD AT ALL -- semaphore only.  The transport FLOOR: what the
#                  round costs when the bytes are deleted and only the fan-in of atomic
#                  increments and the root's fold remain.  Not a candidate (it computes
#                  the wrong answer); it is the denominator for "how much is left to win".
#   xpose_send  0 = none.
#               1 = the MEMBER runs `transpose_tile` on its partial before shipping, which
#                   is what actually PRODUCES a row-shaped partial.  Send end of the
#                   permute price, as a SEPARATE compute stage (its own CB round trip).
#               2 = CONTROL: the same separate stage but `copy_tile` instead of
#                   `transpose_tile`.  (1) minus (2) is the MARGINAL cost of the transpose
#                   inside a DEST window that already exists and already packs -- which is
#                   what the op would actually pay, because pass A's reduce IS such a
#                   window.  The round trip itself is NOT a cost the op would pay.
#   xpose_back  0 = none.
#               1 = the ROOT runs `transpose_tile` on the folded stat as a SEPARATE stage,
#                   turning it back into a column vector so the multicast and pass B are
#                   UNCHANGED.  Costs its own CB round trip.
#               2 = the same map with NO round trip: raw `llk_math_transpose_dest` applied
#                   INSIDE the fold's existing DEST window, right before its pack.  This is
#                   what the op would really do (its fold has a DEST window and one pack).
#               With a send and a back permute on, the variant's input->output map is
#               IDENTICAL to the baseline's.
VARIANTS = {
    "col_2x1024": (0, 0, 0, 0, 0),  # BASELINE (== shipped op)
    "col_1x3072": (1, 0, 0, 0, 0),
    "col_1x4096": (2, 0, 0, 0, 0),
    "row_2x64": (3, 0, 0, 0, 0),
    "sem_only": (4, 0, 0, 0, 0),
    "col_2x1024_flush": (0, 1, 0, 0, 0),
    "row_2x64_flush": (3, 1, 0, 0, 0),
    "col_2x1024_dualnoc": (0, 0, 1, 0, 0),
    "row_2x64_dualnoc": (3, 0, 1, 0, 0),
    "col_2x1024_dualnoc_flush": (0, 1, 1, 0, 0),
    # *** MEASURED HANG -- DO NOT RE-RUN.  Issuing the odd slots' write / flush / atomic on
    # NOC_1 from the BRISC kernel (get_noc_addr(..., noc=1) + Noc(1)) hangs the dispatch at
    # GROUP_SIZE 28.  In DM_DEDICATED_NOC each RISC owns one NoC's counters and command
    # buffers, so the second NIU at the root can only be reached from the OTHER RISC.  The
    # dual-NoC gather is therefore a TWO-KERNEL shape (`*_dualnoc`), not a four-line writer
    # edit.  Kept so the negative result is on the record.
    "col_2x1024_altnoc": (0, 0, 2, 0, 0),
    "col_2x1024_altnoc_flush": (0, 1, 2, 0, 0),
    "row_2x64_altnoc_flush": (3, 1, 2, 0, 0),
    # ---- the permute priced, one end at a time and then both ----------------
    "row_xpose_send": (3, 1, 0, 1, 0),
    "row_xpose_full": (3, 1, 0, 1, 1),
    # controls that isolate the permute cost from the transport cost: same permutes,
    # the BASELINE's 2 x 1024 B ship.
    "col_xpose_full": (0, 1, 0, 1, 1),
    # the MARGINAL-cost pair: identical structure, transpose vs plain copy.
    "xs_copy_col_flush": (0, 1, 0, 2, 0),
    "xs_xpose_col_flush": (0, 1, 0, 1, 0),
    # the back permute with NO round trip (raw llk_math_transpose_dest in the fold window)
    # *** MEASURED HANG -- DO NOT RE-RUN.  `llk_math_transpose_dest` inside the fold's own
    # DEST window stalls forever: its TTI_STALLWAIT waits on SRCA_VLD|SRCB_VLD and
    # `add_tiles` has just CLEARED both (SETRWC CLR_AB).  `llk_unpack_set_srcb_dummy_valid`
    # covers SrcB only -- there is no SrcA equivalent, so a real unpack would have to be
    # re-issued.  This is the capability that would make the ROOT's back-permute free; as
    # the API stands it costs a whole CB round trip (223 ns at GROUP_SIZE 28, see
    # `row_xpose_full` minus `row_xpose_send`).
    "row_flush_xb_dest": (3, 1, 0, 0, 2),
    "row_xdest_full": (3, 1, 0, 1, 2),
    # THE OP-INTEGRATED PROXY: the send permute is FREE in the op (pass A's reduce already
    # owns a DEST window and a pack, and the measured marginal cost of a transpose inside
    # such a window is 13 ns -- `xs_xpose_col_flush` minus `xs_copy_col_flush`), so the
    # honest op-integrated cost is the row ship PLUS only the ROOT's back permute.
    "row_flush_xb1": (3, 1, 0, 0, 1),
}
BASE_VARIANT = "col_2x1024"


# Which slice of the tile a variant actually transports.
#   "col" -> column 0 (rows 0..31)      "row" -> row 0 (cols 0..31)
def _slice_of(cfg):
    """Which slice of the OUTPUT tile holds the group sum, and which slice of each
    member's INPUT tile fed it.  A transpose on either end flips one of them."""
    ship_shape, _, _, xs, xb = cfg
    flip = {"row": "col", "col": "row"}
    shipped = "row" if ship_shape == 3 else "col"  # slice of the member's LOCAL tile
    src = flip[shipped] if xs == 1 else shipped  # ... expressed in the ORIGINAL x (xs==2 copies)
    dst = flip[shipped] if xb else shipped  # ... where it ends up in the output tile
    return src, dst


VARIANT_SLICE = {name: _slice_of(cfg) for name, cfg in VARIANTS.items()}


_SHIP_KERNEL = r"""
// Isolated gather-transport bench for rms_norm_ttnn (perf_experiments/gather_transport).
//
// RAW-API JUSTIFICATION (this is a bench, /perf-lab suspends the prefer-helpers rule):
// the ship uses the RAW `noc_async_write` / `noc_async_writes_flushed` free functions
// instead of a dataflow_kernel_lib page helper because the whole point of the
// measurement is the exact BYTE EXTENT and TRANSACTION COUNT of the transfer -- a
// page-shaped helper would fix both.  This mirrors what the shipped writer already
// does at `ship_partial` (rms_norm_ttnn_writer.cpp), which is the honest baseline.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t GATHER_SLOTS = get_compile_time_arg_val(1);
    constexpr uint32_t SHIP_SHAPE = get_compile_time_arg_val(2);
    constexpr uint32_t USE_FLUSH = get_compile_time_arg_val(3);
    constexpr uint32_t NOC_SPLIT = get_compile_time_arg_val(4);
    constexpr uint32_t ROLE = get_compile_time_arg_val(5);  // 0 = BRISC/NOC_0, 1 = NCRISC/NOC_1
    constexpr uint32_t XPOSE_SEND = get_compile_time_arg_val(6);
    constexpr uint32_t CB_X = 0;       // alias of this core's resident partial tile
    constexpr uint32_t CB_GATHER = 1;  // the root's landing ring (declared on every core)
    constexpr uint32_t CB_OUT = 2;     // alias of the output shard (root only)
    constexpr uint32_t CB_SHIP = 3;    // the permuted partial compute hands back
    constexpr uint32_t TILE_BYTES = 4096;
    constexpr uint32_t FACE_BYTES = TILE_BYTES / 4;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_slot = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);
    const uint32_t root_x = get_arg_val<uint32_t>(3);
    const uint32_t root_y = get_arg_val<uint32_t>(4);

    Noc noc;
    Semaphore<> gather_sem(0);

    // The landing base is computed LOCALLY: cb_gather is declared on every core of the
    // program with the same size, so its L1 address is identical everywhere -- exactly
    // the invariant the shipped writer relies on.
    const uint32_t gather_base = get_write_ptr(CB_GATHER);

    // The permuted-partial handoff.  ROLE 1 (NCRISC) publishes the resident shard into
    // CB_X so compute can unpack it -- the shipped op's `publish_native_shard` does
    // exactly this for its own zero-copy input CB.
    if constexpr (XPOSE_SEND != 0) {
        if constexpr (ROLE == 1) {
            cb_push_back(CB_X, 1);
        }
    }

    // Who ships from this RISC.  Single-NoC: only the BRISC/NOC_0 kernel (the shipped
    // op's combine NoC on a resident-x plan).  Dual-NoC: even slots on NOC_0, odd on
    // NOC_1, so the root's L1 is filled through both NIUs.
    // NOC_SPLIT: 0 = single NoC (BRISC/NOC_0 only, the shipped op's combine NoC on a
    // resident-x plan); 1 = TWO kernels, even slots on BRISC/NOC_0 and odd on NCRISC/NOC_1;
    // 2 = ONE kernel, odd slots issued on NOC_1 explicitly (same two NIUs at the root).
    const bool i_ship = (NOC_SPLIT == 1) ? ((my_slot & 1u) == ROLE) : (ROLE == 0);
    const bool alt_noc = (NOC_SPLIT == 2) && ((my_slot & 1u) == 1u) && (is_root == 0);
    const Noc noc_alt(1);
    const uint8_t ship_noc = alt_noc ? noc_alt.get_noc_id() : noc.get_noc_id();

    uint32_t src_addr = x_addr;
    if constexpr (XPOSE_SEND != 0) {
        if (i_ship) {
            cb_wait_front(CB_SHIP, 1);
            src_addr = get_read_ptr(CB_SHIP);
        }
    }

    if (i_ship) {
        const uint64_t dst =
            (is_root != 0) ? get_noc_addr(gather_base + my_slot * TILE_BYTES)
                           : get_noc_addr(root_x, root_y, gather_base + my_slot * TILE_BYTES, ship_noc);
        if constexpr (SHIP_SHAPE == 0) {
            // Faces 0 and 2 -- the only pair that can carry a REDUCE_ROW column vector.
            noc_async_write(src_addr, dst, FACE_BYTES, ship_noc);
            noc_async_write(src_addr + 2 * FACE_BYTES, dst + 2 * FACE_BYTES, FACE_BYTES, ship_noc);
        } else if constexpr (SHIP_SHAPE == 1) {
            noc_async_write(src_addr, dst, 3 * FACE_BYTES, ship_noc);
        } else if constexpr (SHIP_SHAPE == 2) {
            noc_async_write(src_addr, dst, TILE_BYTES, ship_noc);
        } else if constexpr (SHIP_SHAPE == 3) {
            // Row 0 of faces 0 and 1 -- where a ROW-shaped partial's 32 fp32 live.
            noc_async_write(src_addr, dst, 64, ship_noc);
            noc_async_write(src_addr + FACE_BYTES, dst + FACE_BYTES, 64, ship_noc);
        } else {
            (void)dst;  // SHIP_SHAPE 4: the semaphore-only FLOOR, no payload at all.
        }
        if constexpr (USE_FLUSH != 0) {
            if (alt_noc) {
                noc_alt.async_writes_flushed();
            } else {
                noc.async_writes_flushed();
            }
        } else {
            noc_async_write_barrier(ship_noc);
        }
        // NO self-signal on the root: Semaphore::up is a non-atomic local RMW and would
        // race the members' remote atomics (the shipped writer says the same).
        if (is_root == 0) {
            gather_sem.up(alt_noc ? noc_alt : noc, root_x, root_y, 1);
        }
    }

    if ((is_root != 0) && (ROLE == 0)) {
        // A pad slot no sender ever writes is folded WHOLE, so it must be an exact +0.0.
        // Zeroing only the NEVER-WRITTEN slot is race-free by construction (the shipped
        // writer's `zero_pad_slots` makes the same argument).
        if constexpr (GATHER_SLOTS > GROUP_SIZE) {
            DataflowBuffer dfb(CB_GATHER);
            for (uint32_t p = GROUP_SIZE; p < GATHER_SLOTS; ++p) {
                noc.async_write_zeros(dfb, TILE_BYTES, {.offset_bytes = p * TILE_BYTES});
            }
            noc.write_zeros_l1_barrier();
        }
        gather_sem.wait_min(GROUP_SIZE - 1);
        cb_push_back(CB_GATHER, GATHER_SLOTS);
        // completion barrier: the folded tile IS the output shard (zero-copy CB).
        cb_wait_front(CB_OUT, 1);
    }
}
"""


_FOLD_KERNEL = r"""
// The bench's compute half.  Two jobs, both compile-time gated:
//
//  1. XPOSE_SEND (every core) -- `transpose_tile` turns this core's COLUMN-shaped
//     partial into a ROW-shaped one, which is what actually MAKES the 2 x 64 B ship
//     possible.  This is the SEND end of the permute price, paid in the bench so the
//     transport win is never quoted without it.
//     RAW-API NOTE: `transpose_tile` (api/compute/transpose.h) is the primitive, not a
//     kernel_lib helper -- the op's own D27 note records that a COLUMN permutation has
//     no kernel_lib expression and that `transpose_wh` is "a different map".  For THIS
//     idea the whole-tile transpose IS the map we want (column vector <-> row vector),
//     which is exactly the case D27 did not need.
//
//  2. the ROOT's fold -- lifted UNCHANGED in kind from rms_norm_ttnn_compute.cpp's
//     `combine_fold`: one pairwise add_tiles walk over the gather ring inside a single
//     DEST window, then one pack.  Identical for every variant: it is the control.
//     No finalize (rsqrt) here -- it is a constant that would only dilute the delta.
//     XPOSE_BACK then transposes the folded stat back to a COLUMN vector, which is what
//     leaves the op's multicast and pass B untouched.  RECEIVE end of the permute price.
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t SLOTS = get_compile_time_arg_val(0);
    constexpr uint32_t XPOSE_SEND = get_compile_time_arg_val(1);
    constexpr uint32_t XPOSE_BACK = get_compile_time_arg_val(2);
    constexpr uint32_t HALF = SLOTS / 2;
    constexpr uint32_t CB_X = 0;
    constexpr uint32_t CB_GATHER = 1;
    constexpr uint32_t CB_OUT = 2;
    constexpr uint32_t CB_SHIP = 3;
    constexpr uint32_t CB_FOLD = 4;

    const uint32_t is_root = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(CB_GATHER, CB_GATHER, CB_OUT);

    if constexpr (XPOSE_SEND != 0) {
        cb_wait_front(CB_X, 1);
        cb_reserve_back(CB_SHIP, 1);
        reconfig_data_format_srca(CB_X);
        pack_reconfig_data_format(CB_SHIP);
        if constexpr (XPOSE_SEND == 1) {
            transpose_init(CB_X);
        } else {
            copy_tile_to_dst_init_short(CB_X);
        }
        tile_regs_acquire();
        if constexpr (XPOSE_SEND == 1) {
            transpose_tile(CB_X, 0, 0);
        } else {
            copy_tile(CB_X, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SHIP);
        tile_regs_release();
        cb_push_back(CB_SHIP, 1);
    }

    if (is_root == 0) {
        return;
    }

    constexpr uint32_t FOLD_OUT = (XPOSE_BACK == 1) ? CB_FOLD : CB_OUT;
    cb_wait_front(CB_GATHER, SLOTS);
    reconfig_data_format(CB_GATHER, CB_GATHER);
    pack_reconfig_data_format(FOLD_OUT);
    add_tiles_init(CB_GATHER, CB_GATHER, /*acc_to_dest=*/true);
    if constexpr (XPOSE_BACK == 2) {
        // RAW LLK, and this is the whole point of the variant.  HELPER BYPASSED:
        // `transpose_tile` (api/compute/transpose.h).  CAPABILITY, not ergonomics -- the
        // helper is unpack-to-DEST + in-DEST transpose + (caller's) pack, so using it
        // forces the folded stat OUT to L1 and back IN through a second CB before it can
        // be transposed.  The value is ALREADY in DEST at the end of the fold; all that is
        // missing from the API is a way to run just the in-DEST half.  Measured pair at
        // GROUP_SIZE 28: helper as its own stage (`row_xpose_full`) vs this
        // (`row_xdest_full`) -- see the bench's RESULT table.
        MATH((llk_math_transpose_dest_init<true, false>()));
    }
    tile_regs_acquire();
    for (uint32_t p = 0; p < HALF; ++p) {
        add_tiles(CB_GATHER, CB_GATHER, p, HALF + p, 0);
    }
    if constexpr (XPOSE_BACK == 2) {
        UNPACK((llk_unpack_set_srcb_dummy_valid()));
        MATH((llk_math_transpose_dest<true, false>(0)));
    }
    tile_regs_commit();
    cb_reserve_back(FOLD_OUT, 1);
    tile_regs_wait();
    pack_tile(0, FOLD_OUT);
    tile_regs_release();
    cb_push_back(FOLD_OUT, 1);
    cb_pop_front(CB_GATHER, SLOTS);

    if constexpr (XPOSE_BACK == 1) {
        cb_wait_front(CB_FOLD, 1);
        cb_reserve_back(CB_OUT, 1);
        reconfig_data_format_srca(CB_FOLD);
        pack_reconfig_data_format(CB_OUT);
        transpose_init(CB_FOLD);
        tile_regs_acquire();
        transpose_tile(CB_FOLD, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_FOLD, 1);
    }
}
"""


def _core_list(grid):
    """Row-major logical cores of a (w, h) grid -- slot i == core i, root == slot 0."""
    w, h = grid
    return [(x, y) for y in range(h) for x in range(w)]


def _core_ranges(grid):
    w, h = grid
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, h - 1))])


def _shard(device, shape, core_ranges, shard_shape):
    return ttnn.create_sharded_memory_config(
        shape=tuple(shard_shape),
        core_grid=core_ranges,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def build(device, grid, variant):
    """One fresh program + tensors for (grid, variant).  Returns (run, expected, live)."""
    import torch  # function-local: ttnn/ttnn may not import torch globally (pre-commit)

    cores = _core_list(grid)
    g = len(cores)
    slots = g if g % 2 == 0 else g + 1
    ship_shape, use_flush, noc_split, xpose_send, xpose_back = VARIANTS[variant]
    assert not (xpose_send and noc_split == 1), "the two-kernel split is not wired to the permute handoff"

    all_ranges = _core_ranges(grid)
    root_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    # A PER-VARIANT SEED, and it is load-bearing.  The gather ring is a CB at a fixed L1
    # address that nothing clears between programs, so a variant that transported NOTHING
    # would otherwise "pass" on the PREVIOUS variant's bytes.  Different data per variant
    # makes a stale-L1 read a guaranteed correctness failure instead of a silent pass.
    torch.manual_seed(zlib.crc32(variant.encode()) % (2**31))
    tx = torch.randn(g * TILE, TILE, dtype=torch.float32)
    x = ttnn.from_torch(
        tx,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_shard(device, None, all_ranges, (TILE, TILE)),
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE]),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        device,
        _shard(device, None, root_ranges, (TILE, TILE)),
    )

    root_v = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))

    ship_rt = [ttnn.RuntimeArgs(), ttnn.RuntimeArgs()]
    for slot, (cx, cy) in enumerate(cores):
        args = [
            x.buffer_address(),
            slot,
            1 if slot == 0 else 0,
            root_v.x,
            root_v.y,
        ]
        for r in (0, 1):
            ship_rt[r][cx][cy] = list(args)
    fold_rt = ttnn.RuntimeArgs()
    # runtime args ONLY for the cores the compute kernel is actually placed on
    for slot, (cx, cy) in enumerate(cores if xpose_send else cores[:1]):
        fold_rt[cx][cy] = [1 if slot == 0 else 0]

    tile_bytes = ttnn.tile_size(ttnn.float32)
    cbs = [
        ttnn.CBDescriptor(
            total_size=slots * tile_bytes,
            core_ranges=all_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_GATHER, data_format=ttnn.float32, page_size=tile_bytes)
            ],
        ),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
    ]
    if xpose_send:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_X, x))
        cbs.append(
            ttnn.CBDescriptor(
                total_size=tile_bytes,
                core_ranges=all_ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_SHIP, data_format=ttnn.float32, page_size=tile_bytes)
                ],
            )
        )
    if xpose_back == 1:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=tile_bytes,
                core_ranges=root_ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_FOLD, data_format=ttnn.float32, page_size=tile_bytes)
                ],
            )
        )

    compute_cfg = ttnn.ComputeConfigDescriptor()
    compute_cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    compute_cfg.fp32_dest_acc_en = False
    compute_cfg.math_approx_mode = False

    kernels = []
    for role, cfg in (
        (0, ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
        (1, ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
    ):
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_SHIP_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=all_ranges,
                compile_time_args=[g, slots, ship_shape, use_flush, noc_split, role, xpose_send],
                runtime_args=ship_rt[role],
                config=cfg,
            )
        )
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=_FOLD_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=all_ranges if xpose_send else root_ranges,
            compile_time_args=[slots, xpose_send, xpose_back],
            runtime_args=fold_rt,
            config=compute_cfg,
        )
    )

    descriptor = ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=[ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_ranges, initial_value=0)],
        cbs=cbs,
    )

    per_core = tx.reshape(g, TILE, TILE)
    src_slice, _ = VARIANT_SLICE[variant]
    if src_slice == "col":
        expected = per_core[:, :, 0].sum(dim=0)  # column 0 of every partial
    else:
        expected = per_core[:, 0, :].sum(dim=0)  # row 0 of every partial

    def run():
        return ttnn.generic_op([x, out], descriptor)

    return run, expected, [x, out]


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def measure(device, grid, variant, trials=5):
    import torch  # function-local: ttnn/ttnn may not import torch globally (pre-commit)

    run, expected, live = build(device, grid, variant)
    out = run()
    got = ttnn.to_torch(out).float()
    _, dst_slice = VARIANT_SLICE[variant]
    landed = got[:, 0] if dst_slice == "col" else got[0, :]
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = []
    for _ in range(trials):
        run()
        ttnn.synchronize_device(device)
        v = _read_kernel_ns(device)
        if v is not None:
            samples.append(v)
    ns = statistics.median(samples) if samples else float("nan")
    for t in live:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return ns, landed.double(), expected.double()


def relerr(a, b):
    return float((a - b).abs().max() / (b.abs().max() + 1e-30))


def pcc(a, b):
    a = a.flatten() - a.flatten().mean()
    b = b.flatten() - b.flatten().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def sweep(grids, variants, trials=5, device=None):
    own = device is None
    if own:
        device = ttnn.open_device(device_id=0)
    res = {}
    try:
        for grid in grids:
            g = grid[0] * grid[1]
            for v in variants:
                ns, landed, expected = measure(device, grid, v, trials=trials)
                res[(g, v)] = (ns, relerr(landed, expected), pcc(landed, expected))
                print(f"RESULT RAW G={g:3d} {v:26s} {ns:9.0f} ns  relerr {res[(g,v)][1]:.3e}  pcc {res[(g,v)][2]:.7f}")
        print("RESULT " + f"{'G':>4s}" + "".join(f"{v:>27s}" for v in variants))
        for grid in grids:
            g = grid[0] * grid[1]
            print("RESULT " + f"{g:4d}" + "".join(f"{res[(g,v)][0]:27.0f}" for v in variants))
        print(f"RESULT --- speedup vs {BASE_VARIANT} (>1 = faster) ---")
        for grid in grids:
            g = grid[0] * grid[1]
            b = res[(g, BASE_VARIANT)][0]
            print("RESULT " + f"{g:4d}" + "".join(f"{b / res[(g,v)][0]:27.3f}" for v in variants))
    finally:
        if own:
            ttnn.close_device(device)
    return res
