// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer for rms_norm (BRISC, NoC1).
//
// Mirror image of the reader — same (row-block, width-chunk) loop nest, same
// WT_CHUNK transaction granularity, so both NoC halves are batched the same
// way (a reader-only batching lever just moves the bottleneck across the CB):
//   * TILE build      : cb_output_tiles  -> whole output tiles
//   * ROW_MAJOR build  : cb_output_sticks -> output sticks, W*elem bytes each
//   * NATIVE_OUT      : nothing to move at all — compute packed straight into
//                       the output shard's own L1 through a zero-copy CB; the
//                       writer only takes the completion barrier.
//
// The ROW_MAJOR path uses dataflow_kernel_lib::write_sticks_after_untilize,
// which is exactly the consumer contract of
// compute_kernel_lib::untilize<WT_CHUNK>(rows): it waits WT_CHUNK tile-sized
// pages per tile-row, writes only the VALID sticks (so trailing rows of a short
// final tile-row are never written) and only `row_bytes` of each (so the W tile
// padding is never written).
//
// Pass B runs once per row-block regardless of regime, so unlike the reader
// there is no pass loop here.
//
// ---------------------------------------------------------------------------
// COMBINE — the cross-core width combine (op_design.md section 3.4, Lamp L1/L4)
// ---------------------------------------------------------------------------
// When the cores of a group each own a width SLICE of the same rows, each core's
// sum(x^2) is a PARTIAL and must be combined.  This kernel owns the whole
// topology, per row-block:
//
//   1  every core   write its BLOCK_ROWS raw partial tiles (cb_sum_handoff) into
//                   its own slot of the ROOT's cb_partials_gathered, then
//                   remote-inc the root's arrival semaphore.
//   2  root         once arrivals reach (blk+1) * GROUP_SIZE, publish the
//                   gathered block so compute can sum + finalize it.
//   3  root         SenderPipe::send() the finalized stat tiles to the group's
//                   cb_row_final (loopback multicast: src != dst, so the root
//                   gets its own copy too).
//      non-root     ReceiverPipe::receive() into cb_row_final.
//
// It lives in the WRITER, not the reader, for two reasons: NoC1 is idle through
// pass A (so the combine handshake overlaps the reader's NoC0 x/gamma traffic),
// and cb_sum_handoff / cb_row_final then have exactly one dataflow kernel
// touching them — cb_row_stat stays compute-private, which is the CB-ownership
// rule the design calls out for this exact handoff.
//
// The gather landing address is `get_write_ptr(cb_partials_gathered)` computed
// LOCALLY on the sender: that CB is declared on every core of the program, so
// its L1 address is identical everywhere, and its ring holds exactly
// GROUP_SIZE * BLOCK_ROWS pages so a whole-block push returns the pointer to the
// base each round.  The host therefore never has to know a CB address.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_output_tiles = 8;
constexpr uint32_t cb_output_sticks = 9;
constexpr uint32_t cb_sum_handoff = 10;
constexpr uint32_t cb_partials_gathered = 11;
constexpr uint32_t cb_stat_handoff = 12;
constexpr uint32_t cb_row_final = 13;
constexpr uint32_t TILE_DIM = 32;
}  // namespace

void kernel_main() {
    // ---- compile-time knobs (all from rms_norm_program_descriptor.py) -----
    constexpr uint32_t IS_TILE = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_W_CHUNKS = get_compile_time_arg_val(3);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(4);
    constexpr uint32_t ELEM_BYTES = get_compile_time_arg_val(5);
    constexpr uint32_t R_RM = get_compile_time_arg_val(6);
    constexpr uint32_t W_ELEMS = get_compile_time_arg_val(7);
    constexpr uint32_t NATIVE_OUT = get_compile_time_arg_val(8);
    constexpr uint32_t COMBINE = get_compile_time_arg_val(9);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(10);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(11);
    constexpr uint32_t OUT_SHARD_PAGES = get_compile_time_arg_val(12);
    constexpr auto mc = dataflow_kernel_lib::McastArgs</*CT=*/13, /*RT=*/6>();
    constexpr auto out_args = TensorAccessorArgs<mc.next_compile_time_args_offset()>();

    constexpr bool RM = (IS_TILE == 0);
    constexpr bool NATIVE = (NATIVE_OUT != 0);
    constexpr bool CROSS_CORE = (COMBINE != 0);
    static_assert(!CROSS_CORE || !RM, "rms_norm: the cross-core width combine is TILE-only");
    static_assert(!CROSS_CORE || NUM_W_CHUNKS == 1, "rms_norm: a width-split core takes its slice in one chunk");
    constexpr uint32_t CHUNK_ROW_BYTES = WT_CHUNK * TILE_DIM * ELEM_BYTES;
    constexpr uint32_t LAST_CHUNK_ROW_BYTES = W_ELEMS * ELEM_BYTES - (NUM_W_CHUNKS - 1) * CHUNK_ROW_BYTES;

    // ---- runtime work assignment -----------------------------------------
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);  // this core's first tile-row
    const uint32_t num_rows = get_arg_val<uint32_t>(2);   // tile-rows owned by this core
    const uint32_t w_start = get_arg_val<uint32_t>(3);    // first width tile this core owns
    const uint32_t is_root = get_arg_val<uint32_t>(4);    // group root (multicast sender)
    const uint32_t my_slot = get_arg_val<uint32_t>(5);    // index within the width group

    // An INACTIVE core (see the reader): it exists only so the stat multicast
    // lands in a cb_row_final this program owns.  No shard, no work.
    if (num_rows == 0) {
        return;
    }

    const auto out_acc = TensorAccessor(out_args, out_addr);
    const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);

    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

    // The combine's pipes and semaphore are built ONCE, above the loop (their
    // ctors are the documented handshake init) and only on a participating core.
    Noc noc;
    Semaphore<> gather_sem(CROSS_CORE ? GATHER_SEM_ID : 0);
    uint32_t arrivals = 0;

    if constexpr (CROSS_CORE) {
        const uint32_t stat_bytes = get_tile_size(cb_stat_handoff);
        if (is_root != 0) {
            auto sender = mc.sender(noc);
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                const uint32_t r0 = blk * BLOCK_ROWS;
                const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;

                // 1. the root's own partial goes into slot 0 of its own gather CB.
                cb_wait_front(cb_sum_handoff, rows);
                cb_reserve_back(cb_partials_gathered, GROUP_SIZE * rows);
                noc_async_write(
                    get_read_ptr(cb_sum_handoff),
                    get_noc_addr(get_write_ptr(cb_partials_gathered) + my_slot * rows * stat_bytes),
                    rows * stat_bytes);
                noc_async_write_barrier();
                gather_sem.up(1);
                cb_pop_front(cb_sum_handoff, rows);

                // 2. publish the gathered block once every member has landed.
                arrivals += GROUP_SIZE;
                gather_sem.wait_min(arrivals);
                cb_push_back(cb_partials_gathered, GROUP_SIZE * rows);

                // 3. multicast the finalized stat back to the whole group.
                //
                // The root places its OWN copy first, then broadcasts in place
                // (src == dst => EXCLUDE-source).  Doing it this way makes the two
                // host emitters behave identically: Mcast1D's per-row sender rect
                // EXCLUDES the sender (mcast_host.hpp sender_rect_), while Mcast2D's
                // rect contains it -- an in-place send takes the same EXCLUDE path
                // in both, so the root is never served twice and never skipped.
                cb_wait_front(cb_stat_handoff, rows);
                cb_reserve_back(cb_row_final, rows);
                const uint32_t stat_dst = get_write_ptr(cb_row_final);
                noc_async_write(get_read_ptr(cb_stat_handoff), get_noc_addr(stat_dst), rows * stat_bytes);
                noc_async_write_barrier();
                if constexpr (mc.active) {
                    sender.send(stat_dst, stat_dst, rows * stat_bytes);
                }
                cb_push_back(cb_row_final, rows);
                cb_pop_front(cb_stat_handoff, rows);
            }
        } else {
            auto receiver = mc.receiver(noc);
            const uint32_t root_x = mc.sender_x();
            const uint32_t root_y = mc.sender_y();
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                const uint32_t r0 = blk * BLOCK_ROWS;
                const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;

                // 1. ship this core's raw partial to the root's slot, then signal.
                cb_wait_front(cb_sum_handoff, rows);
                noc_async_write(
                    get_read_ptr(cb_sum_handoff),
                    get_noc_addr(root_x, root_y, get_write_ptr(cb_partials_gathered) + my_slot * rows * stat_bytes),
                    rows * stat_bytes);
                noc_async_write_barrier();  // data before signal
                gather_sem.up(noc, root_x, root_y, 1);
                cb_pop_front(cb_sum_handoff, rows);

                // 3. reserve the landing slot FIRST: receive()'s ack means "free".
                cb_reserve_back(cb_row_final, rows);
                receiver.receive();
                cb_push_back(cb_row_final, rows);
            }
        }
    }

    if constexpr (NATIVE) {
        // Zero-copy output: compute packed into the shard itself.  Take the
        // completion barrier and leave the pages pushed — they ARE the tensor.
        cb_wait_front(cb_output_tiles, num_rows * WT_CHUNK);
        return;
    }

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
        const uint32_t first_tile_row = row_start + r0;

        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            if constexpr (RM) {
                const uint32_t stick_start = first_tile_row * TILE_DIM;
                uint32_t sticks = rows * TILE_DIM;
                if (stick_start + sticks > R_RM) {
                    sticks = R_RM - stick_start;  // short final tile-row
                }
                const uint32_t row_bytes = (c + 1 == NUM_W_CHUNKS) ? LAST_CHUNK_ROW_BYTES : CHUNK_ROW_BYTES;
                dataflow_kernel_lib::write_sticks_after_untilize<cb_output_sticks>(
                    out_acc, sticks, row_bytes, stick_start, /*byte_offset_within_page=*/c * CHUNK_ROW_BYTES);
            } else {
                for (uint32_t r = 0; r < rows; ++r) {
                    // + w_start: this core's width slice under a cross-core width
                    // split (0 on the whole-row schemes).
                    const uint32_t tile_base = (first_tile_row + r) * WT + w_start + c * WT_CHUNK;
                    cb_wait_front(cb_output_tiles, WT_CHUNK);
                    uint32_t l1_addr = get_read_ptr(cb_output_tiles);
                    for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                        const uint32_t wt = w_start + c * WT_CHUNK + w;
                        if (wt < WT) {  // a ragged width shard ends in pad tiles
                            noc_async_write_tile(tile_base + w, out_acc, l1_addr);
                        }
                        l1_addr += out_tile_bytes;
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_output_tiles, WT_CHUNK);
                }
            }
        }
    }
}
