// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm writer (BRISC / NoC1). Owns BOTH halves of the cross-core combine
// and the output drain.
//
// Per block:
//   combine_stat_block (gather) — every member unicasts its block_row_tiles
//       partial tiles into the ROOT's cb_stat_gather at slot r*G + my_slot, then
//       bumps the gather semaphore. The root waits (block+1)*(G-1) and pushes
//       cb_stat_gather for its compute kernel. The tile-row-major slot layout
//       makes a tile-row's G contributions contiguous, which is what lets the
//       combine be a single eltwise_chain over grid(R, G).
//   combine_stat_block (mcast) — the root multicasts the finalized rstd tiles to
//       the group rectangle with src != dst, so INCLUDE_SRC loopback lands the
//       root's own copy in its cb_rstd too: cb_rstd has exactly one producer
//       (the writer) on EVERY member, root included.
//   store_block — the output block to DRAM, batched one tile-row per barrier.
//
// Helper substitutions (raw NoC instead of a kernel_lib helper), with reasons:
//   * The GATHER leg is raw noc_async_write + semaphore. mcast_pipe's SenderPipe
//     is a one-to-many broadcast of one buffer to a rectangle; the gather is
//     many-to-one into DISJOINT SLOTS of a single destination
//     (mcast_pipe.hpp:44-45 states its precondition as "one sender per receiver,
//     dst_l1 identical on all receivers" — the opposite direction). The RETURN
//     multicast in the same phase does use SenderPipe/ReceiverPipe.
//   * store_block on the TILE path uses raw noc_async_write_tile over a
//     TensorAccessor; write_sticks_after_untilize is ROW-MAJOR only
//     (tilize_helpers_dataflow.inl:82-85). The RM branch uses the helper.

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
constexpr uint32_t cb_stat_partial = 7;
constexpr uint32_t cb_stat_gather = 8;
constexpr uint32_t cb_rstd_send = 10;
constexpr uint32_t cb_rstd = 11;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
constexpr uint32_t TILE_HW_DIM = 32;
}  // namespace

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t CORE_W_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t TENSOR_W_TILES = get_compile_time_arg_val(1);
    constexpr bool IS_RM_OUT = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t W_GROUP_SIZE = get_compile_time_arg_val(3);
    constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(4);
    constexpr uint32_t MCAST_CT_BASE = 5;
    constexpr uint32_t MCAST_RT_BASE = 14;
    constexpr auto mc = McastArgs<MCAST_CT_BASE, MCAST_RT_BASE>();
    constexpr auto dst_args = TensorAccessorArgs<mc.next_compile_time_args_offset()>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(3);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(4);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(5);
    const uint32_t my_slot = get_arg_val<uint32_t>(6);
    const uint32_t is_root = get_arg_val<uint32_t>(7);
    const uint32_t root_x = get_arg_val<uint32_t>(8);
    const uint32_t root_y = get_arg_val<uint32_t>(9);
    const uint32_t num_sticks = get_arg_val<uint32_t>(10);
    const uint32_t stick_start = get_arg_val<uint32_t>(11);
    const uint32_t out_slice_bytes = get_arg_val<uint32_t>(12);
    const uint32_t out_byte_offset = get_arg_val<uint32_t>(13);

    Noc noc;
    Semaphore<> gather_sem(SEM_GATHER);

    const uint32_t stat_tile_bytes = get_tile_size(cb_stat_gather);
    // cb_stat_gather holds exactly block_row_tiles * W_GROUP_SIZE pages and the
    // root pushes/pops that many per block, so its write pointer is back at the
    // CB base at the start of every block — identical on every group member,
    // which is what lets a member address the root's slots by local pointer.
    const uint32_t gather_base = get_write_ptr(cb_stat_gather);

    const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);
    // Default page size == the accessor args' aligned page size, which is the
    // tile size on the tiled path and the stick size on the row-major path.
    const auto out_acc = TensorAccessor(dst_args, dst_addr);

    // --- store_block: drain one output block to DRAM -------------------------
    uint32_t sticks_done = 0;
    auto store_block = [&](uint32_t b, uint32_t rows_t) {
        if constexpr (IS_RM_OUT) {
            uint32_t sticks_this = rows_t * TILE_HW_DIM;
            if (sticks_this > num_sticks - sticks_done) {
                sticks_this = num_sticks - sticks_done;
            }
            write_sticks_after_untilize<cb_output_rm>(
                out_acc, sticks_this, out_slice_bytes, stick_start + sticks_done, out_byte_offset);
            sticks_done += sticks_this;
        } else {
            for (uint32_t r = 0; r < rows_t; ++r) {
                cb_wait_front(cb_output_tiles, CORE_W_TILES);
                const uint32_t src = get_read_ptr(cb_output_tiles);
                const uint32_t row_tile = row_tile_start + b * block_row_tiles + r;
                const uint32_t base = row_tile * TENSOR_W_TILES + w_tile_start;
                for (uint32_t c = 0; c < CORE_W_TILES; ++c) {
                    noc_async_write_tile(base + c, out_acc, src + c * out_tile_bytes);
                }
                // One barrier per tile-row (core_w_tiles tiles), never per tile.
                noc_async_write_barrier();
                cb_pop_front(cb_output_tiles, CORE_W_TILES);
            }
        }
    };

    // --- combine_stat_block (gather leg) -------------------------------------
    auto gather_partials = [&](uint32_t b, uint32_t rows_t) {
        cb_wait_front(cb_stat_partial, rows_t);
        const uint32_t src = get_read_ptr(cb_stat_partial);
        for (uint32_t r = 0; r < rows_t; ++r) {
            const uint32_t dst = gather_base + (r * W_GROUP_SIZE + my_slot) * stat_tile_bytes;
            noc_async_write(src + r * stat_tile_bytes, get_noc_addr(root_x, root_y, dst), stat_tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_stat_partial, rows_t);
        if constexpr (W_GROUP_SIZE > 1) {
            if (!is_root) {
                gather_sem.up(noc, root_x, root_y, 1);
            }
        }
        if (is_root) {
            cb_reserve_back(cb_stat_gather, rows_t * W_GROUP_SIZE);
            if constexpr (W_GROUP_SIZE > 1) {
                gather_sem.wait_min((b + 1) * (W_GROUP_SIZE - 1));
            }
            cb_push_back(cb_stat_gather, rows_t * W_GROUP_SIZE);
        }
    };

    // The two mcast faces are constructed ONCE, outside the block loop: the
    // ReceiverPipe ctor kernel-inits its data_ready cell, which must not be
    // re-run after the sender has started broadcasting.
    if (is_root) {
        auto sender = mc.sender(noc);
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
            gather_partials(b, rows_t);
            cb_reserve_back(cb_rstd, rows_t);
            const uint32_t rstd_dst = get_write_ptr(cb_rstd);
            cb_wait_front(cb_rstd_send, rows_t);
            // src != dst selects INCLUDE_SRC loopback, so the root lands its own
            // copy in cb_rstd through the same path as every other member.
            sender.send(get_read_ptr(cb_rstd_send), rstd_dst, rows_t * stat_tile_bytes);
            cb_pop_front(cb_rstd_send, rows_t);
            cb_push_back(cb_rstd, rows_t);
            store_block(b, rows_t);
        }
    } else {
        auto receiver = mc.receiver(noc);
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
            gather_partials(b, rows_t);
            cb_reserve_back(cb_rstd, rows_t);
            receiver.receive();
            cb_push_back(cb_rstd, rows_t);
            store_block(b, rows_t);
        }
    }
}
