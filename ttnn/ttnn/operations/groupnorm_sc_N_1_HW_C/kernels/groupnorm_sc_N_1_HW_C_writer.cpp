// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — writer (data movement): the cross-core combine + output store.
//
// Roles inside an image rectangle (RT arg):
//   ROOT   (p == 0)      zero-fills the PAD rows (p >= P_used) of the gather tiles once, writes its own record,
//                        waits for every rectangle core's semaphore increment, hands the gather to compute,
//                        multicasts the reduced totals (no pre-handshake).
//   MEMBER (0 < p < P_used) writes its partial record into row p of the root's gather tiles, increments the
//                        root's gather semaphore, waits for the totals multicast.
//   IDLE   (p >= P_used) increments the gather semaphore (so the root knows its ReceiverPipe exists) and
//                        receives the totals multicast.
// Then every non-idle core streams cb_out tiles of its own block to DRAM. Blocks are ragged (op_design.md ->
// Work Distribution): compute pushes the nominal `chunk` pages per block with the valid_rows x valid_cols output
// tiles dense at the front; the writer drains the nominal count and stores only the valid tiles.
//
// Why no gather-ready signal / pre-handshake: records only ever touch rows p < P_used and the root only
// ever zeroes rows >= P_used, so the two never overlap and need no ordering. The totals flag cannot be
// clobbered by a late ReceiverPipe constructor (which resets the flag) because every rectangle core
// increments the gather semaphore AFTER constructing its receiver, and the root broadcasts only after all
// increments arrived. Saves two multicast round trips per image versus handshake=True.
//
// Raw dataflow (per op_design.md -> helpers considered and rejected): the record gather is P_used concurrent
// unicasts of 2*Kg non-contiguous 64 B face-row chunks landing at per-sender offsets — SenderPipe::send
// multicasts ONE contiguous block from ONE sender, so it does not fit; the totals broadcast does use it.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "groupnorm_sc_N_1_HW_C_ragged.hpp"

using namespace dataflow_kernel_lib;

namespace {
constexpr uint32_t ROLE_IDLE = 0;
constexpr uint32_t ROLE_MEMBER = 1;
constexpr uint32_t ROLE_ROOT = 2;
}  // namespace

void kernel_main() {
    // ---------------- compile-time args ----------------
    constexpr uint32_t cb_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(1);
    constexpr uint32_t cb_totals_src = get_compile_time_arg_val(2);
    constexpr uint32_t cb_totals_recv = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t Kg = get_compile_time_arg_val(5);
    constexpr uint32_t cols = get_compile_time_arg_val(6);
    constexpr uint32_t chunk_rows = get_compile_time_arg_val(7);
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t Ct = get_compile_time_arg_val(9);
    constexpr uint32_t gather_tiles_per_stat = get_compile_time_arg_val(10);
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(11);
    constexpr uint32_t out_block = get_compile_time_arg_val(12);  // tiles per store barrier (divides chunk)
    constexpr uint32_t MC_CT = 13;
    constexpr uint32_t MC_RT = 14;
    constexpr auto mc = McastArgs<MC_CT, MC_RT>();
    constexpr auto out_args = TensorAccessorArgs<mc.next_compile_time_args_offset()>();

    // ---------------- runtime args ----------------
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t image_begin = get_arg_val<uint32_t>(1);
    const uint32_t image_count = get_arg_val<uint32_t>(2);
    const uint32_t image_stride = get_arg_val<uint32_t>(3);
    const uint32_t row_begin = get_arg_val<uint32_t>(4);
    const uint32_t col_begin = get_arg_val<uint32_t>(5);
    const uint32_t role = get_arg_val<uint32_t>(6);
    const uint32_t p = get_arg_val<uint32_t>(7);
    const uint32_t p_used = get_arg_val<uint32_t>(8);
    const uint32_t root_x = get_arg_val<uint32_t>(9);
    const uint32_t root_y = get_arg_val<uint32_t>(10);
    const uint32_t num_participants = get_arg_val<uint32_t>(11);  // every core of the rectangle (incl. idle)
    const uint32_t Ht_core = get_arg_val<uint32_t>(12);
    const uint32_t Ct_core = get_arg_val<uint32_t>(13);

    constexpr uint32_t num_stats = 2 * Kg;
    constexpr uint32_t chunk = chunk_rows * cols;
    constexpr uint32_t out_blocks_per_chunk = chunk / out_block;
    static_assert(chunk % out_block == 0, "out_block must divide the chunk (host derives it so)");
    constexpr uint32_t f32_tile_bytes = get_tile_size(cb_partial);
    constexpr uint32_t gather_tiles = num_stats * gather_tiles_per_stat;
    constexpr uint32_t y_tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t face_bytes = 1024;  // 16x16 Float32 face
    constexpr uint32_t face_row_bytes = 64;

    Noc noc;
    CircularBuffer gather(cb_gather);
    Semaphore<> sem_gather(sem_gather_id);
    const auto out_acc = TensorAccessor(out_args, out_addr, y_tile_bytes);

    // Landing addresses: identical allocation on every core of the rectangle; both CBs have capacity equal
    // to their per-image quantum, so their write pointers return to the base after every image.
    const uint32_t gather_base = gather.get_write_ptr();
    const uint32_t totals_recv_base = get_write_ptr(cb_totals_recv);

    // Ragged accounting (idle cores have Ht_core = Ct_core = 0 and never store).
    const auto row_axis = groupnorm_ragged::split(Ht_core, chunk_rows);
    const auto col_axis = groupnorm_ragged::split(Ct_core, cols);
    const uint32_t num_row_chunks = row_axis.count;
    const uint32_t num_col_groups = col_axis.count;

    // Row 0 of each of this core's 2*Kg partial tiles -> row p of gather tile (p / 32) of the same statistic.
    auto send_partial_record = [&]() {
        cb_wait_front(cb_partial, num_stats);
        const uint32_t src = get_read_ptr(cb_partial);
        const uint32_t gt = p / 32;
        const uint32_t r = p % 32;
        const uint32_t row_face = (r >= 16) ? 2u : 0u;
        for (uint32_t s = 0; s < num_stats; ++s) {
            const uint32_t dst_tile = gt * num_stats + s;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t src_off = s * f32_tile_bytes + half * face_bytes;
                const uint32_t dst_off =
                    dst_tile * f32_tile_bytes + (row_face + half) * face_bytes + (r & 15) * face_row_bytes;
                noc_async_write(src + src_off, get_noc_addr(root_x, root_y, gather_base + dst_off), face_row_bytes);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_partial, num_stats);
        sem_gather.up(noc, root_x, root_y, 1);
    };

    // cb_out -> DRAM tiles of this core's block. Compute packs the chunk row-major (chunk_rows x cols, one push
    // per tile); the writer drains it in `out_block`-tile groups with ONE barrier per group, so the number of
    // tiles in flight per barrier is a host knob (OUT_BLOCK_TILES_TARGET) and not an accident of `cols`
    // (which is 1 whenever the ct split gives a core a single tile-column).
    // Ragged blocks: the valid_rows x valid_cols output tiles sit dense at the front of the block's nominal
    // `chunk` pages (idx = r * valid_cols + c); pages idx >= valid carry no data and are only drained.
    auto store_image = [&](uint32_t n) {
        for (uint32_t cg = 0; cg < num_col_groups; ++cg) {
            const uint32_t valid_cols = col_axis.valid(cg, cols);
            for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
                const uint32_t valid_rows = row_axis.valid(rc, chunk_rows);
                const uint32_t valid = valid_rows * valid_cols;
                for (uint32_t b = 0; b < out_blocks_per_chunk; ++b) {
                    cb_wait_front(cb_out, out_block);
                    const uint32_t l1 = get_read_ptr(cb_out);
                    for (uint32_t k = 0; k < out_block; ++k) {
                        const uint32_t idx = b * out_block + k;  // linear tile index inside the chunk
                        if (idx >= valid) {
                            break;
                        }
                        const uint32_t r = row_begin + rc * chunk_rows + idx / valid_cols;
                        const uint32_t t = col_begin + cg * cols + idx % valid_cols;
                        const uint32_t page = n * Ht * Ct + r * Ct + t;
                        noc_async_write(l1 + k * y_tile_bytes, out_acc.get_noc_addr(page), y_tile_bytes);
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_out, out_block);
                }
            }
        }
    };

    if (role == ROLE_ROOT) {
        auto sender = mc.sender(noc);
        // Zero the pad rows (cores p >= P_used) of every gather tile ONCE. Record rows are never zeroed, so
        // a record that lands before this completes is not disturbed; pad rows are never written afterwards.
        {
            uint32_t issued = 0;
            for (uint32_t gt = 0; gt < gather_tiles_per_stat; ++gt) {
                const uint32_t first_pad_row = (p_used > gt * 32) ? (p_used - gt * 32) : 0u;  // within tile
                if (first_pad_row >= 32) {
                    continue;
                }
                for (uint32_t s = 0; s < num_stats; ++s) {
                    const uint32_t tile_off = (gt * num_stats + s) * f32_tile_bytes;
                    for (uint32_t face = 0; face < 4; ++face) {
                        const uint32_t face_row0 = (face >> 1) * 16;
                        const uint32_t r0 = (first_pad_row > face_row0) ? (first_pad_row - face_row0) : 0u;
                        if (r0 >= 16) {
                            continue;
                        }
                        noc.async_write_zeros(
                            gather,
                            (16 - r0) * face_row_bytes,
                            {.offset_bytes = tile_off + face * face_bytes + r0 * face_row_bytes});
                        ++issued;
                    }
                }
            }
            if (issued > 0) {
                noc.write_zeros_l1_barrier();
            }
        }
        for (uint32_t img = 0; img < image_count; ++img) {
            const uint32_t n = image_begin + img * image_stride;
            gather.reserve_back(gather_tiles);  // blocks until root compute consumed the previous image
            send_partial_record();
            sem_gather.wait_min((img + 1) * num_participants);
            gather.push_back(gather_tiles);
            // totals: root compute reduced the gather; multicast to the rectangle (self copy when P_n == 1)
            cb_wait_front(cb_totals_src, num_stats);
            cb_reserve_back(cb_totals_recv, num_stats);
            sender.send(get_read_ptr(cb_totals_src), totals_recv_base, num_stats * f32_tile_bytes);
            cb_pop_front(cb_totals_src, num_stats);
            cb_push_back(cb_totals_recv, num_stats);
            store_image(n);
        }
    } else if (role == ROLE_MEMBER) {
        auto receiver = mc.receiver(noc);
        for (uint32_t img = 0; img < image_count; ++img) {
            const uint32_t n = image_begin + img * image_stride;
            send_partial_record();  // ends with the gather semaphore increment (after the receiver exists)
            cb_reserve_back(cb_totals_recv, num_stats);
            receiver.receive();
            cb_push_back(cb_totals_recv, num_stats);
            store_image(n);
        }
    } else {
        auto receiver = mc.receiver(noc);
        for (uint32_t img = 0; img < image_count; ++img) {
            sem_gather.up(noc, root_x, root_y, 1);  // "my receiver exists"; no record
            receiver.receive();
        }
    }
}
