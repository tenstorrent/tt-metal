// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — writer (data movement): the cross-core combine + output store.
//
// Roles inside an image rectangle (RT arg):
//   ROOT   (p == 0)      zero-fills the gather tiles, signals gather-ready, writes its own record, waits for
//                        P_used records, hands the gather to compute, multicasts the reduced totals.
//   MEMBER (0 < p < P_used) waits gather-ready, writes its partial record into row p of the root's gather
//                        tiles, waits for the totals multicast.
//   IDLE   (p >= P_used) participates in the two multicast handshakes only.
// Then every non-idle core streams cb_out tiles of its own block to DRAM.
//
// Raw dataflow (per op_design.md → helpers considered and rejected): the record gather is P_used concurrent
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
    constexpr uint32_t num_col_groups = get_compile_time_arg_val(8);
    constexpr uint32_t num_row_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t Ht = get_compile_time_arg_val(10);
    constexpr uint32_t Ct = get_compile_time_arg_val(11);
    constexpr uint32_t gather_tiles_per_stat = get_compile_time_arg_val(12);
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(13);
    constexpr uint32_t MC_CT = 14;
    constexpr uint32_t MC_RT = 11;
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

    constexpr uint32_t num_stats = 2 * Kg;
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

    // cb_out -> DRAM tiles of this core's block, one barrier per tile-row of `cols` tiles.
    auto store_image = [&](uint32_t n) {
        for (uint32_t cg = 0; cg < num_col_groups; ++cg) {
            for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
                for (uint32_t i = 0; i < chunk_rows; ++i) {
                    cb_wait_front(cb_out, cols);
                    const uint32_t l1 = get_read_ptr(cb_out);
                    const uint32_t r = row_begin + rc * chunk_rows + i;
                    for (uint32_t j = 0; j < cols; ++j) {
                        const uint32_t t = col_begin + cg * cols + j;
                        const uint32_t page = n * Ht * Ct + r * Ct + t;
                        noc_async_write(l1 + j * y_tile_bytes, out_acc.get_noc_addr(page), y_tile_bytes);
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_out, cols);
                }
            }
        }
    };

    if (role == ROLE_ROOT) {
        auto sender = mc.sender(noc);
        for (uint32_t img = 0; img < image_count; ++img) {
            const uint32_t n = image_begin + img * image_stride;
            // gather tiles: rows >= P_used must be zero before any record can land
            gather.reserve_back(gather_tiles);
            noc.async_write_zeros(gather, gather_tiles * f32_tile_bytes);
            noc.write_zeros_l1_barrier();
            sender.send_signal(VALID);
            send_partial_record();
            sem_gather.wait_min((img + 1) * p_used);
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
            receiver.receive_signal();
            send_partial_record();
            cb_reserve_back(cb_totals_recv, num_stats);
            receiver.receive();
            cb_push_back(cb_totals_recv, num_stats);
            store_image(n);
        }
    } else {
        auto receiver = mc.receiver(noc);
        for (uint32_t img = 0; img < image_count; ++img) {
            receiver.receive_signal();
            receiver.receive();
        }
    }
}
