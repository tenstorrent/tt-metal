// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// groupnorm_sc_N_1_HW_C_b writer (BRISC, NoC1).
//
// Per block:
//   gather   : row 0 of the 2 local partial tiles (group lanes of S1 / S2) -> row `slot` of the root's
//              cb_gathered_partials tile pair slot/32, then a semaphore increment on the root.
//   root     : wait for cpi arrivals, publish the gathered tiles to compute; multicast [rstd, mean] from
//              cb_stats_out into every unit core's cb_group_stats (McastArgs; cpi == 1 is a local copy).
//   receiver : take the multicast into cb_group_stats.
//   output   : pass-2 chunks (resident tail first, then the re-read head) -> DRAM via TensorAccessor.
//
// Unused gather rows (slots >= cpi in the last tile pair) are zeroed once with NoC zeros; senders never
// touch them, so they stay zero across blocks.

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

namespace {
constexpr uint32_t cb_output_tiles = 5;
constexpr uint32_t cb_local_partials = 15;
constexpr uint32_t cb_gathered_partials = 16;
constexpr uint32_t cb_stats_out = 19;
constexpr uint32_t cb_group_stats = 20;

constexpr uint32_t F32_TILE_BYTES = 4096;
constexpr uint32_t F32_FACE_BYTES = 1024;
constexpr uint32_t F32_FACE_ROW_BYTES = 64;  // 16 fp32 lanes
constexpr uint32_t GATHER_SEMAPHORE_ID = 0;

// Byte offsets of the two half-rows (lanes 0..15, 16..31) of tile row `row` in a 32x32 fp32 tile.
FORCE_INLINE uint32_t face_row_lo(uint32_t row) {
    return row < 16 ? row * F32_FACE_ROW_BYTES : 2 * F32_FACE_BYTES + (row - 16) * F32_FACE_ROW_BYTES;
}
FORCE_INLINE uint32_t face_row_hi(uint32_t row) { return face_row_lo(row) + F32_FACE_BYTES; }
}  // namespace

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t CHUNK_HT = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t WT_TOTAL = get_compile_time_arg_val(3);
    constexpr uint32_t HT = get_compile_time_arg_val(4);
    constexpr uint32_t GX = get_compile_time_arg_val(5);
    constexpr uint32_t OUT_PAGE_BYTES = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();
    constexpr auto mc = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();
    using McastT = std::remove_cv_t<decltype(mc)>;

    constexpr uint32_t T = CHUNK_HT * W;
    constexpr uint32_t STATS_BYTES = 2 * F32_TILE_BYTES;

    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t unit_start = get_arg_val<uint32_t>(arg++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t row_start = get_arg_val<uint32_t>(arg++);
    const uint32_t rows_core = get_arg_val<uint32_t>(arg++);
    const uint32_t resident_chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t is_root = get_arg_val<uint32_t>(arg++);
    const uint32_t slot = get_arg_val<uint32_t>(arg++);
    const uint32_t root_x = get_arg_val<uint32_t>(arg++);
    const uint32_t root_y = get_arg_val<uint32_t>(arg++);
    const uint32_t cpi = get_arg_val<uint32_t>(arg++);

    const auto out_acc = TensorAccessor(out_args, out_addr, OUT_PAGE_BYTES);
    Noc noc;

    // CBs are allocated identically on every used core, so the local base is the root's base too.
    const uint32_t gathered_base = get_write_ptr(cb_gathered_partials);
    const uint32_t group_stats_base = get_write_ptr(cb_group_stats);
    const uint32_t sem_l1 = get_semaphore(GATHER_SEMAPHORE_ID);
    const uint64_t root_sem_noc = get_noc_addr(root_x, root_y, sem_l1);

    // Destination of this core's partial row inside the root's gathered tile pair slot/32.
    const uint32_t pair_base = gathered_base + 2 * (slot / 32) * F32_TILE_BYTES;
    const uint32_t row = slot % 32;
    const uint64_t dst_s1_lo = get_noc_addr(root_x, root_y, pair_base + face_row_lo(row));
    const uint64_t dst_s1_hi = get_noc_addr(root_x, root_y, pair_base + face_row_hi(row));
    const uint64_t dst_s2_lo = get_noc_addr(root_x, root_y, pair_base + F32_TILE_BYTES + face_row_lo(row));
    const uint64_t dst_s2_hi = get_noc_addr(root_x, root_y, pair_base + F32_TILE_BYTES + face_row_hi(row));

    if (is_root) {
        const uint32_t first_unused = cpi % 32;
        if (first_unused != 0) {
            // Zero rows first_unused..31 of both tiles of the last pair (the sender-sum matmul adds every row).
            DataflowBuffer gathered_dfb(cb_gathered_partials);
            const uint32_t last_pair = 2 * (R - 1) * F32_TILE_BYTES;
            for (uint32_t t = 0; t < 2; ++t) {
                const uint32_t tile = last_pair + t * F32_TILE_BYTES;
                if (first_unused < 16) {
                    const uint32_t lo = first_unused * F32_FACE_ROW_BYTES;
                    noc.async_write_zeros(gathered_dfb, F32_FACE_BYTES - lo, {.offset_bytes = tile + lo});
                    noc.async_write_zeros(
                        gathered_dfb, F32_FACE_BYTES - lo, {.offset_bytes = tile + F32_FACE_BYTES + lo});
                    noc.async_write_zeros(
                        gathered_dfb, 2 * F32_FACE_BYTES, {.offset_bytes = tile + 2 * F32_FACE_BYTES});
                } else {
                    const uint32_t lo = (first_unused - 16) * F32_FACE_ROW_BYTES;
                    noc.async_write_zeros(
                        gathered_dfb, F32_FACE_BYTES - lo, {.offset_bytes = tile + 2 * F32_FACE_BYTES + lo});
                    noc.async_write_zeros(
                        gathered_dfb, F32_FACE_BYTES - lo, {.offset_bytes = tile + 3 * F32_FACE_BYTES + lo});
                }
            }
            noc.write_zeros_l1_barrier();
        }
    }

    const uint32_t num_chunks = (rows_core + CHUNK_HT - 1) / CHUNK_HT;
    const uint32_t K = resident_chunks < num_chunks ? resident_chunks : num_chunks;
    const uint32_t first_resident = num_chunks - K;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t unit = unit_start + blk;
        const uint32_t n = unit / GX;
        const uint32_t wt0 = (unit % GX) * W;

        // ---------------- gather: local partial row -> root ----------------
        if (is_root) {
            cb_reserve_back(cb_gathered_partials, 2 * R);
        }
        cb_wait_front(cb_local_partials, 2);
        {
            const uint32_t src = get_read_ptr(cb_local_partials);
            noc_async_write(src, dst_s1_lo, F32_FACE_ROW_BYTES);
            noc_async_write(src + F32_FACE_BYTES, dst_s1_hi, F32_FACE_ROW_BYTES);
            noc_async_write(src + F32_TILE_BYTES, dst_s2_lo, F32_FACE_ROW_BYTES);
            noc_async_write(src + F32_TILE_BYTES + F32_FACE_BYTES, dst_s2_hi, F32_FACE_ROW_BYTES);
            noc_async_write_barrier();  // the row must land before the root sees the increment
            noc_semaphore_inc(root_sem_noc, 1);
        }
        cb_pop_front(cb_local_partials, 2);

        // ---------------- combine / broadcast ----------------
        if (is_root) {
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1);
            noc_semaphore_wait_min(sem, cpi);
            noc_semaphore_set(sem, 0);
            cb_push_back(cb_gathered_partials, 2 * R);

            cb_wait_front(cb_stats_out, 2);
            cb_reserve_back(cb_group_stats, 2);
            if constexpr (McastT::sender_available) {
                mc.sender(noc).send(get_read_ptr(cb_stats_out), group_stats_base, STATS_BYTES);
            }
            cb_push_back(cb_group_stats, 2);
            cb_pop_front(cb_stats_out, 2);
        } else {
            // Receiver-only cores exist only when cpi > 1 (the placement then compiles a receiver pipe).
            cb_reserve_back(cb_group_stats, 2);
            if constexpr (McastT::receiver_available) {
                mc.receiver(noc).receive();
            }
            cb_push_back(cb_group_stats, 2);
        }

        // ---------------- output: pass-2 order (resident tail first, then the re-read head) ----------------
        for (uint32_t jj = 0; jj < num_chunks; ++jj) {
            const uint32_t j = jj < K ? first_resident + jj : jj - K;
            const uint32_t row0 = row_start + j * CHUNK_HT;
            const uint32_t rows_left = rows_core - j * CHUNK_HT;
            const uint32_t rows_valid = rows_left < CHUNK_HT ? rows_left : CHUNK_HT;
            cb_wait_front(cb_output_tiles, T);
            uint32_t src = get_read_ptr(cb_output_tiles);
            for (uint32_t r = 0; r < rows_valid; ++r) {
                const uint32_t tile0 = (n * HT + row0 + r) * WT_TOTAL + wt0;
                for (uint32_t wt = 0; wt < W; ++wt) {
                    noc_async_write_page(tile0 + wt, out_acc, src);
                    src += OUT_PAGE_BYTES;
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_output_tiles, T);
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
