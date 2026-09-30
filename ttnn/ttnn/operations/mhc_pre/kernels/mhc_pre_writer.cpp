// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_pre writer (BRISC, NoC1). Per block, in this mandatory order (gather-slot reuse invariant):
//
//   send_partial_block        cb_partial [mix rows | sumsq rows] -> root cb_gathered slot [rank]
//                              (push model; slot layout [mix x block_token_tiles | sumsq x block_token_tiles]),
//                              then one semaphore increment on the root.
//   (root) gather credit       wait the monotonic arrival counter >= group_cores*(block+1), push cb_gathered.
//   broadcast_combined_block   root: SenderPipe::send of cb_combined over the group rectangle (root excluded);
//                              non-root: ReceiverPipe::receive into its cb_combined landing (same address).
//   scatter                    S -> coefficient-major tiles in cb_coef_in (25 slots x 32 lanes per row).
//   expand_pre_block           cb_coef_out pre slots -> column 0 of n pre-column tiles (cb_pre_cols);
//                              owned rows: post slots -> staged post tile -> DRAM.
//   store_y_block              cb_y_out -> y DRAM, windows of y_chunk_tiles (wrap-aware on the CB ring).
//   store_comb_block           owned rows: cb_comb_coef comb slots -> staged comb tile -> DRAM.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "mhc_pre_layout.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t MCAST_CT_BASE = 17;
constexpr uint32_t MCAST_RT_BASE = 11;

void kernel_main() {
    constexpr uint32_t cb_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gathered = get_compile_time_arg_val(1);
    constexpr uint32_t cb_combined = get_compile_time_arg_val(2);
    constexpr uint32_t cb_coef_in = get_compile_time_arg_val(3);
    constexpr uint32_t cb_coef_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_comb_coef = get_compile_time_arg_val(5);
    constexpr uint32_t cb_pre_cols = get_compile_time_arg_val(6);
    constexpr uint32_t cb_y_out = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out_stage = get_compile_time_arg_val(8);
    constexpr uint32_t n_streams = get_compile_time_arg_val(9);
    constexpr uint32_t block_token_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t tensor_c_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t group_cores = get_compile_time_arg_val(12);
    constexpr uint32_t y_chunk_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t y_cb_pages = get_compile_time_arg_val(14);  // y_depth * y_chunk_tiles
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(15);
    constexpr uint32_t mix_cols = get_compile_time_arg_val(16);  // n*(n+2); slot mix_cols holds sum(x^2)
    constexpr auto mc = McastArgs<MCAST_CT_BASE, MCAST_RT_BASE>();
    constexpr auto y_args = TensorAccessorArgs<mc.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t post_addr = get_arg_val<uint32_t>(1);
    const uint32_t comb_addr = get_arg_val<uint32_t>(2);
    const uint32_t t_start = get_arg_val<uint32_t>(3);
    const uint32_t core_token_tiles = get_arg_val<uint32_t>(4);
    const uint32_t c_start = get_arg_val<uint32_t>(5);
    const uint32_t core_c_tiles = get_arg_val<uint32_t>(6);
    const uint32_t num_blocks = get_arg_val<uint32_t>(7);
    const uint32_t rank = get_arg_val<uint32_t>(8);
    const uint32_t root_x = get_arg_val<uint32_t>(9);
    const uint32_t root_y = get_arg_val<uint32_t>(10);

    using mhc_layout::rc_index;
    using mhc_layout::slot_index;
    using mhc_layout::TILE_DATUMS;

    const uint32_t f_tile_bytes = get_tile_size(cb_partial);  // fp32 tile
    const uint32_t y_tile_bytes = get_tile_size(cb_y_out);
    const auto y_acc = TensorAccessor(y_args, y_addr, y_tile_bytes);
    const auto post_acc = TensorAccessor(post_args, post_addr, f_tile_bytes);
    const auto comb_acc = TensorAccessor(comb_args, comb_addr, f_tile_bytes);

    Noc noc;

    // Writer-private staging tiles (post, comb). Padding columns are zeroed once and never written.
    const uint32_t stage_post = get_write_ptr(cb_out_stage);
    const uint32_t stage_comb = stage_post + f_tile_bytes;
    {
        CircularBuffer stage_cb(cb_out_stage);
        noc.async_write_zeros(stage_cb, 2 * f_tile_bytes);
        noc.write_zeros_l1_barrier();
    }

    // Uniform-address CBs (allocated identically on every launched core).
    constexpr uint32_t slot_tiles = 2 * block_token_tiles;
    const uint32_t gathered_base = get_write_ptr(cb_gathered);
    const uint32_t combined_base = get_write_ptr(cb_combined);
    const uint32_t sem_addr = get_semaphore(sem_gather_id);
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    const uint64_t root_sem_noc = get_noc_addr(root_x, root_y, sem_addr);
    const uint64_t root_slot_noc = get_noc_addr(root_x, root_y, gathered_base + rank * slot_tiles * f_tile_bytes);

    // Group combine pipes (the group rectangle, root = rank 0 is the fixed sender, excluded from the landing).
    auto sender = mc.sender(noc);
    auto receiver = mc.receiver(noc);

    uint32_t y_ring_pos = 0;  // position (in pages) of cb_y_out's read pointer within its ring

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t row0 = block_idx * block_token_tiles;
        const uint32_t extent =
            (core_token_tiles - row0) < block_token_tiles ? (core_token_tiles - row0) : block_token_tiles;

        // ---- send_partial_block ----
        cb_wait_front(cb_partial, 2 * extent);
        {
            const uint32_t src = get_read_ptr(cb_partial);
            noc_async_write(src, root_slot_noc, extent * f_tile_bytes);
            noc_async_write(
                src + extent * f_tile_bytes, root_slot_noc + block_token_tiles * f_tile_bytes, extent * f_tile_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(root_sem_noc, 1);
        }
        cb_pop_front(cb_partial, 2 * extent);

        // ---- gather credit (root) + broadcast_combined_block ----
        uint32_t s_addr;
        if (rank == 0) {
            cb_reserve_back(cb_gathered, group_cores * slot_tiles);
            noc_semaphore_wait_min(sem_ptr, group_cores * (block_idx + 1));
            cb_push_back(cb_gathered, group_cores * slot_tiles);
            cb_wait_front(cb_combined, slot_tiles);
            s_addr = get_read_ptr(cb_combined);
            if constexpr (group_cores > 1) {
                sender.send(s_addr, s_addr, slot_tiles * f_tile_bytes);
            }
        } else {
            receiver.receive();
            s_addr = combined_base;
        }

        // ---- scatter S -> coefficient-major tiles ----
        cb_reserve_back(cb_coef_in, extent);
        {
            volatile tt_l1_ptr float* s = reinterpret_cast<volatile tt_l1_ptr float*>(s_addr);
            volatile tt_l1_ptr float* d = reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_coef_in));
            for (uint32_t t = 0; t < extent; ++t) {
                volatile tt_l1_ptr float* mix = s + t * TILE_DATUMS;
                volatile tt_l1_ptr float* sq = s + (block_token_tiles + t) * TILE_DATUMS;
                volatile tt_l1_ptr float* dt = d + t * TILE_DATUMS;
                for (uint32_t l = 0; l < 32; ++l) {
                    for (uint32_t k = 0; k < mix_cols; ++k) {
                        dt[slot_index(k, l)] = mix[rc_index(l, k)];
                    }
                    dt[slot_index(mix_cols, l)] = sq[rc_index(l, 0)];
                }
            }
        }
        cb_push_back(cb_coef_in, extent);
        if (rank == 0) {
            cb_pop_front(cb_combined, slot_tiles);
        }

        // ---- expand_pre_block + post (owned rows) ----
        cb_wait_front(cb_coef_out, extent);
        {
            volatile tt_l1_ptr float* co = reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(cb_coef_out));
            cb_reserve_back(cb_pre_cols, n_streams * extent);
            volatile tt_l1_ptr float* pc = reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_pre_cols));
            for (uint32_t t = 0; t < extent; ++t) {
                for (uint32_t i = 0; i < n_streams; ++i) {
                    volatile tt_l1_ptr float* dst = pc + (t * n_streams + i) * TILE_DATUMS;
                    for (uint32_t l = 0; l < 32; ++l) {
                        dst[rc_index(l, 0)] = co[t * TILE_DATUMS + slot_index(i, l)];
                    }
                }
            }
            cb_push_back(cb_pre_cols, n_streams * extent);

            volatile tt_l1_ptr float* sp = reinterpret_cast<volatile tt_l1_ptr float*>(stage_post);
            for (uint32_t t = 0; t < extent; ++t) {
                if (((row0 + t) % group_cores) != rank) {
                    continue;
                }
                for (uint32_t l = 0; l < 32; ++l) {
                    for (uint32_t i = 0; i < n_streams; ++i) {
                        sp[rc_index(l, i)] = co[t * TILE_DATUMS + slot_index(n_streams + i, l)];
                    }
                }
                noc_async_write_page(t_start + row0 + t, post_acc, stage_post);
                noc_async_write_barrier();
            }
        }
        cb_pop_front(cb_coef_out, extent);

        // ---- store_y_block ----
        for (uint32_t t = 0; t < extent; ++t) {
            const uint32_t y_row_page = (t_start + row0 + t) * tensor_c_tiles + c_start;
            uint32_t c = 0;
            while (c < core_c_tiles) {
                uint32_t w = core_c_tiles - c;
                if (w > y_chunk_tiles) {
                    w = y_chunk_tiles;
                }
                if (w > y_cb_pages - y_ring_pos) {
                    w = y_cb_pages - y_ring_pos;
                }
                cb_wait_front(cb_y_out, w);
                const uint32_t rp = get_read_ptr(cb_y_out);
                for (uint32_t j = 0; j < w; ++j) {
                    noc_async_write_page(y_row_page + c + j, y_acc, rp + j * y_tile_bytes);
                }
                noc_async_write_barrier();
                cb_pop_front(cb_y_out, w);
                y_ring_pos += w;
                if (y_ring_pos == y_cb_pages) {
                    y_ring_pos = 0;
                }
                c += w;
            }
        }

        // ---- store_comb_block (owned rows) ----
        for (uint32_t t = 0; t < extent; ++t) {
            if (((row0 + t) % group_cores) != rank) {
                continue;
            }
            cb_wait_front(cb_comb_coef, 1);
            volatile tt_l1_ptr float* cc = reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(cb_comb_coef));
            volatile tt_l1_ptr float* sc = reinterpret_cast<volatile tt_l1_ptr float*>(stage_comb);
            for (uint32_t l = 0; l < 32; ++l) {
                for (uint32_t q = 0; q < n_streams * n_streams; ++q) {
                    sc[rc_index(l, q)] = cc[slot_index(2 * n_streams + q, l)];
                }
            }
            noc_async_write_page(t_start + row0 + t, comb_acc, stage_comb);
            noc_async_write_barrier();
            cb_pop_front(cb_comb_coef, 1);
        }
    }
    noc_async_atomic_barrier();
}
