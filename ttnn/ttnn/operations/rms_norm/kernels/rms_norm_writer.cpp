// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer (NoC1).
//
//  load_gamma            prologue: γ slice (row-0 lanes only) -> cb_gamma (TILE γ) | cb_gamma_rm (RM γ); once/kernel.
//  gather_partials_block W_SPLIT: cb_partial -> root's cb_gathered slot (r*w_split + line_pos), barrier, sem inc.
//  broadcast_rinv_block  W_SPLIT root: cb_rinv_bcast -> every line core's cb_rinv (Mcast helper, loopback incl.).
//  store_y_block         TILE: cb_y -> DRAM in y_chunk_tiles chunks (one barrier each, never straddling the ring);
//                        RM: cb_out_rm -> DRAM, one tile-row of sticks (valid W-slice bytes) per barrier.
//
// Raw NoC for the DRAM / gather paths (no kernel_lib helper covers them: op_design.md "Helpers considered and
// rejected"); the multicast goes through dataflow_kernel_lib::McastArgs.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t cb_gamma_rm = 2;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_partial = 5;
constexpr uint32_t cb_gathered = 6;
constexpr uint32_t cb_rinv_bcast = 8;
constexpr uint32_t cb_rinv = 9;
constexpr uint32_t cb_y = 10;
constexpr uint32_t cb_out_rm = 11;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;

void kernel_main() {
    constexpr bool is_rm = get_compile_time_arg_val(0) != 0;
    constexpr bool is_wsplit = get_compile_time_arg_val(1) != 0;
    constexpr bool has_gamma = get_compile_time_arg_val(2) != 0;
    constexpr bool gamma_is_rm = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t block_rows = get_compile_time_arg_val(4);
    constexpr uint32_t core_w_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t y_chunk_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t y_depth = get_compile_time_arg_val(7);
    constexpr uint32_t out_rm_depth = get_compile_time_arg_val(8);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t es = get_compile_time_arg_val(10);
    constexpr uint32_t y_page_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t g_page_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t es_g = get_compile_time_arg_val(13);
    constexpr uint32_t g_tile_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t Wt = get_compile_time_arg_val(15);
    constexpr uint32_t w_split = get_compile_time_arg_val(16);
    constexpr uint32_t sem_partials_id = get_compile_time_arg_val(17);
    constexpr uint32_t sem_rinv_local_id = get_compile_time_arg_val(18);
    constexpr uint32_t f32_tile_bytes = get_compile_time_arg_val(19);
    constexpr auto y_args = TensorAccessorArgs<20>();
    constexpr auto g_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto mcast = McastArgs<
        get_named_compile_time_arg_val("rinv_mcast_ct_offset"),
        get_named_compile_time_arg_val("rinv_mcast_rt_offset")>();
    (void)out_rm_depth;

    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);
    const uint32_t line_rows = get_arg_val<uint32_t>(3);
    const uint32_t w_start = get_arg_val<uint32_t>(4);
    const uint32_t w_valid = get_arg_val<uint32_t>(5);
    const uint32_t w_valid_elems = get_arg_val<uint32_t>(6);
    const uint32_t is_root = get_arg_val<uint32_t>(7);
    const uint32_t line_pos = get_arg_val<uint32_t>(8);
    const uint32_t root_x = get_arg_val<uint32_t>(9);
    const uint32_t root_y = get_arg_val<uint32_t>(10);
    const uint32_t num_sticks = get_arg_val<uint32_t>(11);

    const auto y_acc = TensorAccessor(y_args, y_addr, y_page_bytes);
    Noc noc;
    const uint32_t num_blocks = (line_rows + block_rows - 1) / block_rows;

    // ---------------- load_gamma (once per kernel) ----------------
    if constexpr (has_gamma) {
        const auto g_acc = TensorAccessor(g_args, g_addr, g_page_bytes);
        if constexpr (gamma_is_rm) {
            // One stick slice of the W-slice into a pre-zeroed row page; compute tilizes it (1 valid row).
            CircularBuffer g_rm_buf(cb_gamma_rm);
            cb_reserve_back(cb_gamma_rm, 1);
            noc.async_write_zeros(g_rm_buf, core_w_tiles * TILE_DIM * es_g);
            noc.write_zeros_l1_barrier();
            noc_async_read(
                g_acc.get_noc_addr(0, w_start * TILE_DIM * es_g), get_write_ptr(cb_gamma_rm), w_valid_elems * es_g);
            noc_async_read_barrier();
            cb_push_back(cb_gamma_rm, 1);
        } else {
            // TILE γ: only row 0 matters (row broadcast) -> read face0-row0 and face1-row0 of each valid tile.
            CircularBuffer g_buf(cb_gamma);
            cb_reserve_back(cb_gamma, core_w_tiles);
            noc.async_write_zeros(g_buf, core_w_tiles * g_tile_bytes);
            noc.write_zeros_l1_barrier();
            const uint32_t l1 = get_write_ptr(cb_gamma);
            constexpr uint32_t face_row_bytes = FACE_DIM * es_g;
            constexpr uint32_t face1_offset = FACE_DIM * FACE_DIM * es_g;
            for (uint32_t c = 0; c < w_valid; ++c) {
                const uint32_t page = w_start + c;
                noc_async_read(g_acc.get_noc_addr(page, 0), l1 + c * g_tile_bytes, face_row_bytes);
                noc_async_read(
                    g_acc.get_noc_addr(page, face1_offset), l1 + c * g_tile_bytes + face1_offset, face_row_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma, core_w_tiles);
        }
    }

    // ---------------- W_SPLIT plumbing ----------------
    Semaphore<> sem_partials(sem_partials_id);
    Semaphore<> sem_rinv_local(sem_rinv_local_id);
    // Landing addresses are the CB bases: this kernel never pushes cb_gathered / cb_rinv, so its write pointers
    // stay at the (uniform across cores) base.
    uint32_t gathered_base = 0;
    uint32_t rinv_base = 0;
    if constexpr (is_wsplit) {
        gathered_base = get_write_ptr(cb_gathered);
        rinv_base = get_write_ptr(cb_rinv);
    }

    // TILE y stream position inside the cb_y ring (chunks never straddle the ring end).
    constexpr uint32_t y_ring_tiles = y_depth * y_chunk_tiles;
    uint32_t y_ring_pos = 0;

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t row0 = b * block_rows;
        const uint32_t rows = (line_rows - row0 < block_rows) ? (line_rows - row0) : block_rows;

        if constexpr (is_wsplit) {
            // ---- gather_partials_block ----
            cb_wait_front(cb_partial, rows);
            const uint32_t src = get_read_ptr(cb_partial);
            for (uint32_t r = 0; r < rows; ++r) {
                noc_async_write(
                    src + r * f32_tile_bytes,
                    get_noc_addr(root_x, root_y, gathered_base + (r * w_split + line_pos) * f32_tile_bytes),
                    f32_tile_bytes);
            }
            noc_async_write_barrier();
            sem_partials.up(noc, root_x, root_y, 1);
            cb_pop_front(cb_partial, rows);

            // ---- broadcast_rinv_block (root) ----
            if (is_root) {
                cb_wait_front(cb_rinv_bcast, rows);
                auto sender = mcast.optional_sender(noc);
                ASSERT(sender.has_value());
                if (sender) {
                    sender->send(get_read_ptr(cb_rinv_bcast), rinv_base, rows * f32_tile_bytes);
                }
                cb_pop_front(cb_rinv_bcast, rows);
                sem_rinv_local.set(b + 1);  // send() barriered the loopback copy: own cb_rinv holds rinv(b)
            }
        }

        // ---- store_y_block ----
        if constexpr (is_rm) {
            constexpr uint32_t stick_stride = core_w_tiles * TILE_DIM * es;
            const uint32_t valid_bytes = w_valid_elems * es;
            const uint32_t col_offset = w_start * TILE_DIM * es;
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t stick0 = (row_start + row0 + r) * TILE_DIM;
                cb_wait_front(cb_out_rm, core_w_tiles);
                const uint32_t l1 = get_read_ptr(cb_out_rm);
                for (uint32_t s = 0; s < TILE_DIM; ++s) {
                    if (stick0 + s < num_sticks) {
                        noc_async_write(l1 + s * stick_stride, y_acc.get_noc_addr(stick0 + s, col_offset), valid_bytes);
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out_rm, core_w_tiles);
            }
        } else {
            const uint32_t block_tiles = rows * core_w_tiles;
            uint32_t i0 = 0;
            while (i0 < block_tiles) {
                uint32_t n = block_tiles - i0;
                if (n > y_chunk_tiles) {
                    n = y_chunk_tiles;
                }
                if (n > y_ring_tiles - y_ring_pos) {
                    n = y_ring_tiles - y_ring_pos;
                }
                cb_wait_front(cb_y, n);
                const uint32_t l1 = get_read_ptr(cb_y);
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t i = i0 + j;
                    const uint32_t r = i / core_w_tiles;
                    const uint32_t c = i - r * core_w_tiles;
                    if (c < w_valid) {
                        const uint32_t page = (row_start + row0 + r) * Wt + w_start + c;
                        noc_async_write(l1 + j * tile_bytes, y_acc.get_noc_addr(page), tile_bytes);
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_y, n);
                i0 += n;
                y_ring_pos += n;
                if (y_ring_pos == y_ring_tiles) {
                    y_ring_pos = 0;
                }
            }
        }
    }
    (void)es;
    (void)w_split;
    (void)line_pos;
}
