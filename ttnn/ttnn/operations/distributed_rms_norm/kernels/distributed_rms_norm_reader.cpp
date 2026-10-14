// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm — worker reader (NCRISC, NoC0).
//   load_block           : x block (block_rows x core_w_tiles) -> cb_x_resident, one tile row per barrier;
//                          gamma (row 0 of each tile, no tilize) and the reduce scaler once per call.
//   gather_sources_block : wait stats-ready, pull the G*S_w fp32 records of every tile row of the block from the
//                          port cores into cb_stats_combine (source s -> row s%32 of combine tile s/32).
//
// Raw-API justification (gamma placement, record gather): the data lands in tile rows, not whole tiles; there is
// no dataflow helper for a sub-tile row placement (see op_design.md "Helpers considered and rejected").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_stats_combine = get_compile_time_arg_val(3);
    constexpr bool has_gamma = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t sem_stats_ready = get_compile_time_arg_val(5);
    constexpr uint32_t gamma_elem_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t gamma_page_bytes = get_compile_time_arg_val(7);  // one stick = the whole W_local slice
    constexpr uint32_t record_bytes = get_compile_time_arg_val(8);      // host RECORD_BYTES: 32 fp32 partial sums
    constexpr bool gamma_full_rows = get_compile_time_arg_val(9) != 0;  // replicate row 0 (fused, no-prescale path)
    constexpr auto x_args = TensorAccessorArgs<10>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    constexpr uint32_t half_record = record_bytes / 2;  // 16 fp32 = one face row
    constexpr uint32_t f32_tile_bytes = 4096;
    constexpr uint32_t f32_face_bytes = 1024;
    constexpr uint32_t f32_face_row_bytes = 64;

    size_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(a++);
    const uint32_t row_start = get_arg_val<uint32_t>(a++);   // first global tile row of this core
    const uint32_t core_rows = get_arg_val<uint32_t>(a++);   // tile rows assigned to this core
    const uint32_t block_rows = get_arg_val<uint32_t>(a++);  // rows per block (knob)
    const uint32_t w_start = get_arg_val<uint32_t>(a++);     // first W tile of this core's chunk
    const uint32_t cw = get_arg_val<uint32_t>(a++);          // W tiles of this core's chunk
    const uint32_t wt_total = get_arg_val<uint32_t>(a++);    // tensor_w_tiles
    const uint32_t scratch_addr = get_arg_val<uint32_t>(a++);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t rec_base = get_arg_val<uint32_t>(a++);  // link_rec_base of my row group
    const uint32_t s_w = get_arg_val<uint32_t>(a++);
    const uint32_t n_src_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t group_size = get_arg_val<uint32_t>(a++);
    const uint32_t ready_per_round = get_arg_val<uint32_t>(a++);  // stats-ready increments per round (2L)
    const uint32_t ports_idx = a;                                 // G x (noc x, noc y): port holding slot q

    if (core_rows == 0) {
        return;  // idle core of the worker rectangle
    }

    // scaler (once per kernel): SUM -> 1.0, full tile (the same pattern serves REDUCE_ROW and REDUCE_COL)
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    const uint32_t x_tile_bytes = get_tile_size(cb_x_resident);
    const auto x = TensorAccessor(x_args, x_addr, x_tile_bytes);
    volatile tt_l1_ptr uint32_t* stats_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_stats_ready));
    Noc noc;
    CircularBuffer combine_cb(cb_stats_combine);

    // gamma (once per call, after the first x block so the hot single-page read does not delay x):
    // one contiguous read of the chunk's cw*32 elements into the front of each group of <= 32 tiles, then
    // tile t's segment is placed in its row 0 (elements 0..15 -> face 0 row 0, 16..31 -> face 1 row 0).
    // Placement runs from the last tile down: tile t >= 1 lands beyond the staging bytes, tile 0 last.
    auto load_gamma = [&]() {
        if constexpr (has_gamma) {
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_tiles);
            const auto gamma = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
            constexpr uint32_t seg_bytes = 32 * gamma_elem_bytes;
            constexpr uint32_t half_words = 16 * gamma_elem_bytes / 4;
            constexpr uint32_t face_bytes = 256 * gamma_elem_bytes;
            constexpr uint32_t group_tiles = 32;  // staging (n * seg_bytes) stays inside the group's first tile
            cb_reserve_back(cb_gamma_tiles, cw);
            const uint32_t base = get_write_ptr(cb_gamma_tiles);
            for (uint32_t g0 = 0; g0 < cw; g0 += group_tiles) {
                const uint32_t n = (cw - g0) < group_tiles ? (cw - g0) : group_tiles;
                const uint32_t gbase = base + g0 * gamma_tile_bytes;
                noc_async_read(gamma.get_noc_addr(0, (w_start + g0) * seg_bytes), gbase, n * seg_bytes);
                noc_async_read_barrier();
                invalidate_l1_cache();
                for (uint32_t t = n; t-- > 0;) {
                    volatile tt_l1_ptr uint32_t* src =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gbase + t * seg_bytes);
                    volatile tt_l1_ptr uint32_t* row0 =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gbase + t * gamma_tile_bytes);
                    volatile tt_l1_ptr uint32_t* row0_face1 =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gbase + t * gamma_tile_bytes + face_bytes);
                    for (uint32_t i = 0; i < half_words; ++i) {
                        row0_face1[i] = src[half_words + i];
                    }
                    if (t > 0) {
                        for (uint32_t i = 0; i < half_words; ++i) {
                            row0[i] = src[i];
                        }
                    }
                }
                if constexpr (gamma_full_rows) {
                    // DEST-reuse multiply has no broadcast: copy row 0 of faces 0/1 into every row of all 4 faces
                    for (uint32_t t = 0; t < n; ++t) {
                        const uint32_t tb = gbase + t * gamma_tile_bytes;
                        volatile tt_l1_ptr uint32_t* f0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tb);
                        volatile tt_l1_ptr uint32_t* f1 =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tb + face_bytes);
                        for (uint32_t face = 0; face < 4; ++face) {
                            volatile tt_l1_ptr uint32_t* src_row = (face & 1) ? f1 : f0;
                            volatile tt_l1_ptr uint32_t* dst =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tb + face * face_bytes);
                            for (uint32_t r = (face < 2) ? 1 : 0; r < 16; ++r) {
                                for (uint32_t i = 0; i < half_words; ++i) {
                                    dst[r * half_words + i] = src_row[i];
                                }
                            }
                        }
                    }
                }
            }
            cb_push_back(cb_gamma_tiles, cw);
        }
    };

    uint32_t round = 0;
    for (uint32_t r0 = 0; r0 < core_rows; r0 += block_rows, ++round) {
        const uint32_t rows = (core_rows - r0) < block_rows ? (core_rows - r0) : block_rows;

        // ---- load_block: one tile row (cw tiles) per barrier ----
        for (uint32_t j = 0; j < rows; ++j) {
            cb_reserve_back(cb_x_resident, cw);
            uint32_t wp = get_write_ptr(cb_x_resident);
            const uint32_t tile0 = (row_start + r0 + j) * wt_total + w_start;
            for (uint32_t t = 0; t < cw; ++t) {
                noc_async_read_page(tile0 + t, x, wp);
                wp += x_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_x_resident, cw);
        }
        if (round == 0) {
            load_gamma();
        }

        // ---- gather_sources_block ----
        noc_semaphore_wait_min(stats_ready, ready_per_round * (round + 1));
        // Records of (source device q, tile row j) sit in one block of s_w * 128 B: [half 0 of w = 0..s_w-1]
        // [half 1 of w = 0..s_w-1]. Sources s = q*s_w + w are consecutive rows of the combine tiles, so a run of
        // sources inside one 16-row face pair moves as two reads (one per half).
        const uint32_t n_tiles = n_src_tiles * rows;
        const uint32_t n_src = group_size * s_w;
        cb_reserve_back(cb_stats_combine, n_tiles);
        if ((n_src & 31) != 0) {
            // rows beyond G*S_w of the last source tile must not enter the sum
            noc.async_write_zeros(
                combine_cb, rows * f32_tile_bytes, {.offset_bytes = (n_src_tiles - 1) * rows * f32_tile_bytes});
            noc.write_zeros_l1_barrier();
        }
        const uint32_t wbase = get_write_ptr(cb_stats_combine);
        const uint32_t half_block = s_w * half_record;
        for (uint32_t q = 0; q < group_size; ++q) {
            const uint32_t px = get_arg_val<uint32_t>(ports_idx + 2 * q);
            const uint32_t py = get_arg_val<uint32_t>(ports_idx + 2 * q + 1);
            const uint32_t slot = scratch_addr + q * slot_bytes;
            const uint32_t s_first = q * s_w;
            for (uint32_t j = 0; j < rows; ++j) {
                const uint32_t block = slot + (rec_base + (r0 + j) * s_w) * record_bytes;
                for (uint32_t sa = s_first; sa < s_first + s_w;) {
                    const uint32_t seg_end = (sa | 15) + 1;  // next 16-row face boundary
                    const uint32_t sb = seg_end < s_first + s_w ? seg_end : s_first + s_w;
                    const uint32_t r = sa & 31;
                    const uint32_t dst = wbase + ((sa >> 5) * rows + j) * f32_tile_bytes +
                                         (r >> 4) * 2 * f32_face_bytes + (r & 15) * f32_face_row_bytes;
                    const uint32_t src = block + (sa - s_first) * half_record;
                    const uint32_t bytes = (sb - sa) * half_record;
                    noc_async_read(get_noc_addr(px, py, src), dst, bytes);
                    noc_async_read(get_noc_addr(px, py, src + half_block), dst + f32_face_bytes, bytes);
                    sa = sb;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_stats_combine, n_tiles);
    }
}
