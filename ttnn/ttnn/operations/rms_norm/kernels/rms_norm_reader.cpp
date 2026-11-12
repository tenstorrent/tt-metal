// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm reader (regime R1). Per block of `block_rows` tile-rows:
//   load_x_block — TILE: per row-chunk, <= c tile reads, one barrier, one push of c pages into cb_x.
//                  RM:   per row-chunk, <= 32 stick reads, one barrier, one push of c pages into cb_x_rm.
//   gamma        — per W chunk, one contiguous read of the gamma segment, relaid out into row 0 of
//                  each tile (backwards, so the in-place spread never overwrites unread source bytes).
// Once per kernel: the 0/1 reduce mask for a partial last W tile (prepare_reduce_mask).
//
// Raw NoC: the TILE x reads, the gamma chunk and the RM tail chunk of a K>1 row have no dataflow
// helper (read_sticks_for_tilize derives its L1 stride from row_bytes, which is wrong for a tail
// chunk narrower than c tiles — tilize_helpers_dataflow.inl:92-93).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_reduce_mask = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(5);

    constexpr uint32_t block_rows = get_compile_time_arg_val(9);
    constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t core_w_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t w_partial_elems = get_compile_time_arg_val(14);
    constexpr bool has_gamma = get_compile_time_arg_val(15) != 0;
    constexpr bool is_rm = get_compile_time_arg_val(16) != 0;
    constexpr uint32_t W = get_compile_time_arg_val(17);
    constexpr uint32_t num_valid_sticks = get_compile_time_arg_val(18);
    constexpr uint32_t x_elem = get_compile_time_arg_val(19);
    constexpr uint32_t g_elem = get_compile_time_arg_val(20);

    constexpr auto x_args = TensorAccessorArgs<21>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(2);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(3);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(4);

    constexpr uint32_t row_stride_tiles = num_w_chunks * w_chunk_tiles;
    constexpr uint32_t chunk_elems = w_chunk_tiles * 32;
    constexpr uint32_t x_tile_bytes = get_tile_size(cb_x);

    const auto x_acc = TensorAccessor(x_args, x_addr);

    // ---- once per kernel: reduce mask for the partial last W tile ----
    if constexpr (w_partial_elems != 0) {
        dataflow_kernel_lib::prepare_reduce_mask<cb_reduce_mask, ckernel::ReduceDim::REDUCE_ROW>(w_partial_elems);
        if constexpr (is_rm) {
            // The tail columns of a partial RM row are never written by the stick reads; zero the
            // whole staging ring once so the masked x^2 fold never multiplies stale NaN bits by 0.
            Noc noc;
            CircularBuffer x_rm_obj(cb_x_rm);
            noc.async_write_zeros(
                x_rm_obj, get_local_cb_interface(cb_x_rm).fifo_num_pages * get_tile_size(cb_x_rm), {.offset_bytes = 0});
            noc.write_zeros_l1_barrier();
        }
    }

    const uint32_t num_blocks = (num_tile_rows + block_rows - 1) / block_rows;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t first_row = block * block_rows;
        const uint32_t rows_this_block =
            (num_tile_rows - first_row < block_rows) ? (num_tile_rows - first_row) : block_rows;

        // ---- load_x_block ----
        for (uint32_t i = 0; i < rows_this_block; ++i) {
            const uint32_t tile_row = start_tile_row + first_row + i;
            if constexpr (!is_rm) {
                for (uint32_t k = 0; k < num_w_chunks; ++k) {
                    cb_reserve_back(cb_x, w_chunk_tiles);
                    uint32_t l1 = get_write_ptr(cb_x);
                    for (uint32_t j = 0; j < w_chunk_tiles; ++j) {
                        const uint32_t col = k * w_chunk_tiles + j;
                        if (col < core_w_tiles) {
                            noc_async_read_page(tile_row * tensor_w_tiles + w_tile_start + col, x_acc, l1);
                        }
                        l1 += x_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_x, w_chunk_tiles);
                }
            } else {
                const uint32_t first_stick = tile_row * 32;
                const uint32_t sticks = (num_valid_sticks - first_stick < 32) ? (num_valid_sticks - first_stick) : 32;
                for (uint32_t k = 0; k < num_w_chunks; ++k) {
                    const uint32_t elem_off = (w_tile_start * 32) + k * chunk_elems;
                    const uint32_t valid_elems = (W - elem_off < chunk_elems) ? (W - elem_off) : chunk_elems;
                    const uint32_t valid_bytes = valid_elems * x_elem;
                    if ((valid_elems + 31) / 32 == w_chunk_tiles) {
                        dataflow_kernel_lib::read_sticks_for_tilize<cb_x_rm>(
                            x_acc, sticks, valid_bytes, first_stick, elem_off * x_elem);
                    } else {
                        // Tail chunk narrower than c tiles: keep the nominal c*32-element L1 stride.
                        cb_reserve_back(cb_x_rm, w_chunk_tiles);
                        uint32_t l1 = get_write_ptr(cb_x_rm);
                        for (uint32_t s = 0; s < sticks; ++s) {
                            noc_async_read(x_acc.get_noc_addr(first_stick + s, elem_off * x_elem), l1, valid_bytes);
                            l1 += chunk_elems * x_elem;
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_x_rm, w_chunk_tiles);
                    }
                }
            }
        }

        // ---- gamma chunks (one pass per block) ----
        if constexpr (has_gamma) {
            const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
            constexpr uint32_t g_tile_bytes = get_tile_size(cb_gamma);
            constexpr uint32_t face_words = 16 * g_elem / 4;  // one 16-element face row, in 32-bit words
            constexpr uint32_t face1_offset = 16 * 16 * g_elem;
            for (uint32_t k = 0; k < num_w_chunks; ++k) {
                const uint32_t elem_off = (w_tile_start * 32) + k * chunk_elems;
                const uint32_t valid_elems = (W - elem_off < chunk_elems) ? (W - elem_off) : chunk_elems;
                cb_reserve_back(cb_gamma, w_chunk_tiles);
                const uint32_t base = get_write_ptr(cb_gamma);
                noc_async_read(g_acc.get_noc_addr(0, elem_off * g_elem), base, valid_elems * g_elem);
                noc_async_read_barrier();
                // Spread the contiguous segment into row 0 of each tile: elems 0-15 -> face 0 row 0,
                // elems 16-31 -> face 1 row 0. Backwards (last tile first) is overlap-safe.
                const uint32_t valid_tiles = (valid_elems + 31) / 32;
                for (uint32_t t = valid_tiles; t-- > 0;) {
                    volatile tt_l1_ptr uint32_t* src =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + t * 32 * g_elem);
                    volatile tt_l1_ptr uint32_t* dst =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + t * g_tile_bytes);
                    volatile tt_l1_ptr uint32_t* dst1 =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + t * g_tile_bytes + face1_offset);
                    for (uint32_t w = 0; w < face_words; ++w) {
                        dst1[w] = src[face_words + w];
                    }
                    if (t != 0) {
                        for (uint32_t w = 0; w < face_words; ++w) {
                            dst[w] = src[w];
                        }
                    }
                }
                cb_push_back(cb_gamma, w_chunk_tiles);
            }
        }
    }
}
