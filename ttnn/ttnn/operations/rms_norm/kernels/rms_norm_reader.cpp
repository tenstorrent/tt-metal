// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm reader (NCRISC / NoC0).
//
// Once per program: load_gamma_slice — this member's W slice of gamma into cb_gamma, "expanded" so
// every row of tile j holds gamma[32j .. 32j+31] (lets pass 2 multiply with a plain DEST-reuse mul).
//   RM gamma  : one contiguous read of the slice into tile 0's area, RISC-V scatter of the 2 x 16-datum
//               row-0 pieces per tile (descending j, so the staging bytes are consumed before overwritten).
//   TILE gamma: per tile, face 0 row 0 and face 1 row 0 straight from DRAM (64 B reads; a bfloat8_b tile's
//               64 B exponent section rides along with the first read and is re-patterned so every face
//               row carries row 0's shared exponent).
//   Both      : rows 1..15 of faces 0/1 and faces 2/3 are replicated by local NoC copies (doubling, one
//               barrier per stage) — no RISC-V whole-tile store loop.
// Per block: load_x_block
//   TILE: one cb_reserve_back per block; per tile-row: core_w_tiles reads, one barrier, W-tail zero fill
//         (last W member only), one cb_push_back of x_row_pitch pages (the gap pages, if any, are never read).
//   RM  : per (tile-row, chunk) one reserve/barrier/push of rm_chunk_w_tiles stick-piece pages in cb_x_rm;
//         full chunks use read_sticks_for_tilize, the W-tail chunk is read raw + zero-filled to its tile edge.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
#include "rms_norm_common.hpp"

using namespace rms_norm_cb;

constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(0);
constexpr uint32_t W = get_compile_time_arg_val(1);
constexpr uint32_t H = get_compile_time_arg_val(2);
constexpr uint32_t image_row_tiles = get_compile_time_arg_val(3);
constexpr uint32_t x_row_pitch = get_compile_time_arg_val(4);
constexpr uint32_t rm_chunk_w_tiles = get_compile_time_arg_val(5);
constexpr uint32_t x_page_bytes = get_compile_time_arg_val(6);
constexpr uint32_t gamma_page_bytes = get_compile_time_arg_val(7);
constexpr uint32_t elem_bytes = get_compile_time_arg_val(8);
constexpr uint32_t gamma_elem_bytes = get_compile_time_arg_val(9);
constexpr bool gamma_is_tile = get_compile_time_arg_val(10) != 0;
constexpr uint32_t x_block_depth = get_compile_time_arg_val(11);
// Shared-exponent section ahead of the datums in one gamma tile: 64 B for bfloat8_b (one exponent byte per
// 16-datum face row, face-major), 0 for float32 / bfloat16. Datums (bfp8: mantissa bytes) follow in face order.
constexpr uint32_t gamma_exp_bytes = get_compile_time_arg_val(12);
constexpr auto x_args = TensorAccessorArgs<13>();
[[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t w_valid_in_last_tile = W - TILE_DIM * (tensor_w_tiles - 1);

// Zero columns [v, 32) of a tile in L1 (v = valid datums in the tile's row, 1..31).
// Whole faces go through the NoC zero engine; partial face rows (< 16 datums each) are RISC-V stores.
FORCE_INLINE void zero_w_tail_of_tile(const Noc& noc, const CircularBuffer& cb, uint32_t tile_offset_bytes) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * elem_bytes;
    constexpr uint32_t face_row_bytes = FACE_DIM * elem_bytes;
    constexpr uint32_t v = w_valid_in_last_tile;
    const uint32_t tile_addr = cb.get_write_ptr() + tile_offset_bytes;
    if constexpr (v <= FACE_DIM) {
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset_bytes + face_bytes});      // face 1
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset_bytes + 3 * face_bytes});  // face 3
        if constexpr (v < FACE_DIM) {
            for (uint32_t f = 0; f < 4; f += 2) {  // faces 0 and 2: cols v..15 of every row
                for (uint32_t row = 0; row < FACE_DIM; ++row) {
                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                        tile_addr + f * face_bytes + row * face_row_bytes + v * elem_bytes,
                        (FACE_DIM - v) * elem_bytes,
                        0);
                }
            }
        }
        noc.write_zeros_l1_barrier();
    } else {
        for (uint32_t f = 1; f < 4; f += 2) {  // faces 1 and 3: cols (v-16)..15 of every row
            for (uint32_t row = 0; row < FACE_DIM; ++row) {
                dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                    tile_addr + f * face_bytes + row * face_row_bytes + (v - FACE_DIM) * elem_bytes,
                    (TILE_DIM - v) * elem_bytes,
                    0);
            }
        }
    }
}

#ifdef RMS_HAS_GAMMA
// Replicate row 0 of faces 0/1 of `num_tiles` gamma tiles into every row of every face (local NoC copies).
constexpr uint32_t g_tile_bytes = gamma_exp_bytes + TILE_DIM * TILE_DIM * gamma_elem_bytes;

// Block-float gamma: every face row of tile j must carry row 0's shared exponent — faces 0/2 take face 0
// row 0's, faces 1/3 face 1 row 0's. 16 word stores per tile (the pattern, not a whole-tile fill).
FORCE_INLINE void expand_gamma_exponents(uint32_t base, uint32_t num_tiles) {
    static_assert(
        gamma_exp_bytes == 0 || gamma_exp_bytes == 4 * FACE_DIM, "bfp gamma: expected one exponent byte per face row");
    if constexpr (gamma_exp_bytes > 0) {
        constexpr uint32_t words_per_face = FACE_DIM / sizeof(uint32_t);
        for (uint32_t j = 0; j < num_tiles; ++j) {
            volatile tt_l1_ptr uint8_t* exp8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base + j * g_tile_bytes);
            volatile tt_l1_ptr uint32_t* exp32 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * g_tile_bytes);
            const uint32_t e_left = exp8[0] * 0x01010101u;          // face 0 row 0
            const uint32_t e_right = exp8[FACE_DIM] * 0x01010101u;  // face 1 row 0
            for (uint32_t w = 0; w < words_per_face; ++w) {
                exp32[0 * words_per_face + w] = e_left;
                exp32[1 * words_per_face + w] = e_right;
                exp32[2 * words_per_face + w] = e_left;
                exp32[3 * words_per_face + w] = e_right;
            }
        }
    }
}

FORCE_INLINE void expand_gamma_rows(uint32_t base, uint32_t num_tiles) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * gamma_elem_bytes;
    constexpr uint32_t face_row_bytes = FACE_DIM * gamma_elem_bytes;
    expand_gamma_exponents(base, num_tiles);
    for (uint32_t rows = 1; rows < FACE_DIM; rows <<= 1) {
        const uint32_t bytes = rows * face_row_bytes;
        for (uint32_t j = 0; j < num_tiles; ++j) {
            const uint32_t tile = base + j * g_tile_bytes + gamma_exp_bytes;
            noc_async_read(get_noc_addr(tile), tile + bytes, bytes);                            // face 0
            noc_async_read(get_noc_addr(tile + face_bytes), tile + face_bytes + bytes, bytes);  // face 1
        }
        noc_async_read_barrier();
    }
    for (uint32_t j = 0; j < num_tiles; ++j) {  // faces 0,1 -> faces 2,3 (rows 16..31)
        const uint32_t tile = base + j * g_tile_bytes + gamma_exp_bytes;
        noc_async_read(get_noc_addr(tile), tile + 2 * face_bytes, 2 * face_bytes);
    }
    noc_async_read_barrier();
}

template <typename Accessor>
FORCE_INLINE void load_gamma_slice(
    const Accessor& g_acc, uint32_t w_tile_start, uint32_t core_w_tiles, uint32_t gamma_push_tiles) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * gamma_elem_bytes;
    constexpr uint32_t row_read_bytes = 64;  // >= one 16-datum face row; BH DRAM read alignment
    cb_reserve_back(cb_gamma, gamma_push_tiles);
    const uint32_t base = get_write_ptr(cb_gamma);
    if constexpr (gamma_is_tile) {
        // Per tile, two 64 B-aligned reads: [exponent section (bfp only) + face 0 row 0], [face 1 row 0].
        constexpr uint32_t d0 = gamma_exp_bytes;
        for (uint32_t j = 0; j < core_w_tiles; ++j) {
            const uint32_t tile = base + j * g_tile_bytes;
            noc_async_read(g_acc.get_noc_addr(w_tile_start + j, 0), tile, d0 + row_read_bytes);
            noc_async_read(
                g_acc.get_noc_addr(w_tile_start + j, d0 + face_bytes), tile + d0 + face_bytes, row_read_bytes);
        }
        noc_async_read_barrier();
    } else {
        const uint32_t col0 = w_tile_start * TILE_DIM;
        const uint32_t col_end =
            (w_tile_start + core_w_tiles) * TILE_DIM < W ? (w_tile_start + core_w_tiles) * TILE_DIM : W;
        noc_async_read(g_acc.get_noc_addr(0, col0 * gamma_elem_bytes), base, (col_end - col0) * gamma_elem_bytes);
        noc_async_read_barrier();
        // Scatter row 0: datums [32j, 32j+16) -> tile j face 0 row 0, [32j+16, 32j+32) -> face 1 row 0.
        constexpr uint32_t words_per_face_row = FACE_DIM * gamma_elem_bytes / sizeof(uint32_t);
        for (uint32_t jj = core_w_tiles; jj > 0; --jj) {
            const uint32_t j = jj - 1;
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * TILE_DIM * gamma_elem_bytes);
            volatile tt_l1_ptr uint32_t* dst0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * g_tile_bytes);
            volatile tt_l1_ptr uint32_t* dst1 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * g_tile_bytes + face_bytes);
            uint32_t tmp[2 * words_per_face_row];
            for (uint32_t w = 0; w < 2 * words_per_face_row; ++w) {
                tmp[w] = src[w];
            }
            for (uint32_t w = 0; w < words_per_face_row; ++w) {
                dst0[w] = tmp[w];
                dst1[w] = tmp[words_per_face_row + w];
            }
        }
    }
    expand_gamma_rows(base, core_w_tiles);
    cb_push_back(cb_gamma, gamma_push_tiles);
}
#endif

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);  // first global tile-row of this core's group
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(4);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(5);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(6);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(7);
    [[maybe_unused]] const uint32_t chunks_this = get_arg_val<uint32_t>(8);

    const auto x_acc = TensorAccessor(x_args, x_addr, x_page_bytes);

#ifdef RMS_HAS_GAMMA
    {
        const auto g_acc = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
#ifdef RMS_RM
        const uint32_t gamma_push_tiles = chunks_this * rm_chunk_w_tiles;
#else
        const uint32_t gamma_push_tiles = core_w_tiles;
#endif
        load_gamma_slice(g_acc, w_tile_start, core_w_tiles, gamma_push_tiles);
    }
#endif

    // The member owning the global last W tile zero-fills its columns >= W (padding never enters sum x^2).
    [[maybe_unused]] const bool owns_w_tail =
        (w_valid_in_last_tile < TILE_DIM) && (w_tile_start + core_w_tiles == tensor_w_tiles);

#ifdef RMS_RM
    constexpr uint32_t chunk_row_bytes = rm_chunk_w_tiles * TILE_DIM * elem_bytes;  // L1 stride of one stick
    constexpr uint32_t tile_row_bytes = TILE_DIM * elem_bytes;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t rows = (blk + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t t = row_start + blk * block_row_tiles + r;
            const uint32_t image = t / image_row_tiles;
            const uint32_t tr = t - image * image_row_tiles;
            const uint32_t first_stick = image * H + tr * TILE_DIM;
            const uint32_t valid_rows = (H - tr * TILE_DIM) < TILE_DIM ? (H - tr * TILE_DIM) : TILE_DIM;
            for (uint32_t c = 0; c < chunks_this; ++c) {
                const uint32_t chunk_tiles = (core_w_tiles - c * rm_chunk_w_tiles) < rm_chunk_w_tiles
                                                 ? (core_w_tiles - c * rm_chunk_w_tiles)
                                                 : rm_chunk_w_tiles;
                const uint32_t col0 = (w_tile_start + c * rm_chunk_w_tiles) * TILE_DIM;
                const uint32_t col_end_tiles = (w_tile_start + c * rm_chunk_w_tiles + chunk_tiles) * TILE_DIM;
                const uint32_t col_end = col_end_tiles < W ? col_end_tiles : W;
                const uint32_t bytes = (col_end - col0) * elem_bytes;
                if (bytes == chunk_row_bytes) {
                    dataflow_kernel_lib::read_sticks_for_tilize<cb_x_rm>(
                        x_acc, valid_rows, bytes, first_stick, col0 * elem_bytes);
                } else {
                    // Ragged chunk (pad tiles and/or W tail): raw read; zero to the next tile edge.
                    cb_reserve_back(cb_x_rm, rm_chunk_w_tiles);
                    const uint32_t l1 = get_write_ptr(cb_x_rm);
                    for (uint32_t s = 0; s < valid_rows; ++s) {
                        noc_async_read(
                            x_acc.get_noc_addr(first_stick + s, col0 * elem_bytes), l1 + s * chunk_row_bytes, bytes);
                    }
                    const uint32_t tail_end = ((bytes + tile_row_bytes - 1) / tile_row_bytes) * tile_row_bytes;
                    if (tail_end > bytes) {
                        for (uint32_t s = 0; s < valid_rows; ++s) {
                            dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                l1 + s * chunk_row_bytes + bytes, tail_end - bytes, 0);
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_x_rm, rm_chunk_w_tiles);
                }
            }
        }
    }
#else
    constexpr uint32_t tile_bytes = x_page_bytes;
    Noc noc;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t rows = (blk + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
        const uint32_t cb_id = (x_block_depth == 2 && (blk & 1)) ? cb_x_alt : cb_x;
        CircularBuffer cb(cb_id);
        cb_reserve_back(cb_id, rows * x_row_pitch);
        const uint32_t base = get_write_ptr(cb_id);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t t = row_start + blk * block_row_tiles + r;
            const uint32_t first_tile = t * tensor_w_tiles + w_tile_start;
            const uint32_t row_off = r * x_row_pitch * tile_bytes;
            for (uint32_t j = 0; j < core_w_tiles; ++j) {
                noc_async_read(x_acc.get_noc_addr(first_tile + j), base + row_off + j * tile_bytes, tile_bytes);
            }
            noc_async_read_barrier();
            if (owns_w_tail) {
                // Offset relative to the CB's CURRENT write pointer, which has advanced by the r rows pushed so far.
                const uint32_t tail_tile_addr = base + row_off + (core_w_tiles - 1) * tile_bytes;
                zero_w_tail_of_tile(noc, cb, tail_tile_addr - cb.get_write_ptr());
            }
            cb_push_back(cb_id, x_row_pitch);
        }
    }
#endif
}
