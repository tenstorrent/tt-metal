// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for per_token_cast_back step 2 (grouped scale [M, H/128]).
//
// Per tile-row, read all 32 tokens' full scale rows ONCE into a reader-private scratch, each into a
// slot strided by the scale tensor's aligned DRAM page size. Matching that stride keeps the L1
// destination congruent (mod 64) with the 64-aligned DRAM page, satisfying the NOC alignment
// constraint (a 32-byte-strided scratch fails: odd rows land at L1 offset 32 vs a 64-aligned source).
//
// Then per (tile-row, col-block):
//   - e4m3 col-block slice -> cb_e4m3 (32 one-tile pages), column-block-major.
//   - build GROUPS_PER_BLOCK bcast operand tiles in cb_scale_bcast: tile g has column 0 =
//     scale[:, c*groups_per_block + g] (face-aware); rest is don't-care (bcast_cols reads col 0).
// The per-group column selection is the "shift" moved into the reader (no column-shift LLK; §6).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

template <
    uint32_t face_h,
    uint32_t face_w,
    uint32_t FACE_W_BYTES,
    uint32_t FACE_ROWS,
    uint32_t FACE_ROW_STRIDE_BYTES,
    uint32_t scale_aligned_page_bytes>
static inline void append_first_column_to_tile(uint32_t scratch, uint32_t page, uint32_t global_group) {
    // tile column 0 walked face-row by face-row: col0_off is purely additive (no div/mod).
    uint32_t s = 0;
    uint32_t face_base_off = 0;  // = face_row * FACE_ROW_STRIDE_BYTES
    for (uint32_t fr = 0; fr < FACE_ROWS; ++fr) {
        uint32_t col0_off = face_base_off;  // in-face row 0, col 0
        for (uint32_t r = 0; r < face_h; ++r) {
            uint32_t val = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                scratch + s * scale_aligned_page_bytes + global_group * 4);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page + col0_off) = val;
            col0_off += FACE_W_BYTES;
            ++s;
        }
        face_base_off += FACE_ROW_STRIDE_BYTES;
    }
}

void kernel_main() {
    uint32_t e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t scale_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(2);
    uint32_t num_col_blocks = get_arg_val<uint32_t>(3);
    uint32_t start_tile_row = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_e4m3 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scale_bcast = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scale_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t e4m3_col_block_bytes = get_compile_time_arg_val(3);      // 1024
    constexpr uint32_t groups_per_block = get_compile_time_arg_val(4);          // 8
    constexpr uint32_t scale_aligned_page_bytes = get_compile_time_arg_val(5);  // aligned row footprint
    // Tile / face dims from the tensor's tile spec (32x32 / 16x16 by default; tiny tiles supported).
    constexpr uint32_t tile_h = get_compile_time_arg_val(6);
    constexpr uint32_t tile_w = get_compile_time_arg_val(7);
    constexpr uint32_t face_h = get_compile_time_arg_val(8);
    constexpr uint32_t face_w = get_compile_time_arg_val(9);
    constexpr uint32_t face_elems = face_h * face_w;     // fp32 per face
    constexpr uint32_t faces_per_row = tile_w / face_w;  // face columns per tile
    constexpr uint32_t TILE_BYTES = tile_h * tile_w * sizeof(float);
    constexpr uint32_t FACE_ROWS = tile_h / face_h;                                         // face rows per tile
    constexpr uint32_t FACE_ROW_STRIDE_BYTES = faces_per_row * face_elems * sizeof(float);  // per face row
    constexpr uint32_t FACE_W_BYTES = face_w * sizeof(float);                               // per in-face row

    constexpr auto e4m3_args = TensorAccessorArgs<10>();
    constexpr auto scale_args = TensorAccessorArgs<e4m3_args.next_compile_time_args_offset()>();
    const auto e4m3 = TensorAccessor(e4m3_args, e4m3_addr);
    const auto scale = TensorAccessor(scale_args, scale_addr);

    // Reader-private scratch: 32 rows x scale_aligned_page_bytes (reused every tile-row).
    cb_reserve_back(cb_scale_scratch, 1);
    uint32_t scratch = get_write_ptr(cb_scale_scratch);

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        // --- read all 32 tokens' full scale rows once, aligned ---
        for (uint32_t s = 0; s < tile_h; ++s) {
            uint32_t page_id = (start_tile_row + tr) * tile_h + s;
            noc_async_read(
                scale.get_noc_addr(page_id), scratch + s * scale_aligned_page_bytes, scale_aligned_page_bytes);
        }
        noc_async_read_barrier();

        for (uint32_t c = 0; c < num_col_blocks; ++c) {
            // --- e4m3 col-block (32 one-tile pages) ---
            cb_reserve_back(cb_e4m3, tile_h);
            uint32_t e4m3_l1 = get_write_ptr(cb_e4m3);
            uint32_t e4m3_off = c * e4m3_col_block_bytes;
            for (uint32_t s = 0; s < tile_h; ++s) {
                uint32_t page_id = (start_tile_row + tr) * tile_h + s;
                noc_async_read(e4m3.get_noc_addr(page_id) + e4m3_off, e4m3_l1, e4m3_col_block_bytes);
                e4m3_l1 += e4m3_col_block_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_e4m3, tile_h);

            // --- build per-group bcast operands: tile g, column 0 = scale[:, c*groups + g] ---
            cb_reserve_back(cb_scale_bcast, groups_per_block);
            uint32_t bcast_base = get_write_ptr(cb_scale_bcast);
            for (uint32_t g = 0; g < groups_per_block; ++g) {
                uint32_t page = bcast_base + g * TILE_BYTES;
                uint32_t global_group = c * groups_per_block + g;
                append_first_column_to_tile<
                    face_h,
                    face_w,
                    FACE_W_BYTES,
                    FACE_ROWS,
                    FACE_ROW_STRIDE_BYTES,
                    scale_aligned_page_bytes>(scratch, page, global_group);
            }
            cb_push_back(cb_scale_bcast, groups_per_block);
        }
    }
}
