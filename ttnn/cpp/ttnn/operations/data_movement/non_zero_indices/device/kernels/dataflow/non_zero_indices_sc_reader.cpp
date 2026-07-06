// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr_0 = get_arg_val<uint32_t>(1);
    uint32_t output_addr_1 = get_arg_val<uint32_t>(2);
    uint32_t aligned_output_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb_index_0 = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb_index_1 = get_compile_time_arg_val(2);
    constexpr auto src0_args = TensorAccessorArgs<3>();
    constexpr auto dst0_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr auto dst1_args = TensorAccessorArgs<dst0_args.next_compile_time_args_offset()>();

    const auto out0 = TensorAccessor(dst0_args, output_addr_0);
    const auto out1 = TensorAccessor(dst1_args, output_addr_1, aligned_output_bytes);

    Noc noc;
    CircularBuffer input_cb(input_cb_index);
    CircularBuffer output_cb_0(output_cb_index_0);
    CircularBuffer output_cb_1(output_cb_index_1);

    // Reserve the full output_1 page up front — used as a write buffer for indices.
    output_cb_1.reserve_back(1);
    uint32_t indices_l1_addr = output_cb_1.get_write_ptr();
    volatile tt_l1_ptr uint32_t* indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_l1_addr);

    uint32_t num_non_zero = 0;

#ifndef INPUT_IS_TILE
    // ── ROW_MAJOR path ────────────────────────────────────────────────────────
    // Pages are rows (or batches of the last dimension). TensorAccessor handles
    // any underlying buffer layout (interleaved or sharded) transparently.
    uint32_t num_pages = get_arg_val<uint32_t>(4);
    // Physical elements per page; padded_shape==logical_shape is enforced by
    // validate_on_program_cache_miss, so this is safe to use as both the
    // iteration bound and the column-index stride.
    uint32_t last_dim = get_arg_val<uint32_t>(5);
    uint32_t aligned_page_size = get_arg_val<uint32_t>(6);
    uint32_t pages_per_bank = get_arg_val<uint32_t>(7);
    uint32_t grid_w = get_arg_val<uint32_t>(8);
    uint32_t logical_N = get_arg_val<uint32_t>(9);
    uint32_t logical_H = get_arg_val<uint32_t>(10);
    uint32_t grid_h = get_arg_val<uint32_t>(11);
    uint32_t is_col_major = get_arg_val<uint32_t>(12);

    // TensorAccessor computes a page's NOC address as: bank_base + bank_page_id * page_size_bytes.
    // For INTERLEAVED buffers the allocator pads each page to aligned_page_size, so the stride
    // between pages is aligned_page_size — pass that value.
    // For SHARDED L1 buffers the shards are packed with NO inter-page padding, so the stride
    // between pages equals the raw physical page size (last_dim * NUM_BYTES).  Passing
    // aligned_page_size here would overshoot into zero-padding and produce incorrect addresses
    // whenever phys_page_bytes < aligned_page_size (e.g. BLOCK_SHARDED with small shard width).
    constexpr bool input_is_sharded = src0_args.is_sharded;
    const uint32_t input_ta_page_size = input_is_sharded ? (last_dim * NUM_BYTES) : aligned_page_size;
    const auto s0 = TensorAccessor(src0_args, input_addr, input_ta_page_size);

    // Iterate in logical row-major order so that (b,n,h,c) index tuples are emitted in
    // strictly increasing order, matching torch.nonzero() output. Row r decomposes as:
    //   h = r % logical_H, n = (r / logical_H) % logical_N, b = r / (logical_H * logical_N)
    // For INTERLEAVED/HEIGHT_SHARDED (grid_w=1, pages_per_bank=1) this reduces to a simple
    // sequential page scan. For WIDTH/BLOCK_SHARDED (grid_w>1), each logical row r is split
    // across grid_w column shards whose TensorAccessor page_ids are interleaved in bank-major
    // order — the formula below reconstructs the correct page_id for each shard piece.
    const uint32_t total_rows = num_pages / grid_w;
    for (uint32_t r = 0; r < total_rows; ++r) {
        const uint32_t h = r % logical_H;
        const uint32_t n = (r / logical_H) % logical_N;
        const uint32_t b = r / (logical_H * logical_N);

        const uint32_t core_row_idx = r / pages_per_bank;
        const uint32_t bank_page_idx = r % pages_per_bank;
        for (uint32_t core_col = 0; core_col < grid_w; ++core_col) {
            // ROW_MAJOR: banks are ordered left-to-right across rows (row * grid_w + col).
            // COL_MAJOR: banks are ordered top-to-bottom across columns (col * grid_h + row).
            const uint32_t bank =
                is_col_major ? (core_col * grid_h + core_row_idx) : (core_row_idx * grid_w + core_col);
            const uint32_t page_id = bank * pages_per_bank + bank_page_idx;

            input_cb.reserve_back(1);
            noc.async_read(s0, input_cb, aligned_page_size, {.page_id = page_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            input_cb.push_back(1);

            uint32_t input_l1_addr = input_cb.get_read_ptr();
// nonzero_mask strips the IEEE sign bit for float dtypes so that -0.0
// (sign bit set, all others clear) compares equal to zero.  For integer
// dtypes IS_FLOAT is not defined and the mask is all-ones (no-op).
#if NUM_BYTES == 4
            volatile tt_l1_ptr uint32_t* input_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_l1_addr);
#ifdef IS_FLOAT
            constexpr uint32_t nonzero_mask = 0x7FFFFFFFu;
#else
            constexpr uint32_t nonzero_mask = 0xFFFFFFFFu;
#endif
#elif NUM_BYTES == 2
            volatile tt_l1_ptr uint16_t* input_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_l1_addr);
#ifdef IS_FLOAT
            constexpr uint16_t nonzero_mask = 0x7FFFu;
#else
            constexpr uint16_t nonzero_mask = 0xFFFFu;
#endif
#elif NUM_BYTES == 1
            volatile tt_l1_ptr uint8_t* input_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(input_l1_addr);
#ifdef IS_FLOAT
            constexpr uint8_t nonzero_mask = 0x7Fu;
#else
            constexpr uint8_t nonzero_mask = 0xFFu;
#endif
#endif
            for (uint32_t i = 0; i < last_dim; ++i) {
                if ((input_ptr[i] & nonzero_mask) != 0u) {
                    indices_ptr[num_non_zero * 4 + 0] = b;
                    indices_ptr[num_non_zero * 4 + 1] = n;
                    indices_ptr[num_non_zero * 4 + 2] = h;
                    indices_ptr[num_non_zero * 4 + 3] = core_col * last_dim + i;
                    ++num_non_zero;
                }
            }
            input_cb.pop_front(1);
        }
    }

#else  // INPUT_IS_TILE
    // ── TILE path ─────────────────────────────────────────────────────────────
    // Pages are 32×32 tiles. Elements within each tile are stored in face layout
    // (4 faces of 16×16, ordered: top-left, top-right, bottom-left, bottom-right).
    // We map each memory offset m → logical (tile_row, tile_col) and skip padding.
    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;
    constexpr uint32_t FACE_H = 16;
    constexpr uint32_t FACE_W = 16;
    constexpr uint32_t FACE_SIZE = FACE_H * FACE_W;  // 256 elements per face

    uint32_t B = get_arg_val<uint32_t>(4);
    uint32_t N = get_arg_val<uint32_t>(5);
    uint32_t logical_H = get_arg_val<uint32_t>(6);
    uint32_t logical_C = get_arg_val<uint32_t>(7);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(8);
    uint32_t num_tile_cols = get_arg_val<uint32_t>(9);
    uint32_t tile_page_size = get_arg_val<uint32_t>(10);

    const auto s0 = TensorAccessor(src0_args, input_addr, tile_page_size);

    const uint32_t tiles_per_slice = num_tile_rows * num_tile_cols;

    // Iterate in logical row-major order: for each tile row th, iterate each logical row within
    // it first, and for each logical row visit all tile columns tc.  This ensures (b,n,h,c) tuples
    // are emitted in strictly increasing order, matching torch.nonzero() output.  The trade-off is
    // that each tile is loaded once per logical row it contributes (up to TILE_H re-reads per tile).
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t th = 0; th < num_tile_rows; ++th) {
                const uint32_t abs_row_start = th * TILE_H;
                const uint32_t abs_row_end = (abs_row_start + TILE_H < logical_H) ? abs_row_start + TILE_H : logical_H;

                for (uint32_t tile_row = 0; tile_row < TILE_H; ++tile_row) {
                    const uint32_t abs_row = abs_row_start + tile_row;
                    if (abs_row >= abs_row_end) {
                        break;  // remaining rows in this tile are H-padding
                    }

                    for (uint32_t tc = 0; tc < num_tile_cols; ++tc) {
                        // Determine the valid col range within this tile column.
                        const uint32_t abs_col_start = tc * TILE_W;
                        const uint32_t abs_col_end =
                            (abs_col_start + TILE_W < logical_C) ? abs_col_start + TILE_W : logical_C;

                        const uint32_t tile_page_id =
                            b * (N * tiles_per_slice) + n * tiles_per_slice + th * num_tile_cols + tc;

                        input_cb.reserve_back(1);
                        noc.async_read(s0, input_cb, tile_page_size, {.page_id = tile_page_id}, {.offset_bytes = 0});
                        noc.async_read_barrier();
                        input_cb.push_back(1);

                        uint32_t tile_l1_addr = input_cb.get_read_ptr();
#if NUM_BYTES == 4
                        volatile tt_l1_ptr uint32_t* tile_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_l1_addr);
#ifdef IS_FLOAT
                        constexpr uint32_t nonzero_mask = 0x7FFFFFFFu;
#else
                        constexpr uint32_t nonzero_mask = 0xFFFFFFFFu;
#endif
#elif NUM_BYTES == 2
                        volatile tt_l1_ptr uint16_t* tile_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1_addr);
#ifdef IS_FLOAT
                        constexpr uint16_t nonzero_mask = 0x7FFFu;
#else
                        constexpr uint16_t nonzero_mask = 0xFFFFu;
#endif
#elif NUM_BYTES == 1
                        volatile tt_l1_ptr uint8_t* tile_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile_l1_addr);
#ifdef IS_FLOAT
                        constexpr uint8_t nonzero_mask = 0x7Fu;
#else
                        constexpr uint8_t nonzero_mask = 0xFFu;
#endif
#endif
                        // Map logical (tile_row, tile_col) → face-layout offset.
                        // face 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right.
                        for (uint32_t tile_col = 0; tile_col < TILE_W; ++tile_col) {
                            const uint32_t abs_col = abs_col_start + tile_col;
                            if (abs_col >= abs_col_end) {
                                break;  // remaining cols in this tile column are W-padding
                            }

                            // face_idx: bit 1 = row-half (0=top, 1=bottom), bit 0 = col-half (0=left, 1=right)
                            const uint32_t face_idx = ((tile_row >> 4) << 1) | (tile_col >> 4);
                            const uint32_t face_row = tile_row & (FACE_H - 1);
                            const uint32_t face_col = tile_col & (FACE_W - 1);
                            const uint32_t m = face_idx * FACE_SIZE + face_row * FACE_W + face_col;

                            if ((tile_ptr[m] & nonzero_mask) != 0u) {
                                indices_ptr[num_non_zero * 4 + 0] = b;
                                indices_ptr[num_non_zero * 4 + 1] = n;
                                indices_ptr[num_non_zero * 4 + 2] = abs_row;
                                indices_ptr[num_non_zero * 4 + 3] = abs_col;
                                ++num_non_zero;
                            }
                        }

                        input_cb.pop_front(1);
                    }
                }
            }
        }
    }
#endif  // INPUT_IS_TILE

    // ── Write count (output_0) ────────────────────────────────────────────────
    output_cb_0.reserve_back(1);
    uint32_t count_l1_addr = output_cb_0.get_write_ptr();
    volatile tt_l1_ptr uint32_t* count_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(count_l1_addr);
    count_ptr[0] = num_non_zero;
    noc.async_write(
        use<CircularBuffer::AddrSelector::WRITE_PTR>(output_cb_0), out0, 32, {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
    output_cb_0.push_back(1);

    // ── Write indices (output_1) ──────────────────────────────────────────────
    noc.async_write(
        use<CircularBuffer::AddrSelector::WRITE_PTR>(output_cb_1),
        out1,
        aligned_output_bytes,
        {.offset_bytes = 0},
        {.page_id = 0});
    noc.async_write_barrier();
    output_cb_1.push_back(1);
}
