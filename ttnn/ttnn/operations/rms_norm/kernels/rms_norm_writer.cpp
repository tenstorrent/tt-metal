// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer: reads the RM gamma stick once, builds row-0 gamma tiles per W-chunk
// (gamma_depth chunks ahead of the drain), and drains output tiles to DRAM.
//
// Chunks are numbered globally across blocks so the gamma build for the next block overlaps
// the drain of the current one. Order: build(0), build(1), drain(0), build(2), drain(1), ...

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t FACE_WIDTH = 16;
constexpr uint32_t FACE_ELEMENTS = 16 * 16;

FORCE_INLINE void copy_words(uint32_t dst_addr, uint32_t src_addr, uint32_t num_words) {
    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    const auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    for (uint32_t word = 0; word < num_words; ++word) {
        dst[word] = src[word];
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_stick = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_row_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(5);
    constexpr bool has_gamma = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t output_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t gamma_elem_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t gamma_page_bytes = get_compile_time_arg_val(9);
    [[maybe_unused]] constexpr uint32_t gamma_stick_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t gamma_depth = get_compile_time_arg_val(11);
    constexpr auto output_args = TensorAccessorArgs<12>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);
    const uint32_t core_rows = get_arg_val<uint32_t>(3);

    const auto output_accessor = TensorAccessor(output_args, output_addr, output_tile_bytes);

    constexpr uint32_t chunks_per_block = (tensor_w_tiles + w_chunk_tiles - 1) / w_chunk_tiles;
    const uint32_t num_blocks = (core_rows + block_row_tiles - 1) / block_row_tiles;
    const uint32_t total_chunks = num_blocks * chunks_per_block;

    uint32_t gamma_stick_addr = 0;
    if constexpr (has_gamma) {
        static_assert(gamma_page_bytes <= gamma_stick_bytes);
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
        cb_reserve_back(cb_gamma_stick, 1);
        gamma_stick_addr = get_write_ptr(cb_gamma_stick);
        noc_async_read(gamma_accessor.get_noc_addr(0), gamma_stick_addr, gamma_page_bytes);
        noc_async_read_barrier();
    }

    // Row 0 of a tile is face-0 row 0 (cols 0..15) followed by face-1 row 0 (cols 16..31);
    // the row-broadcast multiply reads nothing else.
    auto build_gamma_chunk = [&](uint32_t global_chunk) {
        constexpr uint32_t gamma_tile_bytes = 4 * FACE_ELEMENTS * gamma_elem_bytes;
        constexpr uint32_t segment_bytes = FACE_WIDTH * gamma_elem_bytes;
        constexpr uint32_t face_bytes = FACE_ELEMENTS * gamma_elem_bytes;
        const uint32_t first_col = (global_chunk % chunks_per_block) * w_chunk_tiles;
        const uint32_t cols =
            (tensor_w_tiles - first_col) < w_chunk_tiles ? (tensor_w_tiles - first_col) : w_chunk_tiles;
        cb_reserve_back(cb_gamma_tiles, w_chunk_tiles);
        uint32_t dst_addr = get_write_ptr(cb_gamma_tiles);
        uint32_t src_addr = gamma_stick_addr + first_col * 2 * segment_bytes;
        for (uint32_t col = 0; col < cols; ++col) {
            copy_words(dst_addr, src_addr, segment_bytes / sizeof(uint32_t));
            copy_words(dst_addr + face_bytes, src_addr + segment_bytes, segment_bytes / sizeof(uint32_t));
            dst_addr += gamma_tile_bytes;
            src_addr += 2 * segment_bytes;
        }
        cb_push_back(cb_gamma_tiles, w_chunk_tiles);
    };

    if constexpr (has_gamma) {
        for (uint32_t chunk = 0; chunk < gamma_depth && chunk < total_chunks; ++chunk) {
            build_gamma_chunk(chunk);
        }
    }

    for (uint32_t global_chunk = 0; global_chunk < total_chunks; ++global_chunk) {
        const uint32_t block_row = (global_chunk / chunks_per_block) * block_row_tiles;
        const uint32_t first_col = (global_chunk % chunks_per_block) * w_chunk_tiles;
        const uint32_t rows = (core_rows - block_row) < block_row_tiles ? (core_rows - block_row) : block_row_tiles;
        const uint32_t cols =
            (tensor_w_tiles - first_col) < w_chunk_tiles ? (tensor_w_tiles - first_col) : w_chunk_tiles;

        // Compute packs the chunk row-major over (rows, cols).
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t first_tile = (row_start + block_row + row) * tensor_w_tiles + first_col;
            for (uint32_t col = 0; col < cols; ++col) {
                cb_wait_front(cb_output_tiles, 1);
                noc_async_write(
                    get_read_ptr(cb_output_tiles), output_accessor.get_noc_addr(first_tile + col), output_tile_bytes);
                noc_async_writes_flushed();
                cb_pop_front(cb_output_tiles, 1);
            }
        }

        if constexpr (has_gamma) {
            if (global_chunk + gamma_depth < total_chunks) {
                build_gamma_chunk(global_chunk + gamma_depth);
            }
        }
    }
    noc_async_write_barrier();
}
