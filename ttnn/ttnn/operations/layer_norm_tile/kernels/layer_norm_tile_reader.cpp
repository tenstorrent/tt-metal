// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm reader: loads blocks of block_rows x block_w_tiles input tiles, and once (after block 0's
// reads, so the fetch hides behind block-0 compute) scatters row 0 of gamma/beta into resident CBs.
// Raw NoC: no dataflow helper covers page-granular tile reads or a face-row scatter (op_design.md).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t FACE_WIDTH = 16;

// Writes row 0 of each of block_w_tiles tiles: face-0 row 0 at page offset 0, face-1 row 0 at face_bytes.
// TILE source: one read of page j from offset 0 spanning face 0 and face-1 row 0 (read_bytes = face_bytes
// + one NoC granule); one request instead of two, since the grid-wide fetch is bound by request count on
// the few source banks. ROW_MAJOR source: two half_row_bytes reads at element 32*j of the flattened vector.
// Each core starts at its own column (first_col) so simultaneous fetches spread over the source banks.
template <
    bool is_tile,
    uint32_t stick_elems,
    uint32_t read_bytes,
    uint32_t face_bytes,
    uint32_t half_row_bytes,
    uint32_t block_w_tiles,
    uint32_t tile_bytes,
    typename Accessor>
FORCE_INLINE void issue_affine_reads(const Accessor& accessor, uint32_t dst_base, uint32_t first_col) {
    for (uint32_t step = 0; step < block_w_tiles; ++step) {
        const uint32_t col = (first_col + step) % block_w_tiles;
        const uint32_t dst_addr = dst_base + col * tile_bytes;
        if constexpr (is_tile) {
            static_assert(read_bytes > face_bytes && read_bytes <= tile_bytes);
            noc_async_read(accessor.get_noc_addr(col), dst_addr, read_bytes);
        } else {
            constexpr uint32_t elem_bytes = half_row_bytes / FACE_WIDTH;
            const uint32_t first_elem = TILE_WIDTH * col;
            const uint32_t page = first_elem / stick_elems;
            const uint32_t src_offset = (first_elem % stick_elems) * elem_bytes;
            noc_async_read(accessor.get_noc_addr(page, src_offset), dst_addr, read_bytes);
            noc_async_read(accessor.get_noc_addr(page, src_offset + half_row_bytes), dst_addr + face_bytes, read_bytes);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_rows = get_compile_time_arg_val(1);
    constexpr uint32_t cb_beta_rows = get_compile_time_arg_val(2);
    constexpr uint32_t block_rows = get_compile_time_arg_val(3);
    constexpr uint32_t block_w_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t input_tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t affine_face_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t affine_half_row_bytes = get_compile_time_arg_val(7);
    constexpr bool gamma_is_tile = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t gamma_stick_elems = get_compile_time_arg_val(9);
    constexpr uint32_t gamma_read_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t gamma_page_bytes = get_compile_time_arg_val(11);
    constexpr bool beta_is_tile = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t beta_stick_elems = get_compile_time_arg_val(13);
    constexpr uint32_t beta_read_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t beta_page_bytes = get_compile_time_arg_val(15);
    constexpr auto input_args = TensorAccessorArgs<16>();
    constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_row_tile = get_arg_val<uint32_t>(3);
    const uint32_t core_rows = get_arg_val<uint32_t>(4);
    const uint32_t num_blocks = get_arg_val<uint32_t>(5);
    const uint32_t affine_first_col = get_arg_val<uint32_t>(6);

    const auto input_accessor = TensorAccessor(input_args, input_addr, input_tile_bytes);
    const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr, beta_page_bytes);
    constexpr uint32_t affine_tile_bytes = 4 * affine_face_bytes;

    auto load_affine = [&]() {
        cb_reserve_back(cb_gamma_rows, block_w_tiles);
        cb_reserve_back(cb_beta_rows, block_w_tiles);
        issue_affine_reads<
            gamma_is_tile,
            gamma_stick_elems,
            gamma_read_bytes,
            affine_face_bytes,
            affine_half_row_bytes,
            block_w_tiles,
            affine_tile_bytes>(gamma_accessor, get_write_ptr(cb_gamma_rows), affine_first_col);
        issue_affine_reads<
            beta_is_tile,
            beta_stick_elems,
            beta_read_bytes,
            affine_face_bytes,
            affine_half_row_bytes,
            block_w_tiles,
            affine_tile_bytes>(beta_accessor, get_write_ptr(cb_beta_rows), affine_first_col);
        noc_async_read_barrier();
        cb_push_back(cb_gamma_rows, block_w_tiles);
        cb_push_back(cb_beta_rows, block_w_tiles);
    };

    // Ragged tile-rows (>= core_rows) are pushed at the nominal count without a read: compute runs
    // the full block, and the writer never stores those rows.
    for (uint32_t block = 0; block < num_blocks; ++block) {
        for (uint32_t row = 0; row < block_rows; ++row) {
            const uint32_t core_row = block * block_rows + row;
            cb_reserve_back(cb_input_tiles, block_w_tiles);
            if (core_row < core_rows) {
                uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
                const uint32_t first_tile = (start_row_tile + core_row) * block_w_tiles;
                for (uint32_t col = 0; col < block_w_tiles; ++col) {
                    noc_async_read(input_accessor.get_noc_addr(first_tile + col), l1_write_addr, input_tile_bytes);
                    l1_write_addr += input_tile_bytes;
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_tiles, block_w_tiles);
        }
        // Issued after block 0's reads so the fetch overlaps block-0 compute (needed at P6).
        if (block == 0) {
            load_affine();
        }
    }
}
