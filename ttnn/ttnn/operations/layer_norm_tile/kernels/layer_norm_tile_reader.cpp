// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm reader (RISCV_1, NoC0).
//   load_block   : R_b x Wt tiles of x (and residual) in C-tile chunks, one barrier per chunk.
//   stage_affine : once, behind block 0's first chunk (shared barrier). Only row 0 of each gamma/beta
//                  tile is staged: face-0 row 0 at slot offset 0, face-1 row 0 at slot offset FACE_BYTES.
//                  Every DRAM read is 64 B at a 64 B-aligned offset (BH DRAM alignment).
//
// The RM upper-half move is a self-aimed 32 B NoC read with a per-tile source address; the
// local_copy_helpers stateful read pins one source address, so it does not fit this loop.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block_row_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t io_chunk_tiles = get_compile_time_arg_val(6);
    constexpr bool has_residual = get_compile_time_arg_val(7) != 0;
    constexpr bool affine_is_tile = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t affine_page_elems = get_compile_time_arg_val(9);
    constexpr uint32_t input_tile_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t affine_page_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t affine_elem_bytes = get_compile_time_arg_val(12);
    constexpr auto input_args = TensorAccessorArgs<13>();
    [[maybe_unused]] constexpr auto residual_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(2);
    const uint32_t beta_addr = get_arg_val<uint32_t>(3);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(4);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(5);

    static_assert(tensor_w_tiles % io_chunk_tiles == 0, "every block push must be a whole number of chunks");

    constexpr uint32_t TILE_DIM = 32;
    constexpr uint32_t FACE_DIM = 16;
    constexpr uint32_t affine_row_bytes = TILE_DIM * affine_elem_bytes;  // one 32-element tile row, 64 B
    constexpr uint32_t face_row_bytes = FACE_DIM * affine_elem_bytes;    // 32 B
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * affine_elem_bytes;
    constexpr uint32_t affine_tile_bytes = 4 * face_bytes;
    static_assert(affine_is_tile || affine_page_elems % TILE_DIM == 0, "RM affine stick must hold whole tile rows");

    DataflowBuffer input_tiles(cb_input_tiles);
    [[maybe_unused]] DataflowBuffer residual_tiles(cb_residual_tiles);
    DataflowBuffer gamma_tiles(cb_gamma_tiles);
    DataflowBuffer beta_tiles(cb_beta_tiles);

    const auto input_accessor = TensorAccessor(input_args, input_addr, input_tile_bytes);
    [[maybe_unused]] const auto residual_accessor = TensorAccessor(residual_args, residual_addr, input_tile_bytes);
    const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, affine_page_bytes);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr, affine_page_bytes);

    auto issue_affine_reads = [&](const auto& accessor, DataflowBuffer& dst) {
        dst.reserve_back(tensor_w_tiles);
        uint32_t slot_addr = dst.get_write_ptr();
        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
            if constexpr (affine_is_tile) {
                noc_async_read(accessor.get_noc_addr(col, 0), slot_addr, affine_row_bytes);
                noc_async_read(accessor.get_noc_addr(col, face_bytes), slot_addr + face_bytes, affine_row_bytes);
            } else {
                const uint32_t elem = col * TILE_DIM;
                noc_async_read(
                    accessor.get_noc_addr(elem / affine_page_elems, (elem % affine_page_elems) * affine_elem_bytes),
                    slot_addr,
                    affine_row_bytes);
            }
            slot_addr += affine_tile_bytes;
        }
    };

    // RM only: elements 16..31 landed in face-0 row 1; move them to face-1 row 0.
    auto finish_affine = [&](DataflowBuffer& dst) {
        if constexpr (!affine_is_tile) {
            uint32_t slot_addr = dst.get_write_ptr();
            for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
                noc_async_read(get_noc_addr(slot_addr + face_row_bytes), slot_addr + face_bytes, face_row_bytes);
                slot_addr += affine_tile_bytes;
            }
        }
    };

    bool affine_pending = true;
    for (uint32_t block_row = 0; block_row < core_row_tiles; block_row += block_row_tiles) {
        const uint32_t rows =
            (core_row_tiles - block_row) < block_row_tiles ? (core_row_tiles - block_row) : block_row_tiles;
        const uint32_t first_page = (row_tile_start + block_row) * tensor_w_tiles;
        const uint32_t block_tiles = rows * tensor_w_tiles;

        for (uint32_t chunk_start = 0; chunk_start < block_tiles; chunk_start += io_chunk_tiles) {
            const uint32_t chunk_page = first_page + chunk_start;
            input_tiles.reserve_back(io_chunk_tiles);
            uint32_t input_l1 = input_tiles.get_write_ptr();
            for (uint32_t tile = 0; tile < io_chunk_tiles; ++tile) {
                noc_async_read(input_accessor.get_noc_addr(chunk_page + tile), input_l1, input_tile_bytes);
                input_l1 += input_tile_bytes;
            }
            if constexpr (has_residual) {
                residual_tiles.reserve_back(io_chunk_tiles);
                uint32_t residual_l1 = residual_tiles.get_write_ptr();
                for (uint32_t tile = 0; tile < io_chunk_tiles; ++tile) {
                    noc_async_read(residual_accessor.get_noc_addr(chunk_page + tile), residual_l1, input_tile_bytes);
                    residual_l1 += input_tile_bytes;
                }
            }
            if (affine_pending) {
                issue_affine_reads(gamma_accessor, gamma_tiles);
                issue_affine_reads(beta_accessor, beta_tiles);
            }
            noc_async_read_barrier();
            input_tiles.push_back(io_chunk_tiles);
            if constexpr (has_residual) {
                residual_tiles.push_back(io_chunk_tiles);
            }
            if (affine_pending) {
                if constexpr (!affine_is_tile) {
                    finish_affine(gamma_tiles);
                    finish_affine(beta_tiles);
                    noc_async_read_barrier();
                }
                gamma_tiles.push_back(tensor_w_tiles);
                beta_tiles.push_back(tensor_w_tiles);
                affine_pending = false;
            }
        }
    }
}
