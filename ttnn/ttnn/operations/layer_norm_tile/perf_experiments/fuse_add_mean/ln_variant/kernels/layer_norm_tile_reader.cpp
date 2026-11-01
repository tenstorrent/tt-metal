// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm reader (RISCV_1, NoC0).
//   load_block   : R_b x Wt tiles of x (and residual, unless the writer RISC streams it) in C-tile
//                  chunks on a trid ring of `read_trid_depth` chunks (layer_norm_tile_stream.hpp).
//   stage_affine : once, issued untagged behind block 0's last chunk. Only row 0 of each gamma/beta
//                  tile is staged: face-0 row 0 at slot offset 0, face-1 row 0 at slot offset FACE_BYTES.
//                  Every DRAM read is 64 B at a 64 B-aligned offset (BH DRAM alignment).
//
// The RM upper-half move is a self-aimed 32 B NoC read with a per-tile source address; the
// local_copy_helpers stateful read pins one source address, so it does not fit this loop.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "layer_norm_tile_stream.hpp"

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
    constexpr uint32_t read_trid_depth = get_compile_time_arg_val(13);
    constexpr bool residual_on_writer = get_compile_time_arg_val(14) != 0;
    constexpr bool preload = get_compile_time_arg_val(15) != 0;
    constexpr auto input_args = TensorAccessorArgs<16>();
    [[maybe_unused]] constexpr auto residual_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(2);
    const uint32_t beta_addr = get_arg_val<uint32_t>(3);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(4);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(5);
    const uint32_t rotation = get_arg_val<uint32_t>(6);

    constexpr bool read_residual = has_residual && !residual_on_writer;

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

    const Noc noc;
    bool affine_pending = true;
    // PRELOAD (compute-only isolation): one reserve for the whole core, every push after the last read.
    uint32_t preload_input_base = 0;
    uint32_t preload_residual_base = 0;
    if constexpr (preload) {
        input_tiles.reserve_back(core_row_tiles * tensor_w_tiles);
        preload_input_base = input_tiles.get_write_ptr();
        if constexpr (read_residual) {
            residual_tiles.reserve_back(core_row_tiles * tensor_w_tiles);
            preload_residual_base = residual_tiles.get_write_ptr();
        }
    }
    for (uint32_t block_row = 0; block_row < core_row_tiles; block_row += block_row_tiles) {
        const uint32_t rows =
            (core_row_tiles - block_row) < block_row_tiles ? (core_row_tiles - block_row) : block_row_tiles;
        const uint32_t first_page = (row_tile_start + block_row) * tensor_w_tiles;
        const uint32_t num_chunks = rows * tensor_w_tiles / io_chunk_tiles;

        // Whole-block reserve: the CB holds a whole number of blocks, so the block is contiguous.
        uint32_t input_base = 0;
        uint32_t residual_base = 0;
        {
            MaybeDeviceZoneScope("reader_reserve");
            if constexpr (preload) {
                input_base = preload_input_base + block_row * tensor_w_tiles * input_tile_bytes;
                residual_base = preload_residual_base + block_row * tensor_w_tiles * input_tile_bytes;
            } else {
                input_tiles.reserve_back(rows * tensor_w_tiles);
                input_base = input_tiles.get_write_ptr();
                if constexpr (read_residual) {
                    residual_tiles.reserve_back(rows * tensor_w_tiles);
                    residual_base = residual_tiles.get_write_ptr();
                }
            }
        }

        auto issue = [&](uint32_t chunk) {
            const uint32_t page = first_page + chunk * io_chunk_tiles;
            const uint32_t offset = chunk * io_chunk_tiles * input_tile_bytes;
            layer_norm_tile::read_chunk<io_chunk_tiles, input_tile_bytes>(
                input_accessor, page, input_base + offset, rotation);
            if constexpr (read_residual) {
                layer_norm_tile::read_chunk<io_chunk_tiles, input_tile_bytes>(
                    residual_accessor, page, residual_base + offset, rotation);
            }
            if (affine_pending && chunk + 1 == num_chunks) {
                dataflow_kernel_lib::set_read_trid(noc, 0);
                issue_affine_reads(gamma_accessor, gamma_tiles);
                issue_affine_reads(beta_accessor, beta_tiles);
            }
        };
        auto retire = [&]() {
            if constexpr (preload) {
                return;
            }
            input_tiles.push_back(io_chunk_tiles);
            if constexpr (read_residual) {
                residual_tiles.push_back(io_chunk_tiles);
            }
        };
        {
            MaybeDeviceZoneScope("reader_block");
            layer_norm_tile::read_block_chunks<read_trid_depth>(noc, num_chunks, issue, retire);
        }

        if (affine_pending) {
            MaybeDeviceZoneScope("reader_affine");
            noc_async_read_barrier();
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
    if constexpr (preload) {
        input_tiles.push_back(core_row_tiles * tensor_w_tiles);
        if constexpr (read_residual) {
            residual_tiles.push_back(core_row_tiles * tensor_w_tiles);
        }
    }
}
