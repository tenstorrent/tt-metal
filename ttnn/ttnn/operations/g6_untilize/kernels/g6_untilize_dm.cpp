// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// untilize_dm: the op's only kernel, one data-movement RISC per core, no compute kernel.
//
// The TILE -> ROW_MAJOR reorder is done by the NoC: each 64 B face-row piece of an input tile
// page is read straight into its row position in an L1 slot (gather_block), then whole output
// pages (or page segments) are written from the slot (store_block). Bytes never pass through
// an unpacker, DEST or packer, so every uint32 bit pattern survives.
//
// No kernel_lib helper is used: the compute-side untilize helper routes UInt32 through DEST and
// confines the reorder to one tile row per core, and the dataflow helpers address local L1 or
// read RM sticks, not face-row pieces of DRAM tile pages (op_design.md, API Mapping).
//
// cb_row_staging is an L1 allocation only: producer and consumer are this thread, ordered by the
// read barrier (gather before store) and the write flush (store before slot reuse).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t in_page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t out_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t rows_per_image = get_compile_time_arg_val(3);
    constexpr uint32_t tile_rows_per_image = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(5);
    constexpr uint32_t tensor_rows = get_compile_time_arg_val(6);
    constexpr uint32_t block_rows = get_compile_time_arg_val(7);
    constexpr uint32_t block_tile_cols = get_compile_time_arg_val(8);
    constexpr uint32_t col_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t total_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t num_cores = get_compile_time_arg_val(11);
    constexpr uint32_t cores_x = get_compile_time_arg_val(12);
    constexpr uint32_t origin_x = get_compile_time_arg_val(13);
    constexpr uint32_t origin_y = get_compile_time_arg_val(14);
    constexpr uint32_t scratch_slots = get_compile_time_arg_val(15);
    constexpr uint32_t tile_height = get_compile_time_arg_val(16);
    constexpr uint32_t face_height = get_compile_time_arg_val(17);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(19);
    constexpr uint32_t tile_row_bytes = get_compile_time_arg_val(20);
    constexpr uint32_t cb_row_staging = get_compile_time_arg_val(21);

    constexpr auto in_args = TensorAccessorArgs<22>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    constexpr uint32_t block_row_bytes = block_tile_cols * tile_row_bytes;
    constexpr uint32_t slot_bytes = block_rows * block_row_bytes;
    constexpr uint32_t blocks_per_core = total_blocks / num_cores;
    constexpr uint32_t blocks_remainder = total_blocks % num_cores;

    // A wider read would pull the next face row into this row's columns (op_design.md, Mechanism caps).
    static_assert(face_row_bytes <= NOC_MAX_BURST_SIZE, "a face-row piece must be one NoC packet");
    static_assert(scratch_slots >= 1, "the staging ring needs at least one slot");

    const auto x = TensorAccessor(in_args, get_common_arg_val<uint32_t>(0), in_page_bytes);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), out_page_bytes);

    const uint32_t core_idx = (static_cast<uint32_t>(get_relative_logical_y()) - origin_y) * cores_x +
                              (static_cast<uint32_t>(get_relative_logical_x()) - origin_x);
    const uint32_t first_block =
        core_idx * blocks_per_core + (core_idx < blocks_remainder ? core_idx : blocks_remainder);
    const uint32_t num_blocks_this_core = blocks_per_core + (core_idx < blocks_remainder ? 1 : 0);

    const uint32_t staging_base = get_write_ptr(cb_row_staging);

    // DRAM tile pages -> slot rows, row order. All reads of the block in flight, one barrier.
    auto gather_block = [&](uint32_t slot_addr, uint32_t first_row, uint32_t rows, uint32_t c0, uint32_t cols) {
        {
            DeviceZoneScopedN("G6U-GATHER-ISSUE");
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t p = first_row + row;
                const uint32_t image = p / rows_per_image;
                const uint32_t r = p - image * rows_per_image;
                const uint32_t tile_row = image * tile_rows_per_image + r / tile_height;
                const uint32_t r_in_tile = r % tile_height;
                // Row r_in_tile is face row (r_in_tile % face_height) of faces 2h (left) and 2h+1 (right).
                const uint32_t piece_offset =
                    2 * (r_in_tile / face_height) * face_bytes + (r_in_tile % face_height) * face_row_bytes;
                uint32_t page = tile_row * tensor_tile_cols + c0;
                uint32_t dst = slot_addr + row * block_row_bytes;
                for (uint32_t c = 0; c < cols; ++c, ++page, dst += tile_row_bytes) {
                    const uint64_t src = x.get_noc_addr(page, piece_offset);
                    noc_async_read_one_packet(src, dst, face_row_bytes);
                    noc_async_read_one_packet(src + face_bytes, dst + face_row_bytes, face_row_bytes);
                }
            }
        }
        {
            DeviceZoneScopedN("G6U-GATHER-WAIT");
            noc_async_read_barrier();
        }
    };

    // Slot rows -> output pages: one write per row, clipped to the page when W is not a multiple of 32.
    auto store_block = [&](uint32_t slot_addr, uint32_t first_row, uint32_t rows, uint32_t c0, uint32_t cols) {
        DeviceZoneScopedN("G6U-STORE-ISSUE");
        const uint32_t segment_offset = c0 * tile_row_bytes;
        const uint32_t staged_bytes = cols * tile_row_bytes;
        const uint32_t page_left = out_row_bytes - segment_offset;
        const uint32_t segment_bytes = staged_bytes < page_left ? staged_bytes : page_left;
        for (uint32_t row = 0; row < rows; ++row) {
            noc_async_write<block_row_bytes>(
                slot_addr + row * block_row_bytes, out.get_noc_addr(first_row + row, segment_offset), segment_bytes);
        }
    };

    // Waits until every write issued so far has left L1, so its slot can be refilled.
    auto wait_slot_free = [&]() {
        DeviceZoneScopedN("G6U-SLOT-WAIT");
        noc_async_writes_flushed();
    };

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t block = first_block + block_idx;
        const uint32_t row_block = block / col_blocks;
        const uint32_t col_block = block - row_block * col_blocks;
        const uint32_t first_row = row_block * block_rows;
        const uint32_t rows = (tensor_rows - first_row) < block_rows ? (tensor_rows - first_row) : block_rows;
        const uint32_t c0 = col_block * block_tile_cols;
        const uint32_t cols = (tensor_tile_cols - c0) < block_tile_cols ? (tensor_tile_cols - c0) : block_tile_cols;
        const uint32_t slot_addr = staging_base + (block_idx % scratch_slots) * slot_bytes;

        // One slot: its previous block's writes must leave before the gather overwrites it.
        // Two or more slots: flushing just before each store keeps block k-1's writes departing
        // while block k's reads are in flight, and still frees slot (k+1) % slots in time.
        if constexpr (scratch_slots == 1) {
            if (block_idx > 0) {
                wait_slot_free();
            }
        }
        gather_block(slot_addr, first_row, rows, c0, cols);
        if constexpr (scratch_slots > 1) {
            if (block_idx > 0) {
                wait_slot_free();
            }
        }
        store_block(slot_addr, first_row, rows, c0, cols);
    }

    // Every write acknowledged before return: the next program in the model starts right after.
    {
        DeviceZoneScopedN("G6U-DRAIN");
        noc_async_write_barrier();
    }
}
