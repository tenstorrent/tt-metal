// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// untilize_dm: the op's only kernel (row_gather regime, op_design.md).
//
// Each core gathers the 64 B face-row pieces of its output rows from the input's
// TILE pages straight into row order in an L1 slot (the reorder is done by where
// each NoC read lands), then writes whole output pages. Bytes only move through
// NoC reads and writes, so every 32-bit pattern is preserved.
//
// No kernel_lib helper is used: the helpers that touch untilize are compute-side
// (unit = a tile row through DEST, which cannot carry UInt32 losslessly through the
// packer path) or L1->L1 / RM-stick readers. None reads a face-row piece of a DRAM
// tile page into a row slot (op_design.md, "Helpers considered and rejected").
//
// cb_row_staging is an L1 allocation only: producer and consumer are this one
// thread, ordered by the read barrier (gather before store) and the write flush
// (store departed before the slot is re-gathered). No reserve/push/wait/pop.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Tensor geometry (logical rows only; batch is folded into the row axis).
    constexpr uint32_t tensor_rows = get_compile_time_arg_val(0);
    constexpr uint32_t rows_per_image = get_compile_time_arg_val(1);
    constexpr uint32_t tile_rows_per_image = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t element_bytes = get_compile_time_arg_val(5);
    // Blocking knobs and the work split (op_design.md, Blocking Model / Work Distribution).
    constexpr uint32_t block_rows = get_compile_time_arg_val(6);
    constexpr uint32_t block_tile_cols = get_compile_time_arg_val(7);
    constexpr uint32_t block_row_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t col_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t total_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t num_cores = get_compile_time_arg_val(11);
    constexpr uint32_t cores_x = get_compile_time_arg_val(12);
    constexpr uint32_t core_origin_x = get_compile_time_arg_val(13);
    constexpr uint32_t core_origin_y = get_compile_time_arg_val(14);
    constexpr uint32_t scratch_slots = get_compile_time_arg_val(15);
    constexpr uint32_t cb_row_staging = get_compile_time_arg_val(16);
    constexpr auto input_args = TensorAccessorArgs<17>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // TILE storage: 32x32 tiles of four 16x16 faces (TL, TR, BL, BR), row-major inside a face.
    constexpr uint32_t TILE_DIM = 32;
    constexpr uint32_t FACE_DIM = 16;
    constexpr uint32_t face_row_bytes = FACE_DIM * element_bytes;
    constexpr uint32_t face_bytes = FACE_DIM * face_row_bytes;
    constexpr uint32_t tile_row_bytes = TILE_DIM * element_bytes;
    constexpr uint32_t slot_bytes = block_rows * block_row_bytes;
    static_assert(block_row_bytes >= block_tile_cols * tile_row_bytes, "staging row narrower than the block");
    // BH DRAM reads land at L1 dst == DRAM src mod 64: piece offsets on both sides are multiples of this.
    static_assert(face_row_bytes % 64 == 0 && block_row_bytes % 64 == 0, "staging layout breaks 64 B alignment");

    const auto input = TensorAccessor(input_args, get_common_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(1));

    // The core range is one rectangle (or full rows plus one partial row) at the given origin,
    // filled row by row, so this index is dense in [0, num_cores).
    const uint32_t core_idx =
        (get_relative_logical_y() - core_origin_y) * cores_x + (get_relative_logical_x() - core_origin_x);
    constexpr uint32_t base_blocks = total_blocks / num_cores;
    constexpr uint32_t extra_blocks = total_blocks % num_cores;
    const uint32_t first_block = core_idx * base_blocks + (core_idx < extra_blocks ? core_idx : extra_blocks);
    const uint32_t num_blocks_this_core = base_blocks + (core_idx < extra_blocks ? 1 : 0);

    const uint32_t staging_base = get_write_ptr(cb_row_staging);

    struct BlockExtent {
        uint32_t first_row;
        uint32_t rows;
        uint32_t first_tile_col;
        uint32_t tile_cols;
    };
    auto block_extent = [&](uint32_t block) -> BlockExtent {
        const uint32_t row_block = block / col_blocks;
        const uint32_t col_block = block - row_block * col_blocks;
        const uint32_t first_row = row_block * block_rows;
        const uint32_t first_tile_col = col_block * block_tile_cols;
        const uint32_t rows = (tensor_rows - first_row) < block_rows ? (tensor_rows - first_row) : block_rows;
        const uint32_t tile_cols = (tensor_tile_cols - first_tile_col) < block_tile_cols
                                       ? (tensor_tile_cols - first_tile_col)
                                       : block_tile_cols;
        return {first_row, rows, first_tile_col, tile_cols};
    };

    // DRAM tile pages -> slot, row order. Row p of the block takes faces 2h and 2h+1 of every
    // tile column (h = (r mod 32) div 16), face row r mod 16, one 64 B read per face.
    auto gather_block = [&](const BlockExtent& ext, uint32_t slot_addr) {
        {
            DeviceZoneScopedN("G6U-GATHER-ISSUE");
            uint32_t row_addr = slot_addr;
            for (uint32_t row = 0; row < ext.rows; ++row) {
                const uint32_t p = ext.first_row + row;
                const uint32_t image = p / rows_per_image;
                const uint32_t r = p - image * rows_per_image;
                const uint32_t tile_row = image * tile_rows_per_image + r / TILE_DIM;
                const uint32_t r_in_tile = r % TILE_DIM;
                const uint32_t piece_offset =
                    (r_in_tile / FACE_DIM) * 2 * face_bytes + (r_in_tile % FACE_DIM) * face_row_bytes;
                const uint32_t first_page = tile_row * tensor_tile_cols + ext.first_tile_col;
                uint32_t dst = row_addr;
                for (uint32_t col = 0; col < ext.tile_cols; ++col) {
                    const uint64_t src = input.get_noc_addr(first_page + col, piece_offset);
                    noc_async_read<face_row_bytes>(src, dst, face_row_bytes);
                    noc_async_read<face_row_bytes>(src + face_bytes, dst + face_row_bytes, face_row_bytes);
                    dst += tile_row_bytes;
                }
                row_addr += block_row_bytes;
            }
        }
        {
            DeviceZoneScopedN("G6U-GATHER-WAIT");
            noc_async_read_barrier();
        }
    };

    // Slot -> output pages. The write is clipped to the page so a W that is not a multiple of 32
    // never spills the tile padding into the next row's page.
    auto store_block = [&](const BlockExtent& ext, uint32_t slot_addr) {
        DeviceZoneScopedN("G6U-STORE-ISSUE");
        const uint32_t page_offset = ext.first_tile_col * tile_row_bytes;
        const uint32_t staged_bytes = ext.tile_cols * tile_row_bytes;
        const uint32_t write_bytes =
            (page_bytes - page_offset) < staged_bytes ? (page_bytes - page_offset) : staged_bytes;
        uint32_t src = slot_addr;
        for (uint32_t row = 0; row < ext.rows; ++row) {
            noc_async_write<block_row_bytes>(src, output.get_noc_addr(ext.first_row + row, page_offset), write_bytes);
            src += block_row_bytes;
        }
    };

    // A slot is re-gathered only after the writes that read it have left L1. With one slot that is
    // the previous block's writes, flushed before the gather; with a ring, the flush sits before
    // each store so the previous block's writes drain under this block's gather.
    auto wait_slot_free = [&]() {
        DeviceZoneScopedN("G6U-SLOT-WAIT");
        noc_async_writes_flushed();
    };

    uint32_t slot = 0;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const BlockExtent ext = block_extent(first_block + block_idx);
        const uint32_t slot_addr = staging_base + slot * slot_bytes;
        if constexpr (scratch_slots == 1) {
            if (block_idx > 0) {
                wait_slot_free();
            }
        }
        gather_block(ext, slot_addr);
        if constexpr (scratch_slots > 1) {
            if (block_idx > 0) {
                wait_slot_free();
            }
        }
        store_block(ext, slot_addr);
        slot = (slot + 1 == scratch_slots) ? 0 : slot + 1;
    }

    // BB-03: nothing may be left in flight when the kernel returns.
    {
        DeviceZoneScopedN("G6U-DRAIN");
        noc_async_write_barrier();
    }
}
