// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regime N (new_storage) reader, NCRISC / NoC0.
//
// Per block of block_tiles output tiles: slot i of cb_work_tiles <-> block tile i. Every
// input-backed tile is read whole into its slot; after the block's read barrier, the parts of a
// boundary tile that lie in x's tile padding are overwritten with c. Fill-only slots stay untouched
// (the writer sends those from its own fill page). Push/pop are always the nominal block_tiles so
// the CB ring closes on a ragged last block.
//
// Raw dataflow for the page reads: no kernel_lib helper wraps interleaved page reads for a block.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "g6_pad_new_storage_walk.hpp"

namespace {

constexpr uint32_t FACE_DIM = 16;

// Rows [row_mask_rem, 32) of one tile, all columns: faces of 16 rows, face order 0,1 (rows 0-15),
// 2,3 (rows 16-31).
template <uint32_t row_mask_rem, uint32_t elem_bytes>
FORCE_INLINE void mask_rows(uint32_t slot, uint32_t fill_bits) {
    constexpr uint32_t face_row_bytes = FACE_DIM * elem_bytes;
    constexpr uint32_t face_bytes = FACE_DIM * face_row_bytes;
    if constexpr (row_mask_rem < FACE_DIM) {
        constexpr uint32_t skip = row_mask_rem * face_row_bytes;
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(slot + skip, face_bytes - skip, fill_bits);
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(slot + face_bytes + skip, 3 * face_bytes - skip, fill_bits);
    } else {
        constexpr uint32_t skip = (row_mask_rem - FACE_DIM) * face_row_bytes;
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(slot + 2 * face_bytes + skip, face_bytes - skip, fill_bits);
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(slot + 3 * face_bytes + skip, face_bytes - skip, fill_bits);
    }
}

// Columns [col_mask_rem, 32) of one tile, all rows. Faces 0,2 hold columns 0-15, faces 1,3
// columns 16-31.
template <uint32_t col_mask_rem, uint32_t elem_bytes>
FORCE_INLINE void mask_cols(uint32_t slot, uint32_t fill_bits) {
    constexpr uint32_t face_row_bytes = FACE_DIM * elem_bytes;
    constexpr uint32_t face_bytes = FACE_DIM * face_row_bytes;
    constexpr bool right_faces_whole = col_mask_rem < FACE_DIM;
    constexpr uint32_t first_col = right_faces_whole ? col_mask_rem : col_mask_rem - FACE_DIM;
    constexpr uint32_t skip = first_col * elem_bytes;
    constexpr uint32_t part_face_a = right_faces_whole ? 0 : face_bytes;
    constexpr uint32_t part_face_b = right_faces_whole ? 2 * face_bytes : 3 * face_bytes;
    if constexpr (right_faces_whole) {
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(slot + face_bytes, face_bytes, fill_bits);
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(slot + 3 * face_bytes, face_bytes, fill_bits);
    }
    for (uint32_t row = 0; row < FACE_DIM; ++row) {
        const uint32_t row_offset = row * face_row_bytes + skip;
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(
            slot + part_face_a + row_offset, face_row_bytes - skip, fill_bits);
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(
            slot + part_face_b + row_offset, face_row_bytes - skip, fill_bits);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_work_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t geometry_ct_base = 4;
    using Walk = g6_pad::NewStorageWalk<geometry_ct_base>;
    constexpr auto x_args = TensorAccessorArgs<geometry_ct_base + g6_pad::kGeometryArgCount>();

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t fill_bits = get_common_arg_val<uint32_t>(1);
    const uint32_t core_work_begin = get_arg_val<uint32_t>(0);
    const uint32_t core_work_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);

    const auto x_accessor = TensorAccessor(x_args, x_addr, tile_bytes);

    Walk walk;
    walk.start(core_work_begin);
    uint32_t tiles_left = core_work_tiles;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t block_extent = tiles_left < block_tiles ? tiles_left : block_tiles;
        [[maybe_unused]] const Walk block_first = walk;

        // load_block
        {
            DeviceZoneScopedN("G6PAD_SLOT_WAIT");
            cb_reserve_back(cb_work_tiles, block_tiles);
        }
        const uint32_t block_base = get_write_ptr(cb_work_tiles);
        {
            DeviceZoneScopedN("G6PAD_LOAD");
            for (uint32_t i = 0; i < block_extent; ++i) {
                if (walk.has_input()) {
                    noc_async_read_page(walk.in_page(), x_accessor, block_base + i * tile_bytes);
                }
                walk.advance();
            }
        }
        {
            DeviceZoneScopedN("G6PAD_READ_WAIT");
            noc_async_read_barrier();
        }

        // assemble_block: masks strictly after the barrier, or the landing read overwrites them.
        {
            DeviceZoneScopedN("G6PAD_ASSEMBLE");
            if constexpr (Walk::has_masks) {
                Walk mask_walk = block_first;
                for (uint32_t i = 0; i < block_extent; ++i) {
                    if (mask_walk.has_input()) {
                        const uint32_t slot = block_base + i * tile_bytes;
                        if constexpr (Walk::row_mask_rem != 0) {
                            if (mask_walk.row_mask()) {
                                mask_rows<Walk::row_mask_rem, elem_bytes>(slot, fill_bits);
                            }
                        }
                        if constexpr (Walk::col_mask_rem != 0) {
                            if (mask_walk.col_mask()) {
                                mask_cols<Walk::col_mask_rem, elem_bytes>(slot, fill_bits);
                            }
                        }
                    }
                    mask_walk.advance();
                }
            }
            cb_push_back(cb_work_tiles, block_tiles);
        }
        tiles_left -= block_extent;
    }
}
