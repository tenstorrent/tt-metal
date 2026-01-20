// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using namespace ttnn::operations::ccl::common;

// Helper to get multicast NOC address with proper coordinate ordering for NOC 0 vs NOC 1.
// NOC 0: start = (min_x, min_y), end = (max_x, max_y)
// NOC 1: start = (max_x, max_y), end = (min_x, min_y) - coordinates need to be swapped
FORCE_INLINE uint64_t get_safe_multicast_noc_addr(
    uint32_t noc_x_start,
    uint32_t noc_y_start,
    uint32_t noc_x_end,
    uint32_t noc_y_end,
    uint32_t addr,
    uint8_t noc = noc_index) {
    if (noc == 0) {
        return get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, addr, noc);
    } else {
        // For NOC 1, swap start and end coordinates
        return get_noc_multicast_addr(noc_x_end, noc_y_end, noc_x_start, noc_y_start, addr, noc);
    }
}

void print_tile_rows(
    uint32_t cb_idx,
    uint32_t tile_idx,
    bool untilize = false,
    uint16_t start_row = 0,
    uint16_t end_row = 32,
    uint8_t start_col = 0,
    uint8_t end_col = 32) {
    DPRINT << "cb_idx: " << cb_idx << " tile_idx: " << tile_idx << ENDL();
    DPRINT << "======" << ENDL();
    for (uint16_t r = start_row; r < end_row; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_idx,
                      tile_idx,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)start_col,
                          .w1 = (uint8_t)end_col,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

template <typename DataType>
FORCE_INLINE DataType* tile_row_offset(DataType* indices_address, uint32_t row) {
    constexpr uint32_t num_face_width = 2;
    constexpr uint32_t num_face_height = 2;
    constexpr uint32_t FaceWidth = 16;
    constexpr uint32_t FaceHeight = 16;
    constexpr uint32_t TileHeight = 32;
    constexpr uint32_t TileWidth = 32;
    uint32_t offset = 0;
    uint32_t local_row = row;
    if (row >= FaceHeight) {
        offset += num_face_width * FaceHeight * FaceWidth;  // if it was generic, multiply by row/FaceHeight
        local_row -= FaceHeight;
    }
    offset += local_row * FaceWidth;
    return (DataType*)(indices_address + offset);
}

template <typename DataType>
FORCE_INLINE DataType* tile_col_offset(DataType* indices_address, uint32_t col) {
    constexpr uint32_t FaceWidth = 16;
    constexpr uint32_t FaceHeight = 16;
    uint32_t offset = 0;
    uint32_t local_col = col;
    if (col >= FaceWidth) {
        offset += FaceHeight * FaceWidth;  // if it was generic, multiply by col/FaceWidth
        local_col -= FaceWidth;
    }
    offset += local_col;
    return (DataType*)(indices_address + offset);
}

void kernel_main() {
    //     // Compile-time arguments
    //     constexpr uint32_t tilizer_output_cb_id = get_named_compile_time_arg_val("tilizer_output_cb_id");
    //     constexpr uint32_t per_expert_total_tokens_cb_id =
    //     get_named_compile_time_arg_val("per_expert_total_tokens_cb_id"); constexpr uint32_t total_chunks_cb_id =
    //     get_named_compile_time_arg_val("total_chunks_cb_id");

    //     constexpr uint32_t output_page_size = get_named_compile_time_arg_val("output_page_size");
    //     constexpr uint32_t aligned_output_page_size = get_named_compile_time_arg_val("aligned_output_page_size");

    //     constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    //     constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    //     constexpr uint32_t hidden_size = get_named_compile_time_arg_val("hidden_size");

    //     constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    //     constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");

    //     constexpr uint32_t experts_per_device = (experts + num_devices - 1) / num_devices;

    //     // Tile dimensions
    //     constexpr uint32_t TILE_HEIGHT = 32;
    //     constexpr uint32_t TILE_WIDTH = 32;
    //     constexpr uint32_t tiles_per_row = hidden_size / TILE_WIDTH;  // Number of tiles in width dimension

    //     // Runtime arguments
    //     uint32_t rt_args_idx = 0;
    //     [[maybe_unused]] uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);     // 0
    //     [[maybe_unused]] uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 1
    //     [[maybe_unused]] uint32_t scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);    // 2
    //     [[maybe_unused]] uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 3
    //     uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);                     // 4
    //     [[maybe_unused]] bool is_drain_tilizer_core = (bool)get_arg_val<uint32_t>(rt_args_idx++);  // 5
    //     uint32_t tilizer_subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);                   // 6
    //     uint32_t tilizer_subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);                     // 7

    //     // TensorAccessorArgs are provided in order: input, indices, scores, mapping, output
    //     constexpr auto input_args = TensorAccessorArgs<0>();
    //     constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    //     constexpr auto scores_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    //     constexpr auto mapping_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    //     constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();

    //     const auto output_tensor_addr_gen = TensorAccessor(output_args, output_tensor_address, output_page_size);

    //     // Compute tiles_per_chunk for this core based on its subtoken portion
    //     // tile_width_bytes = TILE_WIDTH * element_size (element_size = output_page_size / (TILE_HEIGHT *
    //     TILE_WIDTH)) constexpr uint32_t element_size = output_page_size / (TILE_HEIGHT * TILE_WIDTH); constexpr
    //     uint32_t tile_width_bytes = TILE_WIDTH * element_size; uint32_t tiles_per_chunk = tilizer_subtoken_size /
    //     tile_width_bytes;

    //     // Compute width tile offset for this core
    //     uint32_t width_tile_start = tilizer_subtoken_offset / tile_width_bytes;

    //     // Wait for reader to push per-expert token counts
    //     cb_wait_front(per_expert_total_tokens_cb_id, experts_per_device);
    //     volatile tt_l1_ptr uint32_t* per_expert_counts =
    //         reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(per_expert_total_tokens_cb_id));

    //     // Read per-expert token counts into local array
    //     uint32_t num_tokens_per_expert[experts_per_device];
    //     for (uint32_t e = 0; e < experts_per_device; e++) {
    //         num_tokens_per_expert[e] = per_expert_counts[e];
    //     }

    //     // Wait for reader to push total_chunks
    //     cb_wait_front(total_chunks_cb_id, 1);
    //     [[maybe_unused]] uint32_t total_chunks =
    //         *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(total_chunks_cb_id));

    //     // Process each expert's chunks
    //     // Order matches reader: all chunks for expert 0, then expert 1, etc.
    //     for (uint32_t e = 0; e < experts_per_device; e++) {
    //         uint32_t num_expert_tokens = num_tokens_per_expert[e];
    //         uint32_t num_expert_chunks = (num_expert_tokens + tokens_per_chunk - 1) / tokens_per_chunk;

    //         for (uint32_t chunk = 0; chunk < num_expert_chunks; chunk++) {
    //             // Wait for compute to push tiles_per_chunk tiles
    //             cb_wait_front(tilizer_output_cb_id, tiles_per_chunk);

    //             // Compute the tile row for this chunk
    //             // linear_row_start = expert * tokens + chunk * tokens_per_chunk
    //             // But for output, we use total_tokens (max possible), not actual activated tokens
    //             // Actually, for correctness, we write at the position based on actual token index
    //             uint32_t token_row_start = e * tokens + chunk * tokens_per_chunk;
    //             uint32_t tile_row = token_row_start / TILE_HEIGHT;

    //             // Get L1 read pointer for the tilized output
    //             uint32_t l1_read_addr = get_read_ptr(tilizer_output_cb_id);

    //             // Write each tile to DRAM
    //             for (uint32_t t = 0; t < tiles_per_chunk; t++) {
    //                 uint32_t tile_col = width_tile_start + t;
    //                 uint32_t tile_idx = tile_row * tiles_per_row + tile_col;

    //                 uint64_t dst_noc_addr = get_noc_addr(tile_idx, output_tensor_addr_gen);
    //                 uint32_t src_l1_addr = l1_read_addr + t * output_page_size;

    //                 noc_async_write(src_l1_addr, dst_noc_addr, output_page_size);
    //             }
    //             noc_async_write_barrier();

    //             // Pop the tiles from CB
    //             cb_pop_front(tilizer_output_cb_id, tiles_per_chunk);
    //         }
    //     }

    //     // Pop the per-expert counts and total_chunks (cleanup)
    //     cb_pop_front(per_expert_total_tokens_cb_id, experts_per_device);
    //     cb_pop_front(total_chunks_cb_id, 1);
}
