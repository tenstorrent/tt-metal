// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
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
    DPRINT << "Writer tilizer started" << ENDL();
    constexpr uint32_t indices_tensor_cb_id = get_named_compile_time_arg_val("indices_tensor_cb_id");
    constexpr uint32_t mapping_tensor_cb_id = get_named_compile_time_arg_val("mapping_tensor_cb_id");
    constexpr uint32_t scores_tensor_cb_id = get_named_compile_time_arg_val("scores_tensor_cb_id");
    constexpr uint32_t tilizer_input_cb_id = get_named_compile_time_arg_val("tilizer_input_cb_id");
    constexpr uint32_t tilizer_output_cb_id = get_named_compile_time_arg_val("tilizer_output_cb_id");
    constexpr uint32_t e_t_buffer_id = get_named_compile_time_arg_val("e_t_buffer_id");
    constexpr uint32_t expert_activation_cb_id = get_named_compile_time_arg_val("expert_activation_cb_id");
    constexpr uint32_t per_expert_total_tokens_cb_id = get_named_compile_time_arg_val("per_expert_total_tokens_cb_id");

    constexpr uint32_t input_pages = get_named_compile_time_arg_val("input_pages");
    constexpr uint32_t indices_pages = get_named_compile_time_arg_val("indices_pages");
    constexpr uint32_t mapping_pages = get_named_compile_time_arg_val("mapping_pages");

    constexpr uint32_t input_page_size = get_named_compile_time_arg_val("input_page_size");
    constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t mapping_page_size = get_named_compile_time_arg_val("mapping_page_size");
    constexpr uint32_t output_page_size = get_named_compile_time_arg_val("output_page_size");

    constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");

    constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
    constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");

    constexpr uint32_t aligned_indices_page_size = get_named_compile_time_arg_val("aligned_indices_page_size");
    constexpr uint32_t aligned_mapping_page_size = get_named_compile_time_arg_val("aligned_mapping_page_size");
    constexpr uint32_t aligned_output_page_size = get_named_compile_time_arg_val("aligned_output_page_size");

    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr uint32_t cluster_axis = get_named_compile_time_arg_val("cluster_axis");

    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t l1_alignment = get_named_compile_time_arg_val("l1_alignment");

    // Multicast coordinates for signaling tilizer cores
    constexpr uint32_t tilizer_mcast_start_x = get_named_compile_time_arg_val("tilizer_mcast_start_x");
    constexpr uint32_t tilizer_mcast_start_y = get_named_compile_time_arg_val("tilizer_mcast_start_y");
    constexpr uint32_t tilizer_mcast_end_x = get_named_compile_time_arg_val("tilizer_mcast_end_x");
    constexpr uint32_t tilizer_mcast_end_y = get_named_compile_time_arg_val("tilizer_mcast_end_y");
    constexpr uint32_t num_tilizer_cores = get_named_compile_time_arg_val("num_tilizer_cores");

    constexpr uint32_t selected_experts_k = get_named_compile_time_arg_val("selected_experts_k");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");

    constexpr uint32_t experts_per_device = (experts + num_devices - 1) / num_devices;

    constexpr ReplicateGroup axis = ReplicateGroup(cluster_axis);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t dispatch_index =
        axis == ReplicateGroup::COLS ? linearized_mesh_coord / mesh_cols : linearized_mesh_coord % mesh_cols;

    uint32_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);     // 0
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 1
    uint32_t scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);    // 2
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 3
    bool is_drain_tilizer_core = (bool)get_arg_val<uint32_t>(rt_args_idx++);  // 4

    // TensorAccessorArgs are provided in order: input, indices, scores, mapping, output
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto scores_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();

    const auto input_tensor_addr_gen = TensorAccessor(input_args, input_tensor_address, input_page_size);
    const auto indices_tensor_addr_gen = TensorAccessor(indices_args, indices_tensor_address, indices_page_size);
    const auto scores_tensor_addr_gen = TensorAccessor(scores_args, scores_tensor_address, indices_page_size);
    const auto mapping_tensor_addr_gen = TensorAccessor(mapping_args, mapping_tensor_address, mapping_page_size);
    // output_tensor_addr_gen will be set up when we have the output tensor address

    // TODO: Wait for reader_tilizer to populate E-T buffer, then compute total_chunks

    // Wait for compute kernel to push tilized output
    // constexpr uint32_t tiles_per_chunk = get_named_compile_time_arg_val("tiles_per_chunk");
    // for (uint32_t chunk = 0; chunk < total_chunks; chunk++) {
    //     // cb_wait_front(tilizer_output_cb_id, tiles_per_chunk);
    //     // cb_pop_front(tilizer_output_cb_id, tiles_per_chunk);
    // }
}
