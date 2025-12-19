// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using namespace ttnn::operations::ccl::common;

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

void kernel_main() {
    constexpr uint32_t input_tensor_cb_id = get_named_compile_time_arg_val("input_tensor_cb_id");
    constexpr uint32_t indices_tensor_cb_id = get_named_compile_time_arg_val("indices_tensor_cb_id");
    constexpr uint32_t mapping_tensor_cb_id = get_named_compile_time_arg_val("mapping_tensor_cb_id");
    constexpr uint32_t scores_tensor_cb_id = get_named_compile_time_arg_val("scores_tensor_cb_id");

    constexpr uint32_t input_pages = get_named_compile_time_arg_val("input_pages");
    constexpr uint32_t indices_pages = get_named_compile_time_arg_val("indices_pages");
    constexpr uint32_t mapping_pages = get_named_compile_time_arg_val("mapping_pages");

    constexpr uint32_t input_page_size = get_named_compile_time_arg_val("input_page_size");
    constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t mapping_page_size = get_named_compile_time_arg_val("mapping_page_size");
    constexpr uint32_t metadata_page_size = get_named_compile_time_arg_val("metadata_page_size");

    constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t tokens_per_device = get_named_compile_time_arg_val("tokens_per_device");

    constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
    constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");

    constexpr uint32_t aligned_indices_page_size = get_named_compile_time_arg_val("aligned_indices_page_size");
    constexpr uint32_t aligned_mapping_page_size = get_named_compile_time_arg_val("aligned_mapping_page_size");
    constexpr uint32_t aligned_metadata_page_size = get_named_compile_time_arg_val("aligned_metadata_page_size");

    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr uint32_t cluster_axis = get_named_compile_time_arg_val("cluster_axis");
    constexpr uint32_t max_indices_pages_per_packet = get_named_compile_time_arg_val("max_indices_pages_per_packet");

    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto input_scores_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();
    constexpr auto output_scores_args = TensorAccessorArgs<input_scores_args.next_compile_time_args_offset()>();

    constexpr ReplicateGroup axis = ReplicateGroup(cluster_axis);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t dispatch_index =
        axis == ReplicateGroup::COLS ? linearized_mesh_coord / mesh_cols : linearized_mesh_coord % mesh_cols;

    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);          // 0
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);        // 1
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);        // 2
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);         // 3
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);       // 4
    uint32_t global_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);      // 5
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);        // 6
    uint32_t input_scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 7
    uint32_t output_scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);  // 8
    uint32_t subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);               // 9
    uint32_t subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);                 // 10
    uint32_t indices_start = get_arg_val<uint32_t>(rt_args_idx++);                 // 11
    uint32_t indices_end = get_arg_val<uint32_t>(rt_args_idx++);                   // 12

    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address, input_page_size);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address, indices_page_size);
    const auto mapping_addr_gen = TensorAccessor(mapping_args, mapping_tensor_address, mapping_page_size);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, metadata_page_size);
    // Scores tensors use same page size as indices tensor
    const auto input_scores_addr_gen =
        TensorAccessor(input_scores_args, input_scores_tensor_address, indices_page_size);
    const auto output_scores_addr_gen =
        TensorAccessor(output_scores_args, output_scores_tensor_address, indices_page_size);

    // read in the indices tensor
    for (uint32_t indices_page = indices_start; indices_page < indices_end;
         indices_page += max_indices_pages_per_packet) {
        uint32_t pages_left = indices_end - indices_page;
        uint32_t pages_to_read = std::min(max_indices_pages_per_packet, pages_left);
        cb_reserve_back(
            indices_tensor_cb_id,
            max_indices_pages_per_packet);  // always reserve the max number of pages so writer logic is simpler
        for (uint32_t i = 0; i < pages_to_read; i++) {
            noc_async_read_page(
                indices_page + i,
                indices_addr_gen,
                get_write_ptr(indices_tensor_cb_id) + i * aligned_indices_page_size);
        }
        noc_async_read_barrier();
        cb_push_back(indices_tensor_cb_id, max_indices_pages_per_packet);
    }

    // read in the input scores tensor
    for (uint32_t input_scores_page = indices_start; input_scores_page < indices_end;
         input_scores_page += max_indices_pages_per_packet) {
        uint32_t pages_left = indices_end - input_scores_page;
        uint32_t pages_to_read = std::min(max_indices_pages_per_packet, pages_left);
        cb_reserve_back(
            scores_tensor_cb_id,
            max_indices_pages_per_packet);  // always reserve the max number of pages so writer logic is simpler
        for (uint32_t i = 0; i < pages_to_read; i++) {
            noc_async_read_page(
                input_scores_page + i,
                input_scores_addr_gen,
                get_write_ptr(scores_tensor_cb_id) + i * aligned_indices_page_size);
        }
        noc_async_read_barrier();
        cb_push_back(scores_tensor_cb_id, max_indices_pages_per_packet);
    }

    // read in the mapping tensor
    for (uint32_t mapping_page = 0; mapping_page < mapping_pages; mapping_page++) {
        cb_reserve_back(mapping_tensor_cb_id, 1);
        noc_async_read_page(mapping_page, mapping_addr_gen, get_write_ptr(mapping_tensor_cb_id));
        noc_async_read_barrier();
        cb_push_back(mapping_tensor_cb_id, 1);
    }

    constexpr bool reuse_index = true;
    constexpr uint32_t tile_height = 32;
    for (uint32_t token = 0; token < tokens_per_device; token++) {
        if constexpr (!reuse_index) {
            if ((token % tile_height) == 0) {
                cb_reserve_back(indices_tensor_cb_id, 1);
                noc_async_read_page(token / tile_height, indices_addr_gen, get_write_ptr(indices_tensor_cb_id));
            }
        }
        cb_reserve_back(input_tensor_cb_id, 1);
        noc_async_read(
            get_noc_addr(token, input_addr_gen) + subtoken_offset, get_write_ptr(input_tensor_cb_id), subtoken_size);
        noc_async_read_barrier();
        if constexpr (!reuse_index) {
            if ((token % tile_height) == 0) {
                cb_push_back(indices_tensor_cb_id, 1);
            }
        }
        cb_push_back(input_tensor_cb_id, 1);
    }
}
