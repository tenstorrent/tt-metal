// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using namespace ttnn::operations::ccl::common;

void kernel_main() {
    constexpr uint32_t input_tensor_cb_id = get_named_compile_time_arg_val("input_tensor_cb_id");
    constexpr uint32_t indices_tensor_cb_id = get_named_compile_time_arg_val("indices_tensor_cb_id");
    constexpr uint32_t mapping_tensor_cb_id = get_named_compile_time_arg_val("mapping_tensor_cb_id");

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

    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t dispatch_index =
        axis == ReplicateGroup::COLS ? linearized_mesh_coord / mesh_cols : linearized_mesh_coord % mesh_cols;

    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_ags++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_ags++);
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_ags++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_ags++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_ags++);

    uint32_t global_semaphore_address = get_arg_val<uint32_t>(rt_ags++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_ags++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_ags++);

    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address, input_page_size);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address, indices_page_size);
    const auto mapping_addr_gen = TensorAccessor(mapping_args, mapping_tensor_address, mapping_page_size);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, metadata_page_size);

    // read the expert mapping table
    cb_reserve_back(mapping_tensor_cb_id, mapping_pages);
    uint32_t base_indices_addr = get_write_ptr(mapping_tensor_cb_id);
    for (uint32_t i = 0; i < mapping_pages; i++) {
        uint32_t l1_write_addr = get_write_ptr(mapping_tensor_cb_id) + i * aligned_mapping_page_size;
        noc_async_read_page(i, mapping_addr_gen, l1_write_addr);
    }
    noc_async_read_barrier();
    cb_push_back(mapping_tensor_cb_id, mapping_pages);

    for (uint32_t i = 0; i < indices_pages; i++) {
        cb_reserve_back(indices_tensor_cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(indices_tensor_cb_id) + i * aligned_indices_page_size;
        noc_async_read_page(i, indices_addr_gen, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(indices_tensor_cb_id, 1);
    }

    // read the input tokens and the selected experts for each token
    for (uint32_t i = token_start_idx; i < token_end_idx; i++) {
        cb_reserve_back(input_tensor_cb_id, 1);

        uint32_t l1_write_addr = get_write_ptr(input_tensor_cb_id);
        noc_async_read_page(i, input_addr_gen, l1_write_addr);

        noc_async_read_barrier();
        cb_push_back(input_tensor_cb_id, 1);
    }
}
