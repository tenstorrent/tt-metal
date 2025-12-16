// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using namespace ttnn::operations::ccl::common;

void kernel_main() {
    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t indices_tensor_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t mapping_tensor_cb_id = get_compile_time_arg_val(2);

    constexpr uint32_t input_pages = get_compile_time_arg_val(5);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(6);
    constexpr uint32_t mapping_pages = get_compile_time_arg_val(7);

    constexpr uint32_t input_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t mapping_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(14);

    constexpr uint32_t num_devices = get_compile_time_arg_val(15);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(20);

    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(23);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(24);

    constexpr uint32_t mesh_rows = get_compile_time_arg_val(25);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(26);  // ew_dim

    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(28);
    constexpr uint32_t aligned_mapping_page_size = get_compile_time_arg_val(29);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(31);

    constexpr uint32_t metadata_buffer_id = get_compile_time_arg_val(34);

    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(36);

    constexpr auto input_args = TensorAccessorArgs<37>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t dispatch_index =
        axis == ReplicateGroup::COLS ? linearized_mesh_coord / mesh_cols : linearized_mesh_coord % mesh_cols;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t dispatch_devices = num_devices;
    constexpr uint32_t dispatch_index = linearized_mesh_coord;
#endif
    uint32_t rt_ags = 0;
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
