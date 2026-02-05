// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
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

    constexpr bool write_page_by_page = get_compile_time_arg_val(35);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(36);

    // scores tensor compile time args
    constexpr uint32_t scores_tensor_cb_id = get_compile_time_arg_val(37);
    constexpr uint32_t scores_pages = get_compile_time_arg_val(38);
    constexpr uint32_t scores_page_size = get_compile_time_arg_val(39);
    constexpr uint32_t aligned_scores_page_size = get_compile_time_arg_val(40);

    constexpr auto input_args = TensorAccessorArgs<41>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto scores_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
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
    uint32_t scores_tensor_address = get_arg_val<uint32_t>(rt_ags++);
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_ags++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_ags++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_ags++);
    [[maybe_unused]] uint32_t scores_out_tensor_address = get_arg_val<uint32_t>(rt_ags++);

    uint32_t global_semaphore_address = get_arg_val<uint32_t>(rt_ags++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_ags++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_ags++);

    // Payload split parameters (always read from RT args)
    // In payload split mode: each worker reads only its portion of the input token
    // In non-split mode: defaults are payload_offset=0, payload_size=input_page_size, is_primary=true
    uint32_t payload_offset = get_arg_val<uint32_t>(rt_ags++);
    uint32_t payload_size = get_arg_val<uint32_t>(rt_ags++);
    bool is_primary_payload_worker = get_arg_val<uint32_t>(rt_ags++) == 1;

    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address, input_page_size);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address, indices_page_size);
    const auto scores_addr_gen = TensorAccessor(scores_args, scores_tensor_address, scores_page_size);
    const auto mapping_addr_gen = TensorAccessor(mapping_args, mapping_tensor_address, mapping_page_size);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, metadata_page_size);

    // Read the expert mapping table - new format: [devices, experts]
    // Each page is one device's view of the mapping. Read only the source device's page.
    // Page index = linearized_mesh_coord (source device index)
    constexpr uint32_t mapping_pages_to_read = 1;
    cb_reserve_back(mapping_tensor_cb_id, mapping_pages_to_read);
    uint32_t base_mapping_addr = get_write_ptr(mapping_tensor_cb_id);
    noc_async_read_page(linearized_mesh_coord, mapping_addr_gen, base_mapping_addr);

    ASSERT(indices_pages == input_pages);
    ASSERT(scores_pages == indices_pages);
    // read the input tokens, selected experts, and scores for each token
    uint32_t base_indices_addr = get_write_ptr(indices_tensor_cb_id);
    uint32_t base_scores_addr = get_write_ptr(scores_tensor_cb_id);
    noc_async_read_barrier();
    cb_push_back(mapping_tensor_cb_id, mapping_pages_to_read);

    for (uint32_t i = token_start_idx; i < token_end_idx; i++) {
        cb_reserve_back(indices_tensor_cb_id, 1);
        cb_reserve_back(input_tensor_cb_id, 1);

        // All workers read indices (needed for routing decisions)
        uint32_t l1_write_addr = get_write_ptr(indices_tensor_cb_id);
        noc_async_read_page(i, indices_addr_gen, l1_write_addr);

        // Only primary worker reads scores (only primary sends metadata)
        if (is_primary_payload_worker) {
            cb_reserve_back(scores_tensor_cb_id, 1);
            l1_write_addr = get_write_ptr(scores_tensor_cb_id);
            noc_async_read_page(i, scores_addr_gen, l1_write_addr);
        }

        // Read input token (or portion of it in payload split mode)
        // In non-split mode: payload_offset=0, payload_size=input_page_size (reads full page)
        // In split mode: reads only this worker's portion
        uint64_t input_page_noc_addr = get_noc_addr(i, input_addr_gen);
        l1_write_addr = get_write_ptr(input_tensor_cb_id);
        noc_async_read(input_page_noc_addr + payload_offset, l1_write_addr, payload_size);

        noc_async_read_barrier();
        cb_push_back(indices_tensor_cb_id, 1);
        if (is_primary_payload_worker) {
            cb_push_back(scores_tensor_cb_id, 1);
        }
        cb_push_back(input_tensor_cb_id, 1);
    }

    // Only primary worker copies indices and scores to metadata buffer (only primary sends metadata)
    if (is_primary_payload_worker) {
        cb_reserve_back(metadata_buffer_id, tokens_per_device);
        noc_async_read(
            get_noc_addr(base_indices_addr),
            get_write_ptr(metadata_buffer_id),
            (token_end_idx - token_start_idx) * aligned_indices_page_size);
        noc_async_read(
            get_noc_addr(base_scores_addr),
            get_write_ptr(metadata_buffer_id) + (token_end_idx - token_start_idx) * aligned_indices_page_size,
            (token_end_idx - token_start_idx) * aligned_scores_page_size);
        noc_async_read_barrier();
        cb_push_back(metadata_buffer_id, tokens_per_device);

        // Wait for all other devices to finish dispatching their input tokens and metadata.
        // The writer now writes metadata directly to the sharded output tensor on the drain sync tilizer core,
        // so we no longer need to copy from the intermediate buffer to the final output here.
        noc_semaphore_wait((uint32_t*)global_semaphore_address, dispatch_devices);
        noc_semaphore_set((uint32_t*)global_semaphore_address, 0);
    }
}
