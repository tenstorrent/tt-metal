// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/tensor/noc_traits.h"

using namespace ttnn::operations::ccl::common;
using tt::data_movement::common::tt_memmove;

void kernel_main() {
    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t indices_tensor_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t mapping_tensor_cb_id = get_compile_time_arg_val(2);

    [[maybe_unused]] constexpr uint32_t input_pages = get_compile_time_arg_val(5);
    [[maybe_unused]] constexpr uint32_t indices_pages = get_compile_time_arg_val(6);
    [[maybe_unused]] constexpr uint32_t mapping_pages = get_compile_time_arg_val(7);

    constexpr uint32_t input_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t mapping_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(14);

    [[maybe_unused]] constexpr uint32_t num_devices = get_compile_time_arg_val(15);
    [[maybe_unused]] constexpr uint32_t selected_experts_k = get_compile_time_arg_val(18);
    constexpr uint32_t num_shared_experts = get_compile_time_arg_val(20);

    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(21);

    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(23);

    [[maybe_unused]] constexpr uint32_t src_mesh_id = get_compile_time_arg_val(24);
    [[maybe_unused]] constexpr uint32_t src_chip_id = get_compile_time_arg_val(25);

    [[maybe_unused]] constexpr uint32_t mesh_rows = get_compile_time_arg_val(26);
    [[maybe_unused]] constexpr uint32_t mesh_cols = get_compile_time_arg_val(27);  // ew_dim

    [[maybe_unused]] constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(29);
    [[maybe_unused]] constexpr uint32_t aligned_mapping_page_size = get_compile_time_arg_val(30);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(32);

    constexpr uint32_t metadata_buffer_id = get_compile_time_arg_val(35);

    [[maybe_unused]] constexpr bool write_page_by_page = get_compile_time_arg_val(36);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(37);

    constexpr uint32_t dispatch_devices = get_compile_time_arg_val(38);

    // scores tensor compile time args
    constexpr uint32_t scores_tensor_cb_id = get_compile_time_arg_val(39);
    [[maybe_unused]] constexpr uint32_t scores_pages = get_compile_time_arg_val(40);
    constexpr uint32_t scores_page_size = get_compile_time_arg_val(41);
    constexpr uint32_t aligned_output_scores_page_size = get_compile_time_arg_val(44);

    constexpr auto input_args = TensorAccessorArgs<45>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto scores_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    uint32_t rt_ags = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_ags++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_ags++);
    uint32_t scores_tensor_address = get_arg_val<uint32_t>(rt_ags++);
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_ags++);
    [[maybe_unused]] uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_ags++);
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

    // Note: these values are uint16_t packed in the uint32_t rt arg buffer
    const auto shared_expert_ids_addr = get_arg_addr(rt_ags++);
    constexpr uint32_t shared_expert_data_size_bytes = num_shared_experts * sizeof(uint16_t);
    constexpr uint16_t bf16_one = 0x3f80;

    Noc noc;

    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address, input_page_size);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address, indices_page_size);
    const auto scores_addr_gen = TensorAccessor(scores_args, scores_tensor_address, scores_page_size);
    const auto mapping_addr_gen = TensorAccessor(mapping_args, mapping_tensor_address, mapping_page_size);
    [[maybe_unused]] const auto metadata_addr_gen =
        TensorAccessor(metadata_args, metadata_tensor_address, metadata_page_size);

    Noc noc_obj;
    CircularBuffer cb_input(input_tensor_cb_id);
    CircularBuffer cb_indices(indices_tensor_cb_id);
    CircularBuffer cb_scores(scores_tensor_cb_id);
    CircularBuffer cb_mapping(mapping_tensor_cb_id);
    CircularBuffer cb_metadata(metadata_buffer_id);

    if (token_start_idx == token_end_idx) {
        return;
    }

    // Read the expert mapping table - new format: [devices, experts]
    // Each page is one device's view of the mapping. Read only the source device's page.
    // Page index = linearized_mesh_coord (source device index)
    constexpr uint32_t mapping_pages_to_read = 1;
    cb_mapping.reserve_back(mapping_pages_to_read);
    uint32_t base_mapping_addr = cb_mapping.get_write_ptr();
    noc_obj.async_read(mapping_addr_gen, cb_mapping, mapping_page_size, {.page_id = linearized_mesh_coord}, {});

    ASSERT(indices_pages == input_pages);
    ASSERT(scores_pages == indices_pages);
    // read the input tokens, selected experts, and scores for each token
    uint32_t base_indices_addr = cb_indices.get_write_ptr();
    uint32_t base_scores_addr = cb_scores.get_write_ptr();
    noc_obj.async_read_barrier();
    cb_mapping.push_back(mapping_pages_to_read);

    for (uint32_t i = token_start_idx; i < token_end_idx; i++) {
        cb_indices.reserve_back(1);
        cb_input.reserve_back(1);

        // All workers read indices (needed for routing decisions)
        uint32_t l1_write_addr = cb_indices.get_write_ptr();
        noc_obj.async_read(indices_addr_gen, cb_indices, indices_page_size, {.page_id = i}, {});

        // manually fill in shared expert IDs to the metadata
        if constexpr (num_shared_experts > 0) {
            const uint32_t shared_expert_id_l1_addr = l1_write_addr + indices_page_size;
            tt_memmove<false, true, true, shared_expert_data_size_bytes>(
                noc, shared_expert_id_l1_addr, shared_expert_ids_addr, shared_expert_data_size_bytes);
        }

        // Only primary worker reads scores (only primary sends metadata)
        if (is_primary_payload_worker) {
            cb_scores.reserve_back(1);
            l1_write_addr = cb_scores.get_write_ptr();
            noc_obj.async_read(scores_addr_gen, cb_scores, scores_page_size, {.page_id = i}, {});

            if constexpr (num_shared_experts > 0) {
                auto* l1_scores_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
                for (uint32_t e = 0; e < num_shared_experts; ++e) {
                    l1_scores_ptr[selected_experts_k + e] = bf16_one;
                }
            }
        }

        // Read input token (or portion of it in payload split mode)
        // In non-split mode: payload_offset=0, payload_size=input_page_size (reads full page)
        // In split mode: reads only this worker's portion
        noc_obj.async_read(input_addr_gen, cb_input, payload_size, {.page_id = i, .offset_bytes = payload_offset}, {});

        noc_obj.async_read_barrier();
        cb_indices.push_back(1);
        if (is_primary_payload_worker) {
            cb_scores.push_back(1);
        }
        cb_input.push_back(1);
    }

    // Only primary worker copies indices and scores to metadata buffer (only primary sends metadata)
    if (is_primary_payload_worker) {
        cb_metadata.reserve_back(tokens_per_device);
        const uint32_t metadata_buffer_addr = cb_metadata.get_write_ptr();
        // Device 2.0 migration: legacy primitive retained: local-L1 self-read via get_noc_addr(local_l1_addr)
        noc_async_read(
            get_noc_addr(base_indices_addr),
            metadata_buffer_addr,
            (token_end_idx - token_start_idx) * aligned_metadata_page_size);
        // Device 2.0 migration: legacy primitive retained: local-L1 self-read via get_noc_addr(local_l1_addr)
        noc_async_read(
            get_noc_addr(base_scores_addr),
            cb_metadata.get_write_ptr() + (token_end_idx - token_start_idx) * aligned_metadata_page_size,
            (token_end_idx - token_start_idx) * aligned_output_scores_page_size);

        noc_obj.async_read_barrier();
        cb_metadata.push_back(tokens_per_device);

        // Wait for all other devices to finish dispatching their input tokens and metadata.
        // The writer now writes metadata directly to the sharded output tensor on the drain sync tilizer core,
        // so we no longer need to copy from the intermediate buffer to the final output here.
        constexpr uint32_t expected_dispatch_device_inc =
            (topology == tt::tt_fabric::Topology::Linear) ? (dispatch_devices - 1) : dispatch_devices;
        // Device 2.0 migration: legacy primitive retained: global_semaphore_address is a GlobalSemaphore address.
        noc_semaphore_wait((uint32_t*)global_semaphore_address, expected_dispatch_device_inc);
        noc_semaphore_set((uint32_t*)global_semaphore_address, 0);
    }
}
