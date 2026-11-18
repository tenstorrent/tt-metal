// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using namespace ttnn::operations::ccl::common;

void kernel_main() {
    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t indices_tensor_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t mapping_tensor_cb_id = get_compile_time_arg_val(2);

    constexpr uint32_t input_pages = get_compile_time_arg_val(5);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(6);
    constexpr uint32_t mapping_pages = get_compile_time_arg_val(7);

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

    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address);
    const auto mapping_addr_gen = TensorAccessor(mapping_args, mapping_tensor_address);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address);

    Noc noc;
    CircularBuffer input_cb(input_tensor_cb_id);
    CircularBuffer indices_cb(indices_tensor_cb_id);
    CircularBuffer mapping_cb(mapping_tensor_cb_id);

    // read the expert mapping table
    mapping_cb.reserve_back(mapping_pages);
    for (uint32_t i = 0; i < mapping_pages; i++) {
        noc.async_read(
            mapping_addr_gen,
            mapping_cb,
            mapping_addr_gen.get_aligned_page_size(),
            {.page_id = i},
            {.offset_bytes = i * aligned_mapping_page_size});
    }
    noc.async_read_barrier();
    mapping_cb.push_back(mapping_pages);

    ASSERT(indices_pages == input_pages);
    // read the input tokens and the selected experts for each token
    for (uint32_t i = token_start_idx; i < token_end_idx; i++) {
        indices_cb.reserve_back(1);
        input_cb.reserve_back(1);

        noc.async_read(
            indices_addr_gen,
            indices_cb,
            indices_addr_gen.get_aligned_page_size(),
            {.page_id = i},
            {.offset_bytes = 0});

        noc.async_read(
            input_addr_gen, input_cb, input_addr_gen.get_aligned_page_size(), {.page_id = i}, {.offset_bytes = 0});

        noc.async_read_barrier();
        indices_cb.push_back(1);
        input_cb.push_back(1);
    }

    // wait for all other devices to finish dispatching their input tokens and metadata
    uint32_t my_device_offset = tokens_per_device * dispatch_index;
    // Ring: dispatch_devices completion credits plus 1 local, for odd and even rings (see the writer's
    // send_ring_completion_credit). Otherwise one credit per metadata packet.
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(22);
    constexpr bool ring_completion = topology == tt::tt_fabric::Topology::Ring && axis != ReplicateGroup::NONE;
    if constexpr (write_page_by_page) {
        // if the writer is directly sending the metadata to its output buffer, we just wait for the semaphore to be set
        noc_semaphore_wait(
            (uint32_t*)global_semaphore_address,
            ring_completion ? dispatch_devices + 1 : (token_end_idx - token_start_idx) * dispatch_devices);
        noc_semaphore_set((uint32_t*)global_semaphore_address, 0);
    } else {
        // if the writer is sending the metadata to the intermediate buffer, we need to write our metadata to the final
        // buffer
        noc_semaphore_wait(
            (uint32_t*)global_semaphore_address, ring_completion ? dispatch_devices + 1 : dispatch_devices);
        noc_semaphore_set((uint32_t*)global_semaphore_address, 0);

        CircularBuffer metadata_buffer_cb(metadata_buffer_id);
        const CoreLocalMem<uint32_t> metadata_buffer(metadata_buffer_cb.get_write_ptr());
        for (uint32_t t = token_start_idx; t < token_end_idx; t++) {
            for (uint32_t d = 0; d < dispatch_devices; d++) {
                uint32_t page = d * tokens_per_device + t;
                noc.async_write(
                    metadata_buffer,
                    metadata_addr_gen,
                    metadata_page_size,
                    {.offset_bytes = page * aligned_indices_page_size},
                    {.page_id = page});
            }
        }
        noc.async_write_barrier();
    }
}
