// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using namespace ttnn::operations::ccl::common;

void kernel_main() {
    constexpr uint32_t input_pages = get_arg(args::input_pages);
    constexpr uint32_t indices_pages = get_arg(args::indices_pages);
    constexpr uint32_t mapping_pages = get_arg(args::mapping_pages);

    constexpr uint32_t metadata_page_size = get_arg(args::metadata_page_size);

    constexpr uint32_t num_devices = get_arg(args::num_devices);
    constexpr uint32_t tokens_per_device = get_arg(args::tokens_per_device);

    constexpr uint32_t src_mesh_id = get_arg(args::src_mesh_id);
    constexpr uint32_t src_chip_id = get_arg(args::src_chip_id);

    constexpr uint32_t mesh_rows = get_arg(args::mesh_rows);
    constexpr uint32_t mesh_cols = get_arg(args::mesh_cols);  // ew_dim

    constexpr uint32_t aligned_indices_page_size = get_arg(args::aligned_indices_page_size);
    constexpr uint32_t aligned_mapping_page_size = get_arg(args::aligned_mapping_page_size);
    constexpr uint32_t aligned_metadata_page_size = get_arg(args::aligned_metadata_page_size);

    constexpr uint32_t linearized_mesh_coord = get_arg(args::linearized_mesh_coord);

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
    experimental::GlobalSemaphore global_semaphore(global_sem::global_semaphore);
    uint32_t token_start_idx = get_arg(args::token_start_idx);
    uint32_t token_end_idx = get_arg(args::token_end_idx);

    const auto input_addr_gen = TensorAccessor(tensor::input);
    const auto indices_addr_gen = TensorAccessor(tensor::indices);
    const auto mapping_addr_gen = TensorAccessor(tensor::mapping);
    const auto metadata_addr_gen = TensorAccessor(tensor::metadata);

    Noc noc;
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_indices(dfb::indices);
    DataflowBuffer dfb_mapping(dfb::mapping);

    // read the expert mapping table
    dfb_mapping.reserve_back(mapping_pages);
    for (uint32_t i = 0; i < mapping_pages; i++) {
        noc.async_read(
            mapping_addr_gen,
            dfb_mapping,
            mapping_addr_gen.get_aligned_page_size(),
            {.page_id = i},
            {.offset_bytes = i * aligned_mapping_page_size});
    }
    noc.async_read_barrier();
    dfb_mapping.push_back(mapping_pages);

    ASSERT(indices_pages == input_pages);
    // read the input tokens and the selected experts for each token
    for (uint32_t i = token_start_idx; i < token_end_idx; i++) {
        dfb_indices.reserve_back(1);
        dfb_input.reserve_back(1);

        noc.async_read(
            indices_addr_gen,
            dfb_indices,
            indices_addr_gen.get_aligned_page_size(),
            {.page_id = i},
            {.offset_bytes = 0});

        noc.async_read(
            input_addr_gen, dfb_input, input_addr_gen.get_aligned_page_size(), {.page_id = i}, {.offset_bytes = 0});

        noc.async_read_barrier();
        dfb_indices.push_back(1);
        dfb_input.push_back(1);
    }

    // wait for all other devices to finish dispatching their input tokens and metadata
    uint32_t my_device_offset = tokens_per_device * dispatch_index;
    // Ring: dispatch_devices completion credits plus 1 local, for odd and even rings (see the writer's
    // send_ring_completion_credit). Otherwise one credit per metadata packet.
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_arg(args::topology);
    constexpr bool ring_completion = topology == tt::tt_fabric::Topology::Ring && axis != ReplicateGroup::NONE;
#ifndef FULL_PACKET
    {
        // if the writer is directly sending the metadata to its output buffer, we just wait for the semaphore to be set
        global_semaphore.wait(
            ring_completion ? dispatch_devices + 1 : (token_end_idx - token_start_idx) * dispatch_devices);
        global_semaphore.set(0);
    }
#else
    {
        // if the writer is sending the metadata to the intermediate buffer, we need to write our metadata to the final
        // buffer
        global_semaphore.wait(ring_completion ? dispatch_devices + 1 : dispatch_devices);
        global_semaphore.set(0);

        DataflowBuffer dfb_metadata_buffer(dfb::metadata_buffer);
        const CoreLocalMem<uint32_t> metadata_buffer(dfb_metadata_buffer.get_write_ptr());
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
#endif
}
