// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::operations::experimental::fabric_all_gather {

// Chip shard layout of one cached program (terms: kernels/fabric_all_gather_chunk_walk.hpp). The cache slot's first
// page and the valid pages per outer slice are runtime values derived from it on every dispatch (host scalars) or on
// device (metadata tensors).
struct ChipShardPageGeometry {
    uint32_t num_outer_slices = 0;
    uint32_t pages_per_outer_slice = 0;
    uint32_t pages_per_cache_slot = 0;          // input pages of one cache slot (selected-batch gathers)
    uint32_t local_gather_dim_length = 0;       // this chip's (input) length of the gather dim
    uint32_t gather_dim_elements_per_page = 1;  // tile height / width when gathering a tile dim, else 1
    uint32_t num_ranks = 0;
    uint32_t pages_per_kv_slab = 0;  // valid pages per outer slice per KV slab (prefix metadata path)
};

struct FabricAllGatherFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle fabric_link_worker_reader_kernel_id{};
        tt::tt_metal::KernelHandle fabric_link_worker_sender_kernel_id{};
        tt::tt_metal::KernelHandle local_copy_core_reader_kernel_id{};
        tt::tt_metal::KernelHandle local_copy_core_writer_kernel_id{};
        tt::tt_metal::GlobalSemaphore downstream_started_counter;
        tt::tt_metal::GlobalSemaphore shards_arrived_counter;
        CoreRangeSet fabric_link_worker_cores;  // the cores that use the two global counters
        ChipShardPageGeometry chip_shard_geometry;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const FabricAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const FabricAllGatherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const FabricAllGatherParams& operation_attributes,
        const FabricAllGatherInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::operations::experimental::fabric_all_gather
