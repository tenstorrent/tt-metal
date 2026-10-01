// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::operations::experimental::fabric_all_gather {

// Shard layout of one cached program (terms: kernels/fabric_all_gather_chunk_walk.hpp). The slot base page and
// the active stripe pages are runtime values derived from it on every dispatch (host scalars) or on device
// (metadata tensors).
struct ShardPageGeometry {
    uint32_t num_stripes = 0;
    uint32_t stripe_pages = 0;
    uint32_t pages_per_slot = 0;         // input pages of one batch slot (selected-batch gathers)
    uint32_t local_gather_dim_size = 0;  // this chip's (input) extent of the gather dim
    uint32_t num_ranks = 0;
    uint32_t pages_per_slab = 0;  // active pages per stripe per block-cyclic slab (prefix metadata path)
};

struct FabricAllGatherFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle link_worker_reader_kernel_id{};
        tt::tt_metal::KernelHandle link_worker_sender_kernel_id{};
        tt::tt_metal::KernelHandle copy_core_reader_kernel_id{};
        tt::tt_metal::KernelHandle copy_core_writer_kernel_id{};
        bool has_copy_cores = false;
        tt::tt_metal::GlobalSemaphore ready_counter;
        tt::tt_metal::GlobalSemaphore arrival_counter;
        CoreRangeSet all_op_cores;
        ShardPageGeometry shard_page_geometry;
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
