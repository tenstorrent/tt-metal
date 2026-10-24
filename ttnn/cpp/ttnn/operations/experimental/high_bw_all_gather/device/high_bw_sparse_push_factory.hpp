// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "high_bw_all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::operations::experimental::high_bw_all_gather {

// Sparse KV push: the index-driven variant of the full-mesh gather. Instead of every chip receiving every row, each
// owner pushes only the rows a requester's top-k references, packed into the requester's compact buffer at the slots
// the plan op rewrote its indices to (see sparse_push_plan_device_operation.hpp and
// kernels/sparse_push_table_reader.cpp). sparse_sdpa then reads that buffer through the rewritten indices.
struct HighBwSparsePushFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const HighBwAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const HighBwAllGatherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const HighBwAllGatherParams& operation_attributes,
        const HighBwAllGatherInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::operations::experimental::high_bw_all_gather
