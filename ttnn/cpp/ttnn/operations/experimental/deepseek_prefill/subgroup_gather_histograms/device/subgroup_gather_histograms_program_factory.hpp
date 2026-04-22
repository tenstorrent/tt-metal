// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "subgroup_gather_histograms_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms {

struct SubgroupGatherHistogramsSharedVariables {
    tt::tt_metal::KernelHandle kernel_id = 0;
    std::vector<CoreCoord> cores;
    GlobalSemaphore init_semaphore;
};

struct SubgroupGatherHistogramsProgramFactory {
    using shared_variables_t = SubgroupGatherHistogramsSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;
    using tensor_return_value_t = ttnn::Tensor;

    static cached_mesh_workload_t create_mesh_workload(
        const SubgroupGatherHistogramsParams& operation_attributes,
        const ttnn::distributed::MeshCoordinateRangeSet& tensor_coords,
        const SubgroupGatherHistogramsInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const SubgroupGatherHistogramsParams& operation_attributes,
        const ttnn::distributed::MeshCoordinate& mesh_coordinate,
        const SubgroupGatherHistogramsInputs& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::distributed::MeshCoordinateRange& subgroup_range,
        const GlobalSemaphore& init_semaphore);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const SubgroupGatherHistogramsParams& operation_attributes,
        const SubgroupGatherHistogramsInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms
