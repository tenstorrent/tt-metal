// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "attn_res_gather_merge_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

struct AttnResGatherMergeSiteOffsets {
    uint32_t shift;
    uint32_t mass;
    uint32_t partial;
};

AttnResGatherMergeSiteOffsets attn_res_gather_merge_site_offsets(
    const AttnResGatherMergeParams& operation_attributes, const AttnResGatherMergeInputs& tensor_args);

struct AttnResGatherMergeSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::KernelHandle gather_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> fold_cores;
    tt::tt_metal::CoreCoord gather_core;
};

struct AttnResGatherMergeMeshWorkloadFactory {
    using shared_variables_t = AttnResGatherMergeSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AttnResGatherMergeParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AttnResGatherMergeInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AttnResGatherMergeParams& operation_attributes,
        const AttnResGatherMergeInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    // Per mesh coordinate rather than once for the mesh: a chip's rank on the
    // tensor-parallel axis decides which slot of the statistics tensor it fills and
    // which way each peer lies on the fabric.
    static cached_program_t create_at(
        const AttnResGatherMergeParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const AttnResGatherMergeInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
