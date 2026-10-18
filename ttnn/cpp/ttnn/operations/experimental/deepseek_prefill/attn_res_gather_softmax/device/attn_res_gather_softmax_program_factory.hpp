// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "attn_res_gather_softmax_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

struct AttnResGatherSoftmaxSiteOffsets {
    uint32_t shift;
    uint32_t mass;
    uint32_t partial;
};

AttnResGatherSoftmaxSiteOffsets attn_res_gather_softmax_site_offsets(
    const AttnResGatherSoftmaxParams& operation_attributes, const AttnResGatherSoftmaxInputs& tensor_args);

struct AttnResGatherSoftmaxSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::KernelHandle gather_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> fold_cores;
    tt::tt_metal::CoreCoord gather_core;
};

struct AttnResGatherSoftmaxMeshWorkloadFactory {
    using shared_variables_t = AttnResGatherSoftmaxSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AttnResGatherSoftmaxParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AttnResGatherSoftmaxInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AttnResGatherSoftmaxParams& operation_attributes,
        const AttnResGatherSoftmaxInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    // Per mesh coordinate rather than once for the mesh: a chip's rank on the
    // tensor-parallel axis decides which slot of the statistics tensor it fills and
    // which way each peer lies on the fabric.
    static cached_program_t create_at(
        const AttnResGatherSoftmaxParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const AttnResGatherSoftmaxInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
