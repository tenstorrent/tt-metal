// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gdn_fused_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <vector>

namespace ttnn::experimental::prim {

struct GdnFusedSharedVariables {
    // Per-group kernel handles (one group per distinct pair count)
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;

    // All core coordinates used, in order
    std::vector<tt::tt_metal::CoreCoord> cores;

    // Which reader kernel id each core belongs to (index into reader_kernel_ids)
    std::vector<uint32_t> core_to_reader_group;
};

struct GdnFusedProgramFactory {
    using shared_variables_t = GdnFusedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const GdnFusedParams& operation_attributes, const GdnFusedInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const GdnFusedParams& operation_attributes,
        const GdnFusedInputs& tensor_args,
        Tensor& tensor_return_value);
};

struct GdnFusedMeshWorkloadFactory {
    using shared_variables_t = GdnFusedSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const GdnFusedParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const GdnFusedInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const GdnFusedParams& operation_attributes,
        const GdnFusedInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
