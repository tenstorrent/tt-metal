// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim::matmul_new_detail {

// MeshWorkloadFactoryConcept factory that uses ProgramDescriptor for clean
// program construction (cache miss) while having efficient runtime-arg
// updates (cache hit) that only touch buffer addresses.
struct MultiCoreDescriptorFactory {
    struct shared_variables_t {
        uint32_t num_cores{};
        uint32_t num_cores_y{};
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MatmulParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    // Internal: build the declarative program descriptor
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim::matmul_new_detail
