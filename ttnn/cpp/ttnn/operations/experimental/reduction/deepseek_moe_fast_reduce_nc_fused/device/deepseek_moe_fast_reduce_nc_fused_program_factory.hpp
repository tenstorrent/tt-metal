// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "deepseek_moe_fast_reduce_nc_fused_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

// TODO(#42193): Migrate to ProgramDescriptor.  This factory follows the
// Contract-2 mesh-workload pattern (create_at + AdaptedCachedMeshWorkload +
// per-MeshCoordinate program) rather than the Contract-1 ProgramDescriptor
// pattern.  Per the descriptor-migration playbook, Contract-2 ops are
// out of scope for the current Contract-1 batch; framework support for
// mesh-workload descriptors is tracked separately.
struct DeepseekMoEFastReduceNCFusedMeshWorkloadFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        std::vector<tt::tt_metal::CoreCoord> all_cores;
        uint32_t ncores;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    // Per-device program construction. The framework iterates the mesh's coordinate range set and
    // assembles the MeshWorkload by calling this for each coordinate.
    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const DeepseekMoEFastReduceNCFusedParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const DeepseekMoEFastReduceNCFusedInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const DeepseekMoEFastReduceNCFusedParams& operation_attributes,
        const DeepseekMoEFastReduceNCFusedInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
