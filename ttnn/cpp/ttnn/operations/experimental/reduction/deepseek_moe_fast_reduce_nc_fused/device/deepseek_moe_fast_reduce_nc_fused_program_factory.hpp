// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/workload_descriptor.hpp>

#include "deepseek_moe_fast_reduce_nc_fused_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

// Contract-2 mesh-workload factory: builds one ProgramDescriptor per mesh
// coordinate (the per-coord reader RT args embed the (row, col) mesh
// coordinate, so a single replicated descriptor would not be correct).
struct DeepseekMoEFastReduceNCFusedMeshWorkloadFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const DeepseekMoEFastReduceNCFusedParams& operation_attributes,
        const DeepseekMoEFastReduceNCFusedInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
