// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim {

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory {
    // one program per chip (each with its own DRAM-bank reader placement)
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim
