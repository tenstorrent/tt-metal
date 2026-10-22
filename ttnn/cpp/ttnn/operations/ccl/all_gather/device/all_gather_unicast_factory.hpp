// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_gather_device_operation_types.hpp"
#include "all_gather_program.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::ccl {

// Store-and-forward all_gather over one mesh axis (line or ring; Fabric 1D or 2D).
struct AllGatherUnicastFactory {
    using shared_variables_t = all_gather_program::SharedVariables;
    using cached_mesh_workload_t = all_gather_program::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherParams& operation_attributes,
        const AllGatherInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::operations::ccl
