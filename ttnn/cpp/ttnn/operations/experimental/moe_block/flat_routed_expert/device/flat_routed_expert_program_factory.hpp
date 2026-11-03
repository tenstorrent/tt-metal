// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flat_program.hpp"
#include "flat_routed_expert_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

// The plain op: one program per chip (the flat fragment alone; the auto gate/up placements differ per chip,
// flat_routed_expert_chip_plan). Cache hits re-apply the buffer addresses (the arena / words are per-call buffers, so
// they move between calls).
struct FlatRoutedExpertProgramFactory {
    using shared_variables_t = FlatProgramHandles;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const FlatRoutedExpertParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const FlatRoutedExpertInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const FlatRoutedExpertParams& operation_attributes,
        const FlatRoutedExpertInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
