// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flat_program.hpp"
#include "ttnn/operations/experimental/moe_block/common/args_table.hpp"
#include "flat_routed_expert_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

// The plain op: one program per range (the flat fragment alone, its args plain runtime args). Every launch writes
// every buffer address and re-points the arena's CBs (common/args_table.hpp write_program_addresses).
struct FlatRoutedExpertProgramFactory {
    using shared_variables_t = ProgramAddresses;
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
