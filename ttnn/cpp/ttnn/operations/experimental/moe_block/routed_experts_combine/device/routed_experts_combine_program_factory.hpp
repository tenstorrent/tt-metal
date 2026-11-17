// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/moe_block/common/args_table.hpp"
#include "ttnn/operations/experimental/moe_block/common/chip_program.hpp"

#include "combine_program.hpp"
#include "routed_experts_combine_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/experimental/moe_block/flat_routed_expert/device/flat_program.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

// The flat routed experts and the sparse combine in one program per chip, every data-movement kernel's per-core args
// in per-chip DRAM tables (common/args_table.hpp), so one program serves the mesh where the chips place their kernels
// alike. The hash holds no address: a cache hit writes every buffer address (the per-call arena included) into the
// kernels' common args and re-points the arena's CBs (write_program_addresses).
struct RoutedExpertsCombineFactory {
    struct shared_variables_t {
        ProgramAddresses addresses;  // what a cache hit writes (and the DRAM tables, kept with the program)
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const RoutedExpertsCombineParams& params,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const RoutedExpertsCombineInputs& inputs,
        ttnn::Tensor& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const RoutedExpertsCombineParams& params,
        const RoutedExpertsCombineInputs& inputs,
        ttnn::Tensor& output);
};

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
