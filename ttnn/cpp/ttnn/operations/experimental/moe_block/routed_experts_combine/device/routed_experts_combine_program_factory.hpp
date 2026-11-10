// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/moe_block/common/args_table.hpp"
#include "ttnn/operations/experimental/moe_block/common/chip_program.hpp"

#include <array>
#include <vector>

#include "combine_program.hpp"
#include "routed_experts_combine_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/experimental/moe_block/flat_routed_expert/device/flat_program.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

// The flat routed experts and the sparse combine in ONE program per chip (every chip's program differs: its ports, its
// accumulators). The hash holds no address: on a cache hit every buffer address (the per-call arena included) is
// re-applied to the flat expert's and the combine's runtime args and arena CBs.
struct RoutedExpertsCombineFactory {
    struct shared_variables_t {
        CombineHandles combine;
        flat_routed_expert::FlatProgramHandles flat;
        std::array<uint32_t, 3> report{};  // (the arena, table and bar addresses the y writers' report holds)
        // the data-movement kernels' per-core args in DRAM (common/args_table.hpp runtime_args_to_dram_fixup): the
        // tables, and per converted kernel (handle, its common args' first index, its block's arena offset) - a cache
        // hit rewrites only those common args (the sources' addresses, the block's L1 address)
        std::shared_ptr<const void> args_tables;
        std::vector<std::array<uint32_t, 3>> args_kernels;
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
