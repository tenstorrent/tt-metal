// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "combine_program.hpp"
#include "routed_experts_combine_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "../../flat_routed_expert/device/flat_program.hpp"

namespace ttnn::operations::bringup::moe_block_main::routed_experts_combine {

// The flat routed experts and the sparse combine in ONE program per chip (every chip's program differs: its ports, its
// accumulators). The hash holds no address: on a cache hit every buffer address (the per-call arena included) is
// re-applied to the flat expert's and the combine's runtime args and arena CBs.
struct RoutedExpertsCombineFactory {
    struct shared_variables_t {
        CombineHandles combine;
        flat_routed_expert::FlatProgramHandles flat;
        std::array<uint32_t, 3> report{};  // (the arena, table and bar addresses the y writers' report holds)
        ChipCoord chip{};                  // (this program's chip: MOE_FILL's per-call args)
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

}  // namespace ttnn::operations::bringup::moe_block_main::routed_experts_combine
