// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllToAllDispatchSelectiveTilize {
    static std::array<ttnn::Tensor, 3> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& expert_indices_tensor,
        const ttnn::Tensor& expert_scores_tensor,
        const ttnn::Tensor& expert_mapping_tensor,
        std::optional<uint32_t> axis,
        std::optional<uint32_t> num_links,
        std::optional<tt::tt_fabric::Topology> topology,
        uint32_t tokens_per_chunk = 32,
        const std::optional<CoreRangeSet>& all_to_all_dispatch_core_range_set = std::nullopt,
        const std::optional<CoreRangeSet>& selective_tilize_core_range_set = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {
constexpr auto all_to_all_dispatch_selective_tilize = ttnn::register_operation<
    "ttnn::experimental::all_to_all_dispatch_selective_tilize",
    ttnn::operations::experimental::ccl::ExecuteAllToAllDispatchSelectiveTilize>();
}  // namespace experimental

}  // namespace ttnn
