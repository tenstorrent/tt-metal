// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#pragma once
#include <tt-metalium/mesh_coord.hpp>

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteDeepseekB1ReduceToOne {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const MeshCoordinate& root_coord,
        const MeshCoordinate& exit_coord,
        tt::tt_fabric::Topology topology,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<std::vector<ttnn::Tensor>>& optional_intermediate_tensors = std::nullopt);
};

}  // namespace operations::experimental::ccl

constexpr auto deepseek_b1_reduce_to_one = ttnn::register_operation<
    "ttnn::experimental::deepseek_b1_reduce_to_one",
    ttnn::operations::experimental::ccl::ExecuteDeepseekB1ReduceToOne>();

}  // namespace ttnn
