// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
namespace operations::ccl {

struct ExecuteReduceToRoot {
    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensor,
        const MeshCoordinate& root_coord,
        tt::tt_fabric::Topology topology,
        const std::optional<std::vector<ttnn::Tensor>>& optional_output_tensor = std::nullopt,
        const std::optional<std::vector<ttnn::Tensor>>& optional_intermediate_tensor = std::nullopt);
};

std::vector<ttnn::TensorSpec> reduce_to_root_compute_intermediate_tensor_spec(
    const std::vector<ttnn::Tensor>& input_tensor, const MeshCoordinate& root_coord, tt::tt_fabric::Topology topology);

}  // namespace operations::ccl

constexpr auto reduce_to_root =
    ttnn::register_operation<"ttnn::reduce_to_root", ttnn::operations::ccl::ExecuteReduceToRoot>();

}  // namespace ttnn
