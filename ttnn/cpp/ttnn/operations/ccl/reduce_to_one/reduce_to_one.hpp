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

struct ExecuteReduceToOne {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const MeshCoordinate& root_coord,
        const MeshCoordinate& exit_coord,
        tt::tt_fabric::Topology topology,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<std::vector<ttnn::Tensor>>& optional_intermediate_tensors = std::nullopt);
};

}  // namespace operations::ccl

constexpr auto reduce_to_one =
    ttnn::register_operation<"ttnn::reduce_to_one", ttnn::operations::ccl::ExecuteReduceToOne>();

}  // namespace ttnn
