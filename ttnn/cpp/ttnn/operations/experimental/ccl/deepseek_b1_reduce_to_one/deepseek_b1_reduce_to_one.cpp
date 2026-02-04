// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/decorators.hpp"

#include "device/deepseek_b1_reduce_to_one_op.hpp"
#include "deepseek_b1_reduce_to_one.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteDeepseekB1ReduceToOne::invoke(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& exit_coord,
    const tt::tt_fabric::Topology topology,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<std::vector<ttnn::Tensor>>& optional_intermediate_tensors) {
    auto result = ttnn::prim::deepseek_b1_reduce_to_one(
        input_tensor, topology, root_coord, exit_coord, optional_output_tensor, optional_intermediate_tensors);
    // Return the final output tensor from result[1][0]
    return result[1][0];
}

}  // namespace ttnn::operations::experimental::ccl
