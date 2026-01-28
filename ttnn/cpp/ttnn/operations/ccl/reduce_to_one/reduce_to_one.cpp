// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/decorators.hpp"

#include "device/reduce_to_one_op.hpp"
#include "reduce_to_one.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteReduceToOne::invoke(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& exit_coord,
    const tt::tt_fabric::Topology topology,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<std::vector<ttnn::Tensor>>& optional_intermediate_tensors) {
    auto result = ttnn::prim::reduce_to_one(
        input_tensor, topology, root_coord, exit_coord, optional_output_tensor, optional_intermediate_tensors);
    // Return the final output tensor from result[1][0]
    return result[1][0];
}

}  // namespace ttnn::operations::ccl
