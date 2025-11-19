// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/decorators.hpp"

#include "device/reduce_to_root_op.hpp"
#include "reduce_to_root.hpp"

namespace ttnn::operations::ccl {

std::vector<ttnn::Tensor> ExecuteReduceToRoot::invoke(
    const std::vector<ttnn::Tensor>& input_tensor,
    const MeshCoordinate& root_coord,
    const tt::tt_fabric::Topology topology,
    const std::optional<std::vector<ttnn::Tensor>>& optional_output_tensor,
    const std::optional<std::vector<ttnn::Tensor>>& optional_intermediate_tensor) {
    // first output tensor in list is intermediate and is discarded
    return ttnn::prim::reduce_to_root(
               input_tensor, topology, root_coord, optional_output_tensor, optional_intermediate_tensor)
        .at(1);
}

std::vector<ttnn::TensorSpec> reduce_to_root_compute_intermediate_tensor_spec(
    const std::vector<ttnn::Tensor>& input_tensor,
    const MeshCoordinate& root_coord,
    const tt::tt_fabric::Topology topology) {
    ReduceToRootOp::operation_attributes_t attrs{
        root_coord,
        topology,
        {input_tensor[0].tensor_spec(), input_tensor[1].tensor_spec(), input_tensor[2].tensor_spec()}};
    ReduceToRootOp::tensor_args_t tensors{input_tensor, std::nullopt, std::nullopt};

    return ReduceToRootOp::compute_output_specs(attrs, tensors).at(0);
}

}  // namespace ttnn::operations::ccl
