// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "device/reduce_to_root_op.hpp"
#include "reduce_to_root.hpp"

namespace ttnn::operations::ccl {

std::vector<ttnn::Tensor> ExecuteReduceToRoot::invoke(
    const ttnn::Tensor& input_tensor_l,
    const ttnn::Tensor& input_tensor_s,
    const ttnn::Tensor& input_tensor_m,
    const MeshCoordinate& root_coord,
    const float scale_fp32,
    const tt::tt_fabric::Topology topology,
    const std::optional<ttnn::Tensor>& optional_output_tensor_l,
    const std::optional<ttnn::Tensor>& optional_output_tensor_s,
    const std::optional<ttnn::Tensor>& optional_output_tensor_m,
    const std::optional<ttnn::Tensor>& optional_intermediate_tensor,
    const std::optional<std::vector<ttnn::CoreCoord>>& input_mux_cores) {
    // first output tensor in list is intermediate and is discarded
    return ttnn::prim::reduce_to_root(
               input_tensor_l,
               input_tensor_s,
               input_tensor_m,
               topology,
               root_coord,
               scale_fp32,
               optional_output_tensor_l,
               optional_output_tensor_s,
               optional_output_tensor_m,
               optional_intermediate_tensor,
               input_mux_cores)
        .at(1);
}

std::vector<ttnn::TensorSpec> reduce_to_root_tensor_spec(
    const ttnn::Tensor& input_tensor_l,
    const ttnn::Tensor& input_tensor_s,
    const ttnn::Tensor& input_tensor_m,
    const MeshCoordinate& root_coord,
    const float scale_fp32,
    const tt::tt_fabric::Topology topology,
    const std::optional<std::vector<ttnn::CoreCoord>>& input_mux_cores) {
    ReduceToRootOp::operation_attributes_t attrs{
        root_coord,
        scale_fp32,
        topology,
        input_mux_cores,
        {input_tensor_l.tensor_spec(), input_tensor_s.tensor_spec(), input_tensor_m.tensor_spec()}};
    ReduceToRootOp::tensor_args_t tensors{input_tensor_l, input_tensor_s, input_tensor_m, std::nullopt};

    return ReduceToRootOp::compute_output_specs(attrs, tensors).at(0);
}

}  // namespace ttnn::operations::ccl
