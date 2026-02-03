// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/decorators.hpp"

#include "device/reduce_to_all_op.hpp"
#include "reduce_to_all.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteReduceToAll::invoke(
    const ttnn::Tensor& input_tensor_l,
    const ttnn::Tensor& input_tensor_ms,  // Combined: col 0 = max, col 1 = sum
    const float scale_fp32,
    const tt::tt_fabric::Topology topology,
    const std::optional<ttnn::Tensor>& optional_output_tensor_l,
    const std::optional<ttnn::Tensor>& optional_fw_intermediate_tensor,
    const std::optional<ttnn::Tensor>& optional_bw_intermediate_tensor,
    const std::optional<std::vector<ttnn::CoreCoord>>& input_forwarder_cores,
    const std::optional<ttnn::Tensor>& optional_forwarder_scratch_tensor) {
    // Returns normalized L tensor (L/S division is fused in compute kernel)
    // Extract final output tensor from [1][0] (second array = final outputs, first tensor = L)
    auto result = ttnn::prim::reduce_to_all(
        input_tensor_l,
        input_tensor_ms,
        topology,
        scale_fp32,
        optional_output_tensor_l,
        optional_fw_intermediate_tensor,
        optional_bw_intermediate_tensor,
        input_forwarder_cores,
        optional_forwarder_scratch_tensor);
    return result.at(1).at(0);  // Final outputs [1], normalized L [0]
}

ttnn::TensorSpec reduce_to_all_tensor_spec(
    const ttnn::Tensor& input_tensor_l,
    const ttnn::Tensor& input_tensor_ms,  // Combined: col 0 = max, col 1 = sum
    const float scale_fp32,
    const tt::tt_fabric::Topology topology,
    const std::optional<std::vector<ttnn::CoreCoord>>& input_forwarder_cores) {
    ReduceToAllOp::operation_attributes_t attrs{
        scale_fp32, topology, input_forwarder_cores, {input_tensor_l.tensor_spec(), input_tensor_ms.tensor_spec()}};
    ReduceToAllOp::tensor_args_t tensors{input_tensor_l, input_tensor_ms, std::nullopt};

    // Extract final output spec from [1][0] (second array = final outputs, first spec = L)
    return ReduceToAllOp::compute_output_specs(attrs, tensors).at(1).at(0);
}

}  // namespace ttnn::operations::ccl
