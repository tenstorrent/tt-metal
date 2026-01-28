// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/decorators.hpp"

#include "device/reduce_to_all_op.hpp"
#include "reduce_to_all.hpp"

namespace ttnn::operations::ccl {

std::vector<ttnn::Tensor> ExecuteReduceToAll::invoke(
    const ttnn::Tensor& input_tensor_l,
    const ttnn::Tensor& input_tensor_s,
    const ttnn::Tensor& input_tensor_m,
    const MeshCoordinate& root_coord,
    const float scale_fp32,
    const tt::tt_fabric::Topology topology,
    const std::optional<ttnn::Tensor>& optional_output_tensor_l,
    const std::optional<ttnn::Tensor>& optional_output_tensor_s,
    const std::optional<ttnn::Tensor>& optional_output_tensor_m,
    const std::optional<ttnn::Tensor>& optional_fw_intermediate_tensor,
    const std::optional<ttnn::Tensor>& optional_bw_intermediate_tensor,
    const std::optional<ttnn::Tensor>& optional_coord_intermediate_tensor,
    const std::optional<std::vector<ttnn::CoreCoord>>& input_mux_cores,
    const std::optional<std::vector<ttnn::CoreCoord>>& extra_worker_cores,
    const std::optional<ttnn::Tensor>& optional_aggregator_scratch_tensor) {
    // first output tensor in list is intermediate and is discarded
    return ttnn::prim::reduce_to_all(
               input_tensor_l,
               input_tensor_s,
               input_tensor_m,
               topology,
               root_coord,
               scale_fp32,
               optional_output_tensor_l,
               optional_output_tensor_s,
               optional_output_tensor_m,
               optional_fw_intermediate_tensor,
               optional_bw_intermediate_tensor,
               optional_coord_intermediate_tensor,
               input_mux_cores,
               extra_worker_cores,
               optional_aggregator_scratch_tensor)
        .at(1);
}

std::vector<ttnn::TensorSpec> reduce_to_all_tensor_spec(
    const ttnn::Tensor& input_tensor_l,
    const ttnn::Tensor& input_tensor_s,
    const ttnn::Tensor& input_tensor_m,
    const MeshCoordinate& root_coord,
    const float scale_fp32,
    const tt::tt_fabric::Topology topology,
    const std::optional<std::vector<ttnn::CoreCoord>>& input_mux_cores,
    const std::optional<std::vector<ttnn::CoreCoord>>& extra_worker_cores) {
    ReduceToAllOp::operation_attributes_t attrs{
        root_coord,
        scale_fp32,
        topology,
        input_mux_cores,
        extra_worker_cores,
        {input_tensor_l.tensor_spec(), input_tensor_s.tensor_spec(), input_tensor_m.tensor_spec()}};
    ReduceToAllOp::tensor_args_t tensors{input_tensor_l, input_tensor_s, input_tensor_m, std::nullopt};

    return ReduceToAllOp::compute_output_specs(attrs, tensors).at(0);
}

}  // namespace ttnn::operations::ccl
