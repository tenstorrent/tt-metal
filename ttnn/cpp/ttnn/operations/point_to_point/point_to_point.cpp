// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "device/host/point_to_point_device_op.hpp"
#include "point_to_point.hpp"

namespace ttnn::operations::point_to_point {

ttnn::Tensor ExecutePointToPoint::invoke(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& receiver_coord,
    const MeshCoordinate& sender_coord,
    const ccl::Topology topology,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<ttnn::Tensor>& optional_intermediate_tensor) {
    // first output tensor in list is intermediate and is discarded
    return ttnn::prim::point_to_point(
               input_tensor,
               topology,
               receiver_coord,
               sender_coord,
               optional_output_tensor,
               optional_intermediate_tensor)
        .at(1);
}

ttnn::TensorSpec p2p_compute_intermediate_tensor_spec(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& receiver_coord,
    const MeshCoordinate& sender_coord,
    const ccl::Topology topology) {
    PointToPointOp::operation_attributes_t attrs{receiver_coord, sender_coord, topology, input_tensor.tensor_spec()};
    PointToPointOp::tensor_args_t tensors{input_tensor, std::nullopt, std::nullopt};

    return PointToPointOp::compute_output_specs(attrs, tensors).at(0);
}

}  // namespace ttnn::operations::point_to_point
