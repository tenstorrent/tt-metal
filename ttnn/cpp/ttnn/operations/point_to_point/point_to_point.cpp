// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include "device/host/point_to_point_device_op.hpp"
#include "point_to_point.hpp"

namespace ttnn {

ttnn::Tensor point_to_point(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& receiver_coord,
    const MeshCoordinate& sender_coord,
    const ::ttnn::ccl::Topology topology,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<ttnn::Tensor>& optional_intermediate_tensor) {
    // Same-device transfer whose output aliases the input is a pure no-op — return the output
    // directly WITHOUT entering the device op. Otherwise it registers a program-cache entry
    // whose writer buffer aliases the input; a later same-shape call with a *distinct*
    // preallocated output would, on cache hit, patch the writer back to the input slot and
    // leave the real output stale (issue #28945 audit). This also matches the ttnn no-op
    // convention (data_movement move/slice return the input for no-ops).
    if (sender_coord == receiver_coord && optional_output_tensor.has_value() &&
        input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE &&
        optional_output_tensor->storage_type() == tt::tt_metal::StorageType::DEVICE &&
        optional_output_tensor->buffer() == input_tensor.buffer()) {
        // Run the op's own validation so an invalid alias call (sharded, out-of-mesh coord,
        // spec/layout/alignment mismatch) is rejected exactly as the device op would — then
        // return the output without entering the op, which would otherwise create a program-
        // cache entry whose writer aliases the input (#28945 audit). validate() runs before
        // launch(), so an invalid call throws before any cache entry is created.
        using OpT = ttnn::operations::point_to_point::PointToPointOp;
        OpT::validate_on_program_cache_miss(
            OpT::operation_attributes_t{receiver_coord, sender_coord, topology, input_tensor.tensor_spec()},
            OpT::tensor_args_t{input_tensor, optional_output_tensor, optional_intermediate_tensor});
        return optional_output_tensor.value();
    }

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

namespace operations::point_to_point {

ttnn::TensorSpec p2p_compute_intermediate_tensor_spec(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& receiver_coord,
    const MeshCoordinate& sender_coord,
    const ::ttnn::ccl::Topology topology) {
    PointToPointOp::operation_attributes_t attrs{receiver_coord, sender_coord, topology, input_tensor.tensor_spec()};
    PointToPointOp::tensor_args_t tensors{input_tensor, std::nullopt, std::nullopt};

    return PointToPointOp::compute_output_specs(attrs, tensors).at(0);
}

}  // namespace operations::point_to_point

}  // namespace ttnn
