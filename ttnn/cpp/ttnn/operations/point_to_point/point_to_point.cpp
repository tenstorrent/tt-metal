// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "ttnn/decorators.hpp"

#include "device/host/point_to_point_device_op.hpp"
#include "point_to_point.hpp"

namespace ttnn::operations::point_to_point {

ttnn::Tensor ExecutePointToPoint::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& send_coord,
    const MeshCoordinate& receive_coord,
    const ccl::Topology topology,
    const GlobalSemaphore& receiver_semaphore) {
    // first output tensor in list is intermediate and is discarded
    return ttnn::prim::point_to_point(input_tensor, topology, send_coord, receive_coord, receiver_semaphore).at(1);
}

}  // namespace ttnn::operations::point_to_point
