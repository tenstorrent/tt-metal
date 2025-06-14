// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include <pybind11/pybind11.h>

#include "point_to_point.hpp"
#include "point_to_point_pybind.hpp"

namespace ttnn::operations::point_to_point {

namespace py = pybind11;

void py_bind_point_to_point(py::module& module) {
    auto doc =
        R"doc(point_to_point(input_tensor: ttnn.Tensor, send_coord: ttnn.MeshCoordinate, receive_coord: ttnn.MeshCoordinate, topology: ttnn.Topology, receiver_semaphore: ttnn.GlobalSemaphore) -> ttnn.Tensor

            Point-to-point send receive Op. Send a tensor from one device to another.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                send_coord (ttnn.MeshCoordinate): Coordinate of device containing input_tensor (shard).
                receive_coord (ttnn.MeshCoordinate): Coordinate of device receiving input_tensor (shard).
                topology (ttnn.Topology): Fabric topology.
                receiver_semaphore (ttnn.GlobalSemaphore): Semaphore allocated on receiving device

            Keyword Args:
                queue_id (int, optional): command queue id. Defaults to `0`.

           Returns:
               ttnn.Tensor: the output tensor, with transferred shard on receiving device.

            Example:

                >>> sent_tensor = ttnn.point_to_point(
                        input_tensor,
                        coord0,
                        coord1,
                        ttnn.Topology.Linear,
                        receiver_semaphore))doc";

    using OperationType = decltype(ttnn::point_to_point);
    ttnn::bind_registered_operation(
        module,
        ttnn::point_to_point,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const MeshCoordinate& send_coord,
               const MeshCoordinate& receive_coord,
               const ccl::Topology topology,
               const GlobalSemaphore& receiver_semaphore,
               QueueId queue_id) {
                return self(queue_id, input_tensor, send_coord, receive_coord, topology, receiver_semaphore);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("send_coord"),
            py::arg("receive_coord"),
            py::arg("topology"),
            py::arg("receiver_semaphore"),
            py::kw_only(),
            py::arg("queue_id") = DefaultQueueId,
        });
}
}  // namespace ttnn::operations::point_to_point
