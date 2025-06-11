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
        R"doc(
        ! TODO
        )doc";

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
