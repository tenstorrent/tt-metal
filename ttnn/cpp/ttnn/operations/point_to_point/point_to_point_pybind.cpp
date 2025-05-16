

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
               const tt::tt_metal::distributed::MeshCoordinate& receive_coord,
               const ccl::Topology topology,
               MeshDevice& mesh_device,
               const GlobalSemaphore& receiver_semaphore,
               QueueId queue_id) {
                return self(queue_id, input_tensor, receive_coord, topology, mesh_device, receiver_semaphore);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("receive_coord"),
            py::arg("topology"),
            py::arg("mesh_device"),
            py::arg("receiver_semaphore"),
            py::kw_only(),
            py::arg("queue_id") = DefaultQueueId,
        });
}
}  // namespace ttnn::operations::point_to_point
