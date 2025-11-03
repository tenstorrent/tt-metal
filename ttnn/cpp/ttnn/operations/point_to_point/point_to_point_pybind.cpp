// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include <pybind11/pybind11.h>

#include "ttnn-pybind/decorators.hpp"

#include "point_to_point.hpp"
#include "point_to_point_pybind.hpp"

namespace ttnn::operations::point_to_point {

namespace py = pybind11;

void py_bind_point_to_point(py::module& module) {
    auto doc =
        R"doc(
            Point-to-point send and receive operation. Send a tensor from one device to another.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                sender_coord (ttnn.MeshCoordinate): Coordinate of device containing input_tensor (shard).
                receiver_coord (ttnn.MeshCoordinate): Coordinate of device receiving input_tensor (shard).

            Keyword Args:
                topology (ttnn.Topology): Fabric topology.
                output_tensor (ttnn.Tensor,optional): Optional output tensor.
                intermediate_tensor (ttnn.Tensor,optional): Optional intermediate tensor.

            Example:

                >>> input_tensor_torch = torch.zeros((2,1,1,16), dtype=dtype)
                >>> input_tensor_torch[0, :, :, :] = data # arbitary data in one shard

                >>> input_tensor = ttnn.from_torch(
                >>>     input_tensor_torch, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
                >>> )
                >>> sender_coord, receiver_coord = ttnn.MeshCoordinate((0,0)), ttnn.MeshCoordinate((0,1))
                >>> sent_tensor = ttnn.point_to_point(
                        input_tensor,
                        sender_coord,
                        receiver_coord,
                        topology=ttnn.Topology.Linear)
            )doc";

    using OperationType = decltype(ttnn::point_to_point);
    ttnn::bind_registered_operation(
        module,
        ttnn::point_to_point,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const MeshCoordinate& sender_coord,
               const MeshCoordinate& receiver_coord,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<ttnn::Tensor>& intermediate_tensor,
               const ccl::Topology topology) {
                return self(input_tensor, receiver_coord, sender_coord, topology, output_tensor, intermediate_tensor);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("sender_coord"),
            py::arg("receiver_coord"),
            py::kw_only(),
            py::arg("output_tensor") = std::nullopt,
            py::arg("intermediate_tensor") = std::nullopt,
            py::arg("topology").noconvert() = ccl::Topology::Linear});
    module.def(
        "p2p_compute_intermediate_tensor_spec",
        p2p_compute_intermediate_tensor_spec,
        py::arg("input_tensor"),
        py::arg("sender_coord"),
        py::arg("receiver_coord"),
        py::arg("topology"));
}
}  // namespace ttnn::operations::point_to_point
