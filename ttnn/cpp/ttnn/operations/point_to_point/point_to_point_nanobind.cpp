// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include "point_to_point_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "point_to_point.hpp"

namespace ttnn::operations::point_to_point {

void bind_point_to_point(nb::module_& mod) {
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

           Returns:
               ttnn.Tensor: the output tensor, with transferred shard on receiving device.

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
        mod,
        ttnn::point_to_point,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const MeshCoordinate& sender_coord,
               const MeshCoordinate& receiver_coord,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<ttnn::Tensor>& intermediate_tensor,
               const ccl::Topology topology) {
                return self(input_tensor, receiver_coord, sender_coord, topology, output_tensor, intermediate_tensor);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("sender_coord"),
            nb::arg("receiver_coord"),
            nb::kw_only(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("intermediate_tensor") = nb::none(),
            nb::arg("topology").noconvert() = ccl::Topology::Linear});
    mod.def(
        "p2p_compute_intermediate_tensor_spec",
        p2p_compute_intermediate_tensor_spec,
        nb::arg("input_tensor"),
        nb::arg("sender_coord"),
        nb::arg("receiver_coord"),
        nb::arg("topology"));
}
}  // namespace ttnn::operations::point_to_point
