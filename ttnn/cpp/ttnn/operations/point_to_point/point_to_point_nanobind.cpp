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
        R"doc(point_to_point(input_tensor: ttnn.Tensor, send_coord: ttnn.MeshCoordinate, receive_coord: ttnn.MeshCoordinate, topology: ttnn.Topology, receiver_semaphore: ttnn.GlobalSemaphore) -> ttnn.Tensor

            Point-to-point send receive Op. Send a tensor from one device to another.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                send_coord (ttnn.MeshCoordinate): Coordinate of device containing input_tensor (shard).
                receive_coord (ttnn.MeshCoordinate): Coordinate of device receiving input_tensor (shard).
                topology (ttnn.Topology): Fabric topology.

            Keyword Args:
                optional_output_tensoe (ttnn.Tensor,optional): Optional output tensor.

           Returns:
               ttnn.Tensor: the output tensor, with transferred shard on receiving device.

            Example:

                >>> input_tensor_torch = torch.zeros((2,1,1,16), dtype=dtype)
                >>> input_tensor_torch[0, :, :, :] = data # arbitary data in one shard

                >>> input_tensor = ttnn.from_torch(
                >>>     input_tensor_torch, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
                >>> )
                >>> sender_coord, receiver_coord = (ttnn.MeshCoordinate(c) for c in ((0,0), (0,1)))
                >>> sent_tensor = ttnn.point_to_point(
                        input_tensor,
                        receiver_coord,
                        sender_coord,
                        ttnn.Topology.Linear,
                >>>  sent_tensor_torch = ttnn.to_torch(
                >>>      sent_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
                >>>  )
                >>> assert sent_tensor_torch[1,:,:,:] == input_tensor_torch[0,:,:,:]
            )doc";

    using OperationType = decltype(ttnn::point_to_point);
    ttnn::bind_registered_operation(
        mod,
        ttnn::point_to_point,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const MeshCoordinate& receiver_coord,
               const MeshCoordinate& sender_coord,
               const ccl::Topology topology,
               const std::optional<ttnn::Tensor>& optional_output_tensor) {
                return self(input_tensor, receiver_coord, sender_coord, topology, optional_output_tensor);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("receiver_coord"),
            nb::arg("sender_coord"),
            nb::arg("topology"),
            nb::kw_only(),
            nb::arg("optional_output_tensor") = nb::none()});
}
}  // namespace ttnn::operations::point_to_point
