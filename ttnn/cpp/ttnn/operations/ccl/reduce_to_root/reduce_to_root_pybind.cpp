// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include <pybind11/pybind11.h>

#include "ttnn-pybind/decorators.hpp"

#include "reduce_to_root.hpp"
#include "reduce_to_root_pybind.hpp"

namespace ttnn::operations::ccl {

namespace py = pybind11;

void py_bind_reduce_to_root(py::module& module) {
    auto doc =
        R"doc(
            Reduce-to-root operation. Performs tree reduction across 4 devices and stores the output on the root device only.

            Args:
                input_tensor (std::vector<ttnn.Tensor>): the input tensor is a tuple for (m,s,l).
                root_coord (ttnn.MeshCoordinate): Coordinate of the root device. Should be (1,0) or (2,0)

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
                >>> sender_coord= ttnn.MeshCoordinate((1,0))
                >>> sent_tensor = ttnn.reduce_to_root(
                        input_tensor,
                        sender_coord,
                        topology=ttnn.Topology.Linear)
            )doc";

    using OperationType = decltype(ttnn::reduce_to_root);
    ttnn::bind_registered_operation(
        module,
        ttnn::reduce_to_root,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const std::vector<ttnn::Tensor>& input_tensor,
               const MeshCoordinate& root_coord,
               const std::optional<std::vector<ttnn::Tensor>>& output_tensor,
               const std::optional<std::vector<ttnn::Tensor>>& intermediate_tensor,
               const tt::tt_fabric::Topology topology) {
                return self(input_tensor, root_coord, topology, output_tensor, intermediate_tensor);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("root_coord"),
            py::kw_only(),
            py::arg("output_tensor") = std::nullopt,
            py::arg("intermediate_tensor") = std::nullopt,
            py::arg("topology").noconvert() = tt::tt_fabric::Topology::Linear});
    module.def(
        "reduce_to_root_compute_intermediate_tensor_spec",
        reduce_to_root_compute_intermediate_tensor_spec,
        py::arg("input_tensor"),
        py::arg("root_coord"),
        py::arg("topology"));
}
}  // namespace ttnn::operations::ccl
