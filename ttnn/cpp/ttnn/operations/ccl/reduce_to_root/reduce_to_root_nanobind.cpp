// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include "reduce_to_root_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "reduce_to_root.hpp"

namespace ttnn::operations::ccl {

void bind_reduce_to_root(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Reduce-to-root operation. Performs sdpa tree reduction across 4 devices and stores the output on the root device only.

            Args:
                input_tensor_l: the input tensor is a vector of values l of SDPA.
                input_tensor_s: the input tensor is a vector of state s of SDPA.
                input_tensor_m: the input tensor is a vector of state m of SDPA.
                root_coord (ttnn.MeshCoordinate): Coordinate of the root device. Should be (1,0) for 4 devices setup.

            Keyword Args:
                topology (ttnn.Topology): Fabric topology.
                output_tensor (ttnn.Tensor,optional): Optional output tensor.
                intermediate_tensor (ttnn.Tensor,optional): Optional intermediate tensor.

           Returns:
               ttnn.Tensor output_tensor_l: the output tensor for values.
                ttnn.Tensor output_tensor_s: the output tensor for sum.
                ttnn.Tensor output_tensor_m: the output tensor for max.

            Example:

                >>> input_tensor_torch_l = torch.zeros((8,128), dtype=dtype)
                >>> input_tensor_torch_s = torch.zeros((8,32), dtype=dtype)
                >>> input_tensor_torch_m = torch.zeros((8,32), dtype=dtype)

                >>> input_tensor_l = ttnn.from_torch(
                >>>     input_tensor_torch_l, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
                >>> )
                >>> input_tensor_s = ttnn.from_torch(
                >>>     input_tensor_torch_s, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
                >>> )
                >>> input_tensor_m = ttnn.from_torch(
                >>>     input_tensor_torch_m, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
                >>> )
                >>> root_coord= ttnn.MeshCoordinate((1,0))
                >>> output_tensor_l, output_tensor_s, output_tensor_m = ttnn.reduce_to_root(
                        input_tensor_l,
                        input_tensor_s,
                        input_tensor_m,
                        root_coord,
                        scale_fp32=1.0,
                        topology=ttnn.Topology.Linear)
            )doc";

    using OperationType = decltype(ttnn::reduce_to_root);
    ttnn::bind_registered_operation(
        mod,
        ttnn::reduce_to_root,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_l,
               const ttnn::Tensor& input_tensor_s,
               const ttnn::Tensor& input_tensor_m,
               const MeshCoordinate& root_coord,
               const float scale_fp32,
               const std::optional<ttnn::Tensor>& output_tensor_l,
               const std::optional<ttnn::Tensor>& output_tensor_s,
               const std::optional<ttnn::Tensor>& output_tensor_m,
               const std::optional<ttnn::Tensor>& intermediate_tensor,
               const std::optional<std::vector<ttnn::CoreCoord>>& input_mux_cores,
               const tt::tt_fabric::Topology topology) {
                return self(
                    input_tensor_l,
                    input_tensor_s,
                    input_tensor_m,
                    root_coord,
                    scale_fp32,
                    topology,
                    output_tensor_l,
                    output_tensor_s,
                    output_tensor_m,
                    intermediate_tensor,
                    input_mux_cores);
            },
            nb::arg("input_tensor_l").noconvert(),
            nb::arg("input_tensor_s").noconvert(),
            nb::arg("input_tensor_m").noconvert(),
            nb::arg("root_coord"),
            nb::kw_only(),
            nb::arg("scale_fp32") = 1.0f,
            nb::arg("output_tensor_l") = nb::none(),
            nb::arg("output_tensor_s") = nb::none(),
            nb::arg("output_tensor_m") = nb::none(),
            nb::arg("intermediate_tensor") = nb::none(),
            nb::arg("input_mux_cores") = nb::none(),
            nb::arg("topology").noconvert() = nb::cast(tt::tt_fabric::Topology::Linear)});

    mod.def(
        "reduce_to_root_tensor_spec",
        reduce_to_root_tensor_spec,
        nb::arg("input_tensor_l"),
        nb::arg("input_tensor_s"),
        nb::arg("input_tensor_m"),
        nb::arg("root_coord"),
        nb::arg("scale_fp32"),
        nb::arg("topology"),
        nb::arg("input_mux_cores") = nb::none());
}
}  // namespace ttnn::operations::ccl
