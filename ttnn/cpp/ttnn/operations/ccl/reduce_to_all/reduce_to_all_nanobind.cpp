// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include "reduce_to_all_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "reduce_to_all.hpp"

namespace ttnn::operations::ccl {

void bind_reduce_to_all(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Reduce-to-all operation. Performs sdpa tree reduction across 4 devices and stores the output on all devices.

            Args:
                input_tensor_l: the input tensor is a vector of values l of SDPA.
                input_tensor_ms: the combined MS tensor with max in col 0 and sum in col 1.

            Keyword Args:
                topology (ttnn.Topology): Fabric topology.
                output_tensor_l (ttnn.Tensor, optional): Optional output tensor for normalized L values.
                fw_intermediate_tensor (ttnn.Tensor, optional): Optional fw intermediate tensor.
                bw_intermediate_tensor (ttnn.Tensor, optional): Optional bw intermediate tensor.
                input_forwarder_cores (List[ttnn.CoreCoord], optional): List of forwarder cores

           Returns:
                ttnn.Tensor output_tensor_l: the normalized output tensor for values (L/S).

            Example:

                >>> input_tensor_torch_l = torch.zeros((8,128), dtype=dtype)
                >>> input_tensor_torch_ms = torch.zeros((8,32), dtype=dtype)  # col 0 = max, col 1 = sum

                >>> input_tensor_l = ttnn.from_torch(
                >>>     input_tensor_torch_l, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
                >>> )
                >>> input_tensor_ms = ttnn.from_torch(
                >>>     input_tensor_torch_ms, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
                >>> )
                >>> output_tensor_l = ttnn.reduce_to_all(
                        input_tensor_l,
                        input_tensor_ms,
                        scale_fp32=1.0,
                        topology=ttnn.Topology.Ring)
            )doc";

    using OperationType = decltype(ttnn::reduce_to_all);
    ttnn::bind_registered_operation(
        mod,
        ttnn::reduce_to_all,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_l,
               const ttnn::Tensor& input_tensor_ms,
               const float scale_fp32,
               const std::optional<ttnn::Tensor>& output_tensor_l,
               const std::optional<ttnn::Tensor>& fw_intermediate_tensor,
               const std::optional<ttnn::Tensor>& bw_intermediate_tensor,
               const std::optional<std::vector<ttnn::CoreCoord>>& input_forwarder_cores,
               const std::optional<ttnn::Tensor>& forwarder_scratch_tensor,
               const tt::tt_fabric::Topology topology) {
                return self(
                    input_tensor_l,
                    input_tensor_ms,
                    scale_fp32,
                    topology,
                    output_tensor_l,
                    fw_intermediate_tensor,
                    bw_intermediate_tensor,
                    input_forwarder_cores,
                    forwarder_scratch_tensor);
            },
            nb::arg("input_tensor_l").noconvert(),
            nb::arg("input_tensor_ms").noconvert(),
            nb::kw_only(),
            nb::arg("scale_fp32") = 1.0f,
            nb::arg("output_tensor_l") = nb::none(),
            nb::arg("fw_intermediate_tensor") = nb::none(),
            nb::arg("bw_intermediate_tensor") = nb::none(),
            nb::arg("input_forwarder_cores") = nb::none(),
            nb::arg("forwarder_scratch_tensor") = nb::none(),
            nb::arg("topology").noconvert() = nb::cast(tt::tt_fabric::Topology::Ring)});

    mod.def(
        "reduce_to_all_tensor_spec",
        reduce_to_all_tensor_spec,
        nb::arg("input_tensor_l"),
        nb::arg("input_tensor_ms"),
        nb::arg("scale_fp32"),
        nb::arg("topology"),
        nb::arg("input_forwarder_cores") = nb::none());
}
}  // namespace ttnn::operations::ccl
