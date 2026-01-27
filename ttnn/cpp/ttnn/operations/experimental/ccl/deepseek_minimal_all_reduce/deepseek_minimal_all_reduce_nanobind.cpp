// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_minimal_all_reduce_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/deepseek_minimal_all_reduce.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_deepseek_minimal_all_reduce_op(nb::module_ mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               std::optional<uint32_t> cluster_axis,
               const std::optional<ttnn::Tensor>& intermediate_tensor,
               const std::optional<ttnn::Tensor>& residual_tensor,
               const std::optional<ttnn::Tensor>& persistent_output_tensor,
               const uint32_t num_links,
               const tt::tt_fabric::Topology topology) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    num_links,
                    topology,
                    cluster_axis,
                    intermediate_tensor,
                    residual_tensor,
                    persistent_output_tensor);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("intermediate_tensor") = nb::none(),
            nb::arg("residual_tensor") = nb::none(),
            nb::arg("persistent_output_tensor") = nb::none(),
            nb::arg("num_links") = 2,
            nb::arg("topology") = nb::cast(tt::tt_fabric::Topology::Linear)});
}

}  // namespace

void bind_deepseek_minimal_all_reduce(nb::module_& mod) {
    bind_deepseek_minimal_all_reduce_op(
        mod,
        ttnn::experimental::deepseek_minimal_all_reduce,
        R"doc(
        Performs an all-reduce collective operation across devices in a mesh along the specified cluster axis.

        Args:
            input_tensor (ttnn.Tensor)
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the operation on.
            residual_tensor (ttnn.Tensor, optional): An optional tensor to be added to the input during the all-reduce operation for fused residual addition.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the all_reduce operation. Defaults to `2`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Defaults to `ttnn.Topology.Linear`.

        Returns:
            ttnn.Tensor of the output on the mesh device.

        Example:
            >>> input_tensor = torch.randn([1, 7168], dtype=torch.bfloat16)
            >>> num_devices = 2
            >>> num_links = 2
            >>> cluster_axis = 0
            >>> topology = ttnn.Topology.Linear
            >>> input_shape = (1, 7168)
            >>> input_dtype = ttnn.DType.BFLOAT16
            >>> layout = ttnn.TILE_LAYOUT
            >>> device_tensors = []
            >>> for device_idx in range(num_devices):
                    tensor = torch.rand(input_shape, dtype=torch.bfloat16)
                    device_tensors.append(tensor)
            >>> mesh_tensor_torch = torch.cat(device_tensors, dim=0)
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(num_devices, 1))
            >>> mesh_mapper_config = ttnn.MeshMapperConfig(
                    [ttnn.PlacementShard(0), ttnn.PlacementReplicate()], ttnn.MeshShape(num_devices, 1)
                )
            >>> ttnn_tensor = ttnn.from_torch(
                            mesh_tensor_torch,
                            dtype=input_dtype,
                            device=mesh_device,
                            layout=layout,
                            memory_config=mem_config,
                            mesh_mapper=ttnn.create_mesh_mapper(mesh_device,mesh_mapper_config))
            >>> output = ttnn.experimental.experimental.deepseek_minimal_all_reduce(ttnn_tensor, cluster_axis=cluster_axis, num_links=num_links, topology=topology)

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
