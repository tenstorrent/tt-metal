// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_nanobind.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_async/reduce_scatter.hpp"
#include "ttnn/types.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_reduce_scatter(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               const GlobalSemaphore& from_remote_multi_device_global_semaphore,
               const GlobalSemaphore& to_remote_multi_device_global_semaphore,
               ttnn::operations::reduction::ReduceType math_op,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology,
               const std::optional<size_t> num_links,
               std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    dim,
                    from_remote_multi_device_global_semaphore,
                    to_remote_multi_device_global_semaphore,
                    math_op,
                    memory_config,
                    topology,
                    num_links,
                    worker_subdevice_id_opt);
            },
            nb::arg("input_tensor"),
            nb::arg("dim"),
            nb::arg("from_remote_multi_device_global_semaphore"),
            nb::arg("to_remote_multi_device_global_semaphore"),
            nb::arg("math_op"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Linear,
            nb::arg("num_links") = nb::none(),
            nb::arg("subdevice_id") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const GlobalSemaphore& from_remote_multi_device_global_semaphore,
               const GlobalSemaphore& to_remote_multi_device_global_semaphore,
               const std::optional<std::vector<ttnn::Tensor>>& persistent_output_tensors,
               ttnn::operations::reduction::ReduceType math_op,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology,
               const std::optional<size_t> num_links,
               std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    dim,
                    cluster_axis,
                    mesh_device,
                    from_remote_multi_device_global_semaphore,
                    to_remote_multi_device_global_semaphore,
                    persistent_output_tensors,
                    math_op,
                    memory_config,
                    topology,
                    num_links,
                    worker_subdevice_id_opt);
            },
            nb::arg("input_tensor"),
            nb::arg("dim"),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::arg("from_remote_multi_device_global_semaphore"),
            nb::arg("to_remote_multi_device_global_semaphore"),
            nb::kw_only(),
            nb::arg("persistent_output_tensors") = nb::none(),
            nb::arg("math_op") = ttnn::operations::reduction::ReduceType::Sum,
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Linear,
            nb::arg("num_links") = nb::none(),
            nb::arg("subdevice_id") = nb::none()});
}

}  // namespace

void bind_reduce_scatter_async(nb::module_& mod) {
    bind_reduce_scatter(
        mod,
        ttnn::experimental::reduce_scatter_async,
        R"doc(

        Performs an reduce_scatter operation on multi-device :attr:`input_tensor` across all devices.  This operation requires a persistent
        fabric to be enabled in order to function.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor
            dim (int): Dimension to perform operation
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the line-reduce-scatter operation on.
            mesh_device (MeshDevice): Device mesh to perform the line-reduce-scatter operation on.
        * cluster_axis and mesh_device parameters are applicable only for Linear Topology.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            num_links (int, optional): Number of links to use for the reduce_scatter operation. Defaults to `None`, which indicates to the operation that it should choose. Note that this value will be ignored if there are fewer links available than requested.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:

            >>> full_tensor = torch.randn([1, 1, 256, 256], dtype=torch.bfloat16)
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
            >>> input_tensor = ttnn.from_torch(
                    full_tensor,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
                )
            >>> output = ttnn.reduce_scatter(input_tensor, dim=0, topology=ttnn.Topology.Linear)

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
