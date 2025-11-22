// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_async_nanobind.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"
#include "ttnn/types.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_all_reduce_async(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t num_devices,
               const std::vector<GlobalSemaphore>& barrier_semaphores,
               const std::vector<GlobalSemaphore>& rs_global_semaphores,
               const std::vector<GlobalSemaphore>& ag_global_semaphores,
               ttnn::operations::reduction::ReduceType math_op,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology,
               const std::optional<size_t> num_links,
               std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    num_devices,
                    barrier_semaphores,
                    rs_global_semaphores,
                    ag_global_semaphores,
                    math_op,
                    memory_config,
                    topology,
                    num_links,
                    worker_subdevice_id_opt);
            },
            nb::arg("input_tensor"),
            nb::arg("num_devices"),
            nb::arg("barrier_semaphores"),
            nb::arg("rs_global_semaphores"),
            nb::arg("ag_global_semaphores"),
            nb::arg("math_op"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Linear,
            nb::arg("num_links") = nb::none(),
            nb::arg("subdevice_id") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const std::vector<GlobalSemaphore>& barrier_semaphores,
               const std::vector<GlobalSemaphore>& rs_global_semaphores,
               const std::vector<GlobalSemaphore>& ag_global_semaphores,
               ttnn::operations::reduction::ReduceType math_op,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology,
               const std::optional<size_t> num_links,
               std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    cluster_axis,
                    mesh_device,
                    barrier_semaphores,
                    rs_global_semaphores,
                    ag_global_semaphores,
                    math_op,
                    memory_config,
                    topology,
                    num_links,
                    worker_subdevice_id_opt);
            },
            nb::arg("input_tensor"),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::arg("barrier_semaphores"),
            nb::arg("rs_global_semaphores"),
            nb::arg("ag_global_semaphores"),
            nb::arg("math_op"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Linear,
            nb::arg("num_links") = nb::none(),
            nb::arg("subdevice_id") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               ttnn::Tensor& buffer_tensor,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const GlobalSemaphore& multi_device_global_semaphore,
               const std::optional<const DataType> dtype,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology,
               const std::optional<size_t> num_links,
               std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
               bool use_noc1_only,
               bool use_optimal_ccl_for_llama) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    buffer_tensor,
                    cluster_axis,
                    mesh_device,
                    multi_device_global_semaphore,
                    dtype,
                    memory_config,
                    topology,
                    num_links,
                    worker_subdevice_id_opt,
                    use_noc1_only,
                    use_optimal_ccl_for_llama);
            },
            nb::arg("input_tensor"),
            nb::arg("buffer_tensor"),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::arg("multi_device_global_semaphore"),
            nb::kw_only(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Linear,
            nb::arg("num_links") = nb::none(),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("use_noc1_only") = false,
            nb::arg("use_optimal_ccl_for_llama") = false});
}

}  // namespace

void bind_all_reduce_async(nb::module_& mod) {
    bind_all_reduce_async(
        mod,
        ttnn::experimental::all_reduce_async,
        R"doc(
        Performs an all_reduce operation on multi-device :attr:`input_tensor` across all devices.  This operation requires a persistent
        fabric to be enabled in order to function.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the line-all-reduce operation on.
            mesh_device (MeshDevice): Device mesh to perform the line-all-reduce operation on.
        * cluster_axis and mesh_device parameters are applicable only for Linear Topology.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            num_links (int, optional): Number of links to use for the all_reduce_async operation. Defaults to `None`, which indicates to the operation that it should choose. Note that this value will be ignored if there are fewer links available than requested.
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
            >>> output = ttnn.experimental.all_reduce_async(input_tensor, topology=ttnn.Topology.Linear)

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
