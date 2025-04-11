// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async_nanobind.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

struct GlobalSemaphoreArg {
    // allows semaphore arguments to be passed as a single semaphore or a vector of semaphores
    std::vector<GlobalSemaphore> semaphores;
    GlobalSemaphoreArg(const GlobalSemaphore& single) : semaphores{single} {}
    GlobalSemaphoreArg(const std::vector<GlobalSemaphore>& vec) : semaphores(vec) {}
    GlobalSemaphoreArg(std::vector<GlobalSemaphore>&& vec) : semaphores(std::move(vec)) {}
    const std::vector<GlobalSemaphore>& get() const { return semaphores; }
    operator const std::vector<GlobalSemaphore>&() const { return semaphores; }
};

template <typename ccl_operation_t>
void bind_all_gather_async(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               const GlobalSemaphoreArg& multi_device_global_semaphore,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               bool use_optimal_ccl_for_llama,
               const std::optional<GlobalSemaphore>& barrier_semaphore) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    dim,
                    multi_device_global_semaphore.get(),
                    num_links,
                    memory_config,
                    topology,
                    subdevice_id,
                    use_optimal_ccl_for_llama,
                    barrier_semaphore,
                    false);
            },
            nb::arg("input_tensor"),
            nb::arg("dim"),
            nb::arg("multi_device_global_semaphore"),
            nb::kw_only(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Ring,
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("use_optimal_ccl_for_llama") = false,
            nb::arg("barrier_semaphore") = nb::none()},

        // ring
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<ttnn::Tensor>& persistent_output_buffer,
               const int32_t dim,
               const GlobalSemaphoreArg& multi_device_global_semaphore,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               std::optional<uint32_t> cluster_axis,
               bool use_optimal_ccl_for_llama,
               const std::optional<GlobalSemaphore>& barrier_semaphore,
               std::optional<uint32_t> chunks_per_sync,
               std::optional<uint32_t> num_workers_per_link,
               std::optional<uint32_t> num_buffers_per_channel) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    persistent_output_buffer,
                    dim,
                    multi_device_global_semaphore.get(),
                    num_links,
                    memory_config,
                    topology,
                    subdevice_id,
                    cluster_axis,
                    use_optimal_ccl_for_llama,
                    barrier_semaphore,
                    chunks_per_sync,
                    num_workers_per_link,
                    num_buffers_per_channel,
                    false);
            },
            nb::arg("input_tensor"),
            nb::arg("persistent_output_buffer"),
            nb::arg("dim"),
            nb::arg("multi_device_global_semaphore"),
            nb::kw_only(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Ring,
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("use_optimal_ccl_for_llama") = false,
            nb::arg("barrier_semaphore") = nb::none(),
            nb::arg("chunks_per_sync") = nb::none(),
            nb::arg("num_workers_per_link") = nb::none(),
            nb::arg("num_buffers_per_channel") = nb::none()},

        // line
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const ttnn::ccl::Topology topology,
               const GlobalSemaphoreArg& multi_device_global_semaphore,
               const std::optional<ttnn::Tensor>& persistent_output_tensor,
               const std::optional<size_t> num_preferred_links,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               bool use_optimal_ccl_for_llama,
               const std::optional<GlobalSemaphore>& barrier_semaphore) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    dim,
                    cluster_axis,
                    mesh_device,
                    topology,
                    multi_device_global_semaphore.get(),
                    persistent_output_tensor,  // = nb::none(),
                    memory_config,             // = nb::none(),
                    num_preferred_links,       // = nb::none(),
                    subdevice_id,
                    use_optimal_ccl_for_llama,
                    barrier_semaphore,
                    false);
            },
            nb::arg("input_tensor"),
            nb::arg("dim"),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::arg("topology"),
            nb::arg("multi_device_global_semaphore"),
            nb::kw_only(),
            nb::arg("persistent_output_tensor") = nb::none(),
            nb::arg("num_links") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("use_optimal_ccl_for_llama") = false,
            nb::arg("barrier_semaphore") = nb::none()});
}

}  // namespace

void bind_all_gather_async(nb::module_& mod) {
    // Define GlobalSemaphoreArg once, outside the template
    nb::class_<GlobalSemaphoreArg>(mod, "GlobalSemaphoreArg")
        .def(nb::init<const GlobalSemaphore&>())
        .def(nb::init<const std::vector<GlobalSemaphore>&>());

    nb::implicitly_convertible<GlobalSemaphore, GlobalSemaphoreArg>();
    nb::implicitly_convertible<std::vector<GlobalSemaphore>, GlobalSemaphoreArg>();

    bind_all_gather_async(
        mod,
        ttnn::experimental::all_gather_async,
        R"doc(
        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to perform operation.
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the line-all-gather operation on.
            mesh_device (MeshDevice): Device mesh to perform the line-all-gather operation on.
        * cluster_axis and mesh_device parameters are applicable only for Linear Topology.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the all-gather operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> full_tensor = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
            >>> ttnn_tensor = ttnn.from_torch(
                            full_tensor,
                            dtype=input_dtype,
                            device=mesh_device,
                            layout=layout,
                            memory_config=mem_config,
                            mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(1, 8), dims=(-1, -2)))
            >>> ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
            >>> output = ttnn.all_gather(ttnn_tensor, dim=0, topology=ttnn.Topology.Ring)

        )doc");

    bind_all_gather_async(
        mod,
        ttnn::experimental::all_gather_async_reversed,
        R"doc(

        Performs a reversed all-gather operation on multi-device :attr:`input_tensor` across all devices.
        This is identical to all_gather_async but with reversed device ordering in the output.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to perform operation.
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the line-all-gather operation on.
            mesh_device (MeshDevice): Device mesh to perform the line-all-gather operation on.
        * cluster_axis and mesh_device parameters are applicable only for Linear Topology.

        Keyword Args:
            num_links (int, optional): Number of links to use for the all-gather operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor: the all-gathered tensor with reversed device ordering.

        Note:
            The tensor width must be divisible by 32*num_devices when using this reversed API.

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
