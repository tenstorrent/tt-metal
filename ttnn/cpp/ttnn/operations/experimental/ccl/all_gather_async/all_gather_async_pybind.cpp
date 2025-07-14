// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
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
void bind_all_gather_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    // namespace py = pybind11;

    py::class_<GlobalSemaphoreArg>(module, "GlobalSemaphoreArg")
        .def(py::init<const GlobalSemaphore&>())
        .def(py::init<const std::vector<GlobalSemaphore>&>());

    py::implicitly_convertible<GlobalSemaphore, GlobalSemaphoreArg>();
    py::implicitly_convertible<std::vector<GlobalSemaphore>, GlobalSemaphoreArg>();

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               const GlobalSemaphoreArg& multi_device_global_semaphore,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               bool use_optimal_ccl_for_llama) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    dim,
                    multi_device_global_semaphore.get(),
                    num_links,
                    memory_config,
                    topology,
                    subdevice_id,
                    use_optimal_ccl_for_llama);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("use_optimal_ccl_for_llama") = false},

        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               ttnn::Tensor& persistent_output_buffer,
               const int32_t dim,
               const GlobalSemaphoreArg& multi_device_global_semaphore,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               std::optional<uint32_t> cluster_axis,
               bool use_optimal_ccl_for_llama) -> ttnn::Tensor {
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
                    use_optimal_ccl_for_llama);
            },
            py::arg("input_tensor"),
            py::arg("persistent_output_buffer"),
            py::arg("dim"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("cluster_axis") = std::nullopt,
            py::arg("use_optimal_ccl_for_llama") = false},

        ttnn::pybind_overload_t{
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
               bool use_optimal_ccl_for_llama) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    dim,
                    cluster_axis,
                    mesh_device,
                    topology,
                    multi_device_global_semaphore.get(),
                    persistent_output_tensor,  // = std::nullopt,
                    memory_config,             // = std::nullopt,
                    num_preferred_links,       // = std::nullopt,
                    subdevice_id,
                    use_optimal_ccl_for_llama);  // = std::nullopt
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::arg("topology"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("persistent_output_tensor") = std::nullopt,
            py::arg("num_links") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("use_optimal_ccl_for_llama") = false});
}

}  // namespace

void py_bind_all_gather_async(pybind11::module& module) {
    bind_all_gather_async(
        module,
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
            >>> physical_device_ids = ttnn.get_t3k_physical_device_ids_ring()
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), physical_device_ids=physical_device_ids[:8])
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
}

}  // namespace ttnn::operations::experimental::ccl
