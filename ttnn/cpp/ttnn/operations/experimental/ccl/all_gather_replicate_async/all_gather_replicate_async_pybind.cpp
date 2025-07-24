// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_replicate_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_replicate_async/all_gather_replicate_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_replicate_async/all_gather_replicate_async.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_llama_all_gather_matmul_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::Tensor& intermediate_tensor,
               const ttnn::Tensor& aggregated_tensor,
               const int32_t dim,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const ttnn::ccl::Topology topology,
               const GlobalSemaphore& multi_device_global_semaphore,
               const std::optional<size_t> num_preferred_links,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
               const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const DataType> dtype) -> ttnn::Tensor {
                return self(
                    input_tensor,    // in0 for matmul, need AG first
                    input_tensor_b,  // in1 for matmul
                    intermediate_tensor,
                    aggregated_tensor,
                    dim,
                    cluster_axis,
                    mesh_device,
                    topology,
                    multi_device_global_semaphore,
                    memory_config,        // = std::nullopt,
                    num_preferred_links,  // = std::nullopt,
                    subdevice_id,         // = std::nullopt
                    // MM optional params
                    program_config,         // = std::nullopt
                    compute_kernel_config,  // = std::nullopt
                    dtype);                 // = std::nullopt
            },
            py::arg("input_tensor"),
            py::arg("input_tensor_b"),
            py::arg("intermediate_tensor"),
            py::arg("aggregated_tensor"),
            py::arg("dim"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::arg("topology"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("num_links") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("dtype") = std::nullopt});
}

}  // namespace

void py_bind_llama_all_gather_matmul_async(pybind11::module& module) {
    bind_llama_all_gather_matmul_async(
        module,
        ttnn::experimental::llama_all_gather_matmul_async,
        R"doc(

        Performs an all-gather-replicate operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to perform operation.
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the line-all-gather-replicate operation on.
            mesh_device (MeshDevice): Device mesh to perform the line-all-gather-replicate operation on.
        * cluster_axis and mesh_device parameters are applicable only for Linear Topology.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the all-gather-replicate operation. Defaults to `1`.
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
            >>> output = ttnn.all_gather_replicate(ttnn_tensor, dim=0, topology=ttnn.Topology.Ring)

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
