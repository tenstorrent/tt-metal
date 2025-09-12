// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_all_gather_matmul_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/llama_all_gather_matmul_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

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
               const ttnn::Tensor& input_tensor0,
               const ttnn::Tensor& input_tensor1,
               const ttnn::Tensor& intermediate_tensor,
               const int32_t dim,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const ttnn::ccl::Topology topology,
               const GlobalSemaphore& multi_device_global_semaphore,
               const std::optional<size_t> num_preferred_links,
               const std::optional<MemoryConfig>& ag_memory_config,
               const std::optional<MemoryConfig>& mm_memory_config,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
               const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const DataType> dtype,
               const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb) -> ttnn::Tensor {
                return self(
                    input_tensor0,        // in0 for matmul, need AG first
                    input_tensor1,        // in1 for matmul
                    intermediate_tensor,  // intermediate tensor for AG operation
                    dim,
                    cluster_axis,
                    mesh_device,
                    topology,
                    multi_device_global_semaphore,
                    ag_memory_config,     // = std::nullopt,
                    mm_memory_config,     // = std::nullopt,
                    num_preferred_links,  // = std::nullopt,
                    subdevice_id,         // = std::nullopt
                    // MM optional params
                    program_config,         // = std::nullopt
                    compute_kernel_config,  // = std::nullopt
                    dtype,                  // = std::nullopt
                    global_cb);             // = std::nullopt
            },
            py::arg("input_tensor0"),
            py::arg("input_tensor1"),
            py::arg("intermediate_tensor"),
            py::arg("dim"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::arg("topology"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("num_links") = std::nullopt,
            py::arg("ag_memory_config") = std::nullopt,
            py::arg("mm_memory_config") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("global_cb") = std::nullopt});
}

}  // namespace

void py_bind_llama_all_gather_matmul_async(pybind11::module& module) {
    bind_llama_all_gather_matmul_async(
        module,
        ttnn::experimental::llama_all_gather_matmul_async,
        R"doc(
        Performs an all-gather-matml operation on multi-device :attr:`input_tensor0` and :attr:`input_tensor1` across all devices.

        Args:
            input_tensor0 (ttnn.Tensor): multi-device tensor.
            input_tensor1 (ttnn.Tensor): multi-device tensor.
            intermediate_tensor (ttnn.Tensor): intermediate tensor for the All-Gather operation.
            dim (int): Dimension to perform All-Gather operation.
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the line-all-gather-replicate operation on.
            mesh_device (MeshDevice): Device mesh to perform the line-all-gather-replicate operation on.
            topology (ttnn.Topology): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Linear`.
            multi_device_global_semaphore (ttnn.GlobalSemaphore): The global semaphore to use for the operation.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the all-gather-replicate operation. Defaults to `1`.
            ag_memory_config (ttnn.MemoryConfig, optional): Memory configuration for the All-Gather operation. Defaults to `input tensor memory config`.
            mm_memory_config (ttnn.MemoryConfig, optional): Memory configuration for the Matmul operation. Defaults to `input tensor memory config`.
            subdevice_id (ttnn.SubDeviceId, optional): The subdevice id to use for the operation. Defaults to `None`.
            program_config (ttnn.MatmulProgramConfig, optional): The program configuration to use for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): The compute kernel configuration to use for the operation. Defaults to `None`.
            dtype (ttnn.DataType, optional): The data type to use for the operation. Defaults to `None`.
            global_cb (ttnn.GlobalCircularBuffer, optional): The global circular buffer to use for the operation. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor generated by the All-Gather of input_tensor0 and the Matmul with input_tensor1.

        Example:
            >>> input_tensor0 = torch.randn([1, 1, 32, 896], dtype=torch.bfloat8)
            >>> input_tensor1 = torch.randn([1, 1, 3584, 2048], dtype=torch.bfloat8)
            >>> ttnn_tensor = ttnn.from_torch(
                            full_tensor,
                            dtype=input_dtype,
                            device=mesh_device,
                            layout=layout,
                            memory_config=mem_config,
                            mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(1, 8), dims=(-1, -2)))
            >>> ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
            >>> output = ttnn.llama_all_gather_matmul_async(ttnn_tensor_a, ttnn_tensor_b, dim=0, topology=ttnn.Topology.Ring)

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
