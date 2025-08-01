// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_matmul_reduce_scatter_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    // namespace py = pybind11;

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               ttnn::Tensor& persistent_intermediate_buffer,
               ttnn::Tensor& persistent_output_buffer,
               const uint32_t dim,
               const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
               const CoreCoord reduce_scatter_core_grid_offset,
               const std::optional<GlobalSemaphore>& barrier_semaphore,
               const std::optional<const Tensor>& bias,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config_rs,
               const std::optional<ttnn::MemoryConfig>& intermediate_memory_config_rs,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
               const std::optional<ttnn::MemoryConfig>& memory_config_mm,
               const bool transpose_a,
               const bool transpose_b,
               const std::optional<const DataType> dtype,
               const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
               const std::optional<const std::string>& activation,
               const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const ttnn::CoreGrid> core_grid) -> std::vector<ttnn::Tensor> {
                return self(
                    input_tensor,
                    weight_tensor,
                    persistent_intermediate_buffer,
                    persistent_output_buffer,
                    dim,
                    multi_device_global_semaphore,
                    reduce_scatter_core_grid_offset,
                    barrier_semaphore,
                    bias,
                    num_links,
                    memory_config_rs,
                    intermediate_memory_config_rs,
                    topology,
                    sub_device_id,
                    memory_config_mm,
                    transpose_a,
                    transpose_b,
                    dtype,
                    program_config,
                    activation,
                    compute_kernel_config,
                    core_grid);
            },
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("persistent_intermediate_buffer"),
            py::arg("persistent_output_buffer"),
            py::arg("dim"),
            py::arg("multi_device_global_semaphore"),
            py::arg("reduce_scatter_core_grid_offset"),
            py::kw_only(),
            py::arg("barrier_semaphore") = std::nullopt,
            py::arg("bias") = std::nullopt,
            py::arg("num_links") = 1,
            py::arg("memory_config_rs") = std::nullopt,
            py::arg("intermediate_memory_config_rs") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("memory_config_mm") = std::nullopt,
            py::arg("transpose_a") = false,
            py::arg("transpose_b") = false,
            py::arg("dtype") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("activation") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("core_grid") = std::nullopt});
}

}  // namespace detail

void py_bind_matmul_reduce_scatter_async(pybind11::module& module) {
    detail::bind_matmul_reduce_scatter_async(
        module,
        ttnn::experimental::matmul_reduce_scatter_async,
        R"doc(matmul_reduce_scatter_async(input_tensor: ttnn.Tensor, weight_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> (ttnn.Tensor, ttnn.Tensor)

        Performs an reduce-scatter operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`weight_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)
            * :attr:`reduce_scatter_core_grid_offset` (ttnn.CoreCoord): Core grid offset for the reduce-scatter operation.

        Keyword Args:
            * :attr:`bias` (ttnn.Tensor): the bias tensor to be added. If specified, needs to be on the device. Defaults to `None`.
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config_rs` (Optional[ttnn.MemoryConfig]): Memory configuration for the Reduce Scatter operation.
            * :attr:`memory_config_mm` (Optional[ttnn.MemoryConfig]): Memory configuration for the Matmul operation.
            * :attr:`transpose_a` (bool)
            * :attr:`transpose_b` (bool)
            * :attr:`dtype` (Optional[DataType])
            * :attr:`program_config` (Optional[ttnn.MatmulProgramConfig])
            * :attr:`activation` (Optional[str])
            * :attr:`compute_kernel_config` (Optional[DeviceComputeKernelConfig])
            * :attr:`core_grid` (Optional[ttnn.CoreGrid])
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
