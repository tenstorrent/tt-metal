// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_async_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_matmul_reduce_scatter_async(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
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
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("persistent_intermediate_buffer"),
            nb::arg("persistent_output_buffer"),
            nb::arg("dim"),
            nb::arg("multi_device_global_semaphore"),
            nb::arg("reduce_scatter_core_grid_offset"),
            nb::kw_only(),
            nb::arg("barrier_semaphore") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config_rs") = nb::none(),
            nb::arg("intermediate_memory_config_rs") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Ring,
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("memory_config_mm") = nb::none(),
            nb::arg("transpose_a") = false,
            nb::arg("transpose_b") = false,
            nb::arg("dtype") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("activation") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("core_grid") = nb::none()});
}

}  // namespace

void bind_matmul_reduce_scatter_async(nb::module_& mod) {
    bind_matmul_reduce_scatter_async(
        mod,
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
