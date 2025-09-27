// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_matmul_async_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/all_gather_matmul_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_all_gather_matmul_async(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               const std::optional<ttnn::Tensor>& persistent_output_buffer,
               const uint32_t dim,
               const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
               const CoreCoord all_gather_core_grid_offset,
               const std::optional<const Tensor>& bias,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config_ag,
               const ttnn::ccl::Topology topology,
               const std::optional<GlobalSemaphore>& barrier_semaphore,
               std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
               const std::optional<ttnn::MemoryConfig>& memory_config_mm,
               const bool transpose_a,
               const bool transpose_b,
               const std::optional<const DataType> dtype,
               const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
               const std::optional<const std::string>& activation,
               const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const ttnn::CoreGrid> core_grid,
               std::optional<uint32_t> chunks_per_sync,
               std::optional<uint32_t> num_workers_per_link,
               std::optional<uint32_t> num_buffers_per_channel) -> std::vector<ttnn::Tensor> {
                return self(
                    input_tensor,
                    weight_tensor,
                    persistent_output_buffer,
                    dim,
                    multi_device_global_semaphore,
                    all_gather_core_grid_offset,
                    bias,
                    num_links,
                    memory_config_ag,
                    topology,
                    barrier_semaphore,
                    sub_device_id,
                    memory_config_mm,
                    transpose_a,
                    transpose_b,
                    dtype,
                    program_config,
                    activation,
                    compute_kernel_config,
                    core_grid,
                    chunks_per_sync,
                    num_workers_per_link,
                    num_buffers_per_channel);
            },
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("persistent_output_buffer"),
            nb::arg("dim"),
            nb::arg("multi_device_global_semaphore"),
            nb::arg("all_gather_core_grid_offset"),
            nb::kw_only(),
            nb::arg("bias") = nb::none(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config_ag") = nb::none(),
            nb::arg("topology") = ttnn::ccl::Topology::Ring,
            nb::arg("barrier_semaphore") = nb::none(),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("memory_config_mm") = nb::none(),
            nb::arg("transpose_a") = false,
            nb::arg("transpose_b") = false,
            nb::arg("dtype") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("activation") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("core_grid") = nb::none(),
            nb::arg("chunks_per_sync") = nb::none(),
            nb::arg("num_workers_per_link") = nb::none(),
            nb::arg("num_buffers_per_channel") = nb::none()});
}

}  // namespace

void bind_all_gather_matmul_async(nb::module_& mod) {
    bind_all_gather_matmul_async(
        mod,
        ttnn::experimental::all_gather_matmul_async,
        R"doc(all_gather_matmul_async(input_tensor: ttnn.Tensor, weight_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> (ttnn.Tensor, ttnn.Tensor)

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`weight_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)
            * :attr:`all_gather_core_grid_offset` (ttnn.CoreCoord): Core grid offset for the all-gather operation.

        Keyword Args:
            * :attr:`bias` (ttnn.Tensor): the bias tensor to be added. If specified, needs to be on the device. Defaults to `None`.
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config_ag` (Optional[ttnn.MemoryConfig]): Memory configuration for the All Gather operation.
            * :attr:`memory_config_mm` (Optional[ttnn.MemoryConfig]): Memory configuration for the Matmul operation.
            * :attr:`transpose_a` (bool)
            * :attr:`transpose_b` (bool)
            * :attr:`dtype` (Optional[DataType])
            * :attr:`program_config` (Optional[ttnn.MatmulProgramConfig])
            * :attr:`activation` (Optional[str])
            * :attr:`compute_kernel_config` (Optional[DeviceComputeKernelConfig])
            * :attr:`core_grid` (Optional[ttnn.CoreGrid])

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> weight_tensor = ttnn.from_torch(torch.tensor((2, 1), dtype=torch.bfloat16), device=device)
            >>> all_gathered_mm_in, mm_out = ttnn.all_gather_matmul_async(tensor, weight_tensor, dim=0, (0, 0))

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
