// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "strided_all_gather_minimal_matmul_async_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/strided_all_gather_minimal_matmul_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_strided_all_gather_minimal_matmul_async_op(
    nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
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
               const CoreCoord strided_all_gather_core_grid_offset,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config_ag,
               const ttnn::ccl::Topology topology,
               std::optional<uint32_t> cluster_axis,
               const std::optional<const Tensor>& bias,
               std::optional<unary::UnaryWithParam> fused_activation,
               const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
               const std::optional<ttnn::MemoryConfig>& memory_config_mm,
               const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
               std::optional<uint32_t> num_workers_per_link,
               std::optional<uint32_t> num_buffers_per_channel,
               std::optional<bool> read_local_slice_from_input) -> std::vector<ttnn::Tensor> {
                return self(
                    input_tensor,
                    weight_tensor,
                    persistent_output_buffer,
                    dim,
                    multi_device_global_semaphore,
                    strided_all_gather_core_grid_offset,
                    num_links,
                    memory_config_ag,
                    topology,
                    cluster_axis,
                    bias,
                    fused_activation,
                    config,
                    memory_config_mm,
                    compute_kernel_config,
                    num_workers_per_link,
                    num_buffers_per_channel,
                    read_local_slice_from_input);
            },
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("persistent_output_buffer"),
            nb::arg("dim"),
            nb::arg("multi_device_global_semaphore"),
            nb::arg("strided_all_gather_core_grid_offset"),
            nb::kw_only(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config_ag") = nb::none(),
            nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Ring),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("fused_activation") = nb::none(),
            nb::arg("config") = nb::none(),
            nb::arg("memory_config_mm") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("num_workers_per_link") = nb::none(),
            nb::arg("num_buffers_per_channel") = nb::none(),
            nb::arg("read_local_slice_from_input") = nb::none()});
}

}  // namespace

void bind_strided_all_gather_minimal_matmul_async(nb::module_& mod) {
    bind_strided_all_gather_minimal_matmul_async_op(
        mod,
        ttnn::experimental::strided_all_gather_minimal_matmul_async,
        R"doc(strided_all_gather_minimal_matmul_async(input_tensor: ttnn.Tensor, weight_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> (ttnn.Tensor, ttnn.Tensor)

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`weight_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)
            * :attr:`all_gather_core_grid_offset` (ttnn.CoreCoord): Core grid offset for the all-gather operation.

        Keyword Args:
            * :attr:`bias` (ttnn.Tensor): the bias tensor to be added. If specified, needs to be on the device. Defaults to `None`.
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`topology` (ttnn.Topology): Communication topology for the all-gather. Defaults to `ttnn.Topology.Ring`.
            * :attr:`memory_config_ag` (Optional[ttnn.MemoryConfig]): Memory configuration for the All Gather operation.
            * :attr:`memory_config_mm` (Optional[ttnn.MemoryConfig]): Memory configuration for the Matmul operation.
            * :attr:`transpose_a` (bool)
            * :attr:`transpose_b` (bool)
            * :attr:`dtype` (Optional[DataType])
            * :attr:`program_config` (Optional[ttnn.MatmulProgramConfig])
            * :attr:`fused_activation` (Optional[str])
            * :attr:`compute_kernel_config` (Optional[DeviceComputeKernelConfig])

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> weight_tensor = ttnn.from_torch(torch.tensor((2, 1), dtype=torch.bfloat16), device=device)
            >>> all_gathered_mm_in, mm_out = ttnn.strided_all_gather_minimal_matmul_async(tensor, weight_tensor, dim=0, (0, 0))

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
