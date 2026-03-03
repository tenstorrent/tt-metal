// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_strided_reduce_scatter_async_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/minimal_matmul_strided_reduce_scatter_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_minimal_matmul_strided_reduce_scatter_async_op(
    nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               const uint32_t dim,
               const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
               const CoreCoord reduce_scatter_core_grid_offset,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config_mm,
               const std::optional<ttnn::MemoryConfig>& rs_output_mem_config,
               const std::optional<ttnn::MemoryConfig>& rs_intermediate_mem_config,
               const ttnn::ccl::Topology topology,
               std::optional<uint32_t> cluster_axis,
               const std::optional<const Tensor>& bias,
               const std::optional<unary::UnaryWithParam>& fused_activation,
               const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
               const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<GlobalSemaphore>& barrier_semaphore,
               bool using_persistent_buffers,
               std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
               std::optional<uint32_t> chunks_per_sync,
               std::optional<uint32_t> num_workers_per_link,
               std::optional<uint32_t> num_buffers_per_channel,
               std::optional<uint32_t> chunk_width_in_mm_blocks,
               const std::optional<Tensor>& optional_rs_intermediate_tensor,
               const std::optional<Tensor>& optional_rs_output_tensor) -> std::vector<ttnn::Tensor> {
                return self(
                    input_tensor,
                    weight_tensor,
                    dim,
                    multi_device_global_semaphore,
                    reduce_scatter_core_grid_offset,
                    num_links,
                    memory_config_mm,
                    rs_output_mem_config,
                    rs_intermediate_mem_config,
                    topology,
                    cluster_axis,
                    bias,
                    fused_activation,
                    config,
                    compute_kernel_config,
                    barrier_semaphore,
                    using_persistent_buffers,
                    sub_device_id,
                    chunks_per_sync,
                    num_workers_per_link,
                    num_buffers_per_channel,
                    chunk_width_in_mm_blocks,
                    optional_rs_intermediate_tensor,
                    optional_rs_output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("dim"),
            nb::arg("multi_device_global_semaphore"),
            nb::arg("reduce_scatter_core_grid_offset"),
            nb::kw_only(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config_mm") = nb::none(),
            nb::arg("rs_output_mem_config") = nb::none(),
            nb::arg("rs_intermediate_mem_config") = nb::none(),
            nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Ring),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("fused_activation") = nb::none(),
            nb::arg("config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("barrier_semaphore") = nb::none(),
            nb::arg("using_persistent_buffers") = false,
            nb::arg("sub_device_id") = nb::none(),
            nb::arg("chunks_per_sync") = nb::none(),
            nb::arg("num_workers_per_link") = nb::none(),
            nb::arg("num_buffers_per_channel") = nb::none(),
            nb::arg("chunk_width_in_mm_blocks") = nb::none(),
            nb::arg("optional_rs_intermediate_tensor") = nb::none(),
            nb::arg("optional_rs_output_tensor") = nb::none()});
}

}  // namespace

void bind_minimal_matmul_strided_reduce_scatter_async(nb::module_& mod) {
    bind_minimal_matmul_strided_reduce_scatter_async_op(
        mod,
        ttnn::experimental::minimal_matmul_strided_reduce_scatter_async,
        R"doc(minimal_matmul_strided_reduce_scatter_async(input_tensor: ttnn.Tensor, weight_tensor: ttnn.Tensor, dim: int, ...) -> (ttnn.Tensor, ttnn.Tensor, ttnn.Tensor)

        Performs a fused matmul followed by strided reduce-scatter operation.
        The matmul output is fed directly into the reduce-scatter, with the matmul
        signaling the reduce-scatter via semaphores as output blocks become ready.

        Returns three tensors:
            [0] matmul output (intermediate between MM and RS)
            [1] reduce-scatter intermediate buffer
            [2] reduce-scatter output (final result)

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device input activations tensor
            * :attr:`weight_tensor` (ttnn.Tensor): multi-device weight tensor
            * :attr:`dim` (int): scatter dimension for reduce-scatter
            * :attr:`multi_device_global_semaphore`: global semaphores for reduce-scatter
            * :attr:`reduce_scatter_core_grid_offset` (ttnn.CoreCoord): Core grid offset for the reduce-scatter operation

        Keyword Args:
            * :attr:`num_links` (int): Number of links for reduce-scatter. Defaults to 1.
            * :attr:`memory_config_mm` (Optional[ttnn.MemoryConfig]): Memory configuration for the matmul output.
            * :attr:`rs_output_mem_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the RS output.
            * :attr:`rs_intermediate_mem_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the RS intermediate.
            * :attr:`topology` (ttnn.Topology): Communication topology. Defaults to Ring.
            * :attr:`cluster_axis` (Optional[int]): Cluster axis for the operation.
            * :attr:`bias` (Optional[ttnn.Tensor]): Optional bias tensor for the matmul.
            * :attr:`fused_activation` (Optional[str]): Fused activation for the matmul.
            * :attr:`config` (Optional[MinimalMatmulConfig]): Matmul configuration.
            * :attr:`compute_kernel_config` (Optional[DeviceComputeKernelConfig]): Compute kernel config.
            * :attr:`barrier_semaphore` (Optional[GlobalSemaphore]): Barrier semaphore for RS.
            * :attr:`using_persistent_buffers` (bool): Use persistent buffers. Defaults to False.
            * :attr:`sub_device_id` (Optional[SubDeviceId]): Sub-device ID.
            * :attr:`chunks_per_sync` (Optional[int]): Chunks per sync for RS.
            * :attr:`num_workers_per_link` (Optional[int]): Workers per link for RS.
            * :attr:`num_buffers_per_channel` (Optional[int]): Buffers per channel for RS.
            * :attr:`chunk_width_in_mm_blocks` (Optional[int]): MM output blocks per RS chunk.
            * :attr:`optional_rs_intermediate_tensor` (Optional[ttnn.Tensor]): Pre-allocated RS intermediate.
            * :attr:`optional_rs_output_tensor` (Optional[ttnn.Tensor]): Pre-allocated RS output.

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
