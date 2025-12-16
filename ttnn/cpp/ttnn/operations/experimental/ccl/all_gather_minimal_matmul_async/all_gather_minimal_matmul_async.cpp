// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "all_gather_minimal_matmul_async.hpp"
#include "device/all_gather_minimal_matmul_async_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::all_gather_minimal_matmul_async {

ttnn::Tensor ExecuteAllGatherMinimalMatmulAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<unary::UnaryWithParam> fused_activation,
    const std::optional<const AllGatherMinimalMatmulAsyncConfig>& config,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const ttnn::ccl::Topology topology,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    std::optional<uint32_t> cluster_axis,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    uint32_t chunks_per_sync,
    uint32_t num_workers_per_link,
    uint32_t num_buffers_per_channel) {
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        false /*approx_mode*/,
        true /*fp32_acc*/,
        true /*packer_acc*/);

    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    bool using_persistent_buffers = persistent_output_buffer.has_value();

    std::vector<std::optional<Tensor>> optional_output_tensors = {persistent_output_buffer};

    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);

    return operation::run(
               AllGatherMinimalMatmulAsyncOp{
                   config,
                   std::move(fused_activation),
                   memory_config,
                   dtype,
                   kernel_config_val,
                   num_links,
                   num_devices,
                   topology_,
                   multi_device_global_semaphore,
                   cluster_axis,
                   barrier_semaphore,
                   using_persistent_buffers,
                   chunks_per_sync,
                   num_workers_per_link,
                   num_buffers_per_channel},
               {input_tensor, weight_tensor},
               {bias_tensor},
               {optional_output_tensors})
        .at(1);
}

}  // namespace ttnn::operations::experimental::all_gather_minimal_matmul_async
