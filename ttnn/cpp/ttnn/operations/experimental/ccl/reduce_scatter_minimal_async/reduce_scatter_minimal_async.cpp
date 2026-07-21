// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async.hpp"
#include "device/reduce_scatter_minimal_async_op_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_program_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

ttnn::Tensor reduce_scatter_minimal_async(
    const ttnn::Tensor& input_tensor,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_buffers,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    std::optional<uint32_t> num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::MemoryConfig>& intermediate_memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device is required for reduce_scatter_minimal_async operation");
    uint32_t resolved_num_links =
        num_links.value_or(ttnn::operations::ccl::common::get_num_links(*mesh_device, cluster_axis));

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t scatter_dim = (dim < 0) ? rank + dim : dim;

    // Calculate ring size based on cluster_axis
    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    TT_FATAL(
        num_devices > 1, "reduce_scatter_minimal_async op will only work for num_devices > 1, but has {}", num_devices);

    // Convert Ring to Linear for 2-device configs where wrapping is not possible
    ttnn::ccl::Topology usable_topology = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);

    log_debug(
        tt::LogOp,
        "reduce_scatter_minimal_async: num_devices = {}, usable_topology = {}",
        num_devices,
        usable_topology);

    // Use composite reduce scatter for edge cases (e.g., tile dimensions not evenly divisible)
    if (composite_common::use_composite_reduce_scatter(input_tensor, dim, cluster_axis)) {
        log_debug(tt::LogOp, "reduce_scatter_minimal_async: using composite_reduce_scatter");
        return composite_common::composite_reduce_scatter(
            input_tensor,
            dim,
            resolved_num_links,
            usable_topology,
            memory_config,
            sub_device_id,
            cluster_axis,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel);
    }

    bool using_persistent_buffers = persistent_output_buffers.has_value();

    std::optional<ttnn::Tensor> optional_intermediate_tensor = std::nullopt;
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt;

    if (using_persistent_buffers) {
        const auto& buffers = persistent_output_buffers.value();
        if (!buffers.empty()) {
            optional_intermediate_tensor = buffers[0];
        }
        if (buffers.size() >= 2) {
            optional_output_tensor = buffers[1];
        }
    }

    // For fp32 inputs without an explicit compute_kernel_config, enable fp32 dest accumulation
    // so the line_reduction sum runs at fp32 precision in dst (Tf32 unpack-dst). Without this,
    // the JIT data-format selection picks a 7-bit-mantissa dst, silently truncating the
    // cross-device sum.
    auto resolved_compute_kernel_config = compute_kernel_config;
    if (!resolved_compute_kernel_config.has_value() && input_tensor.dtype() == DataType::FLOAT32) {
        resolved_compute_kernel_config = ttnn::DeviceComputeKernelConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
        };
    }

    // Call the prim operation
    auto result = ttnn::prim::reduce_scatter_minimal_async(
        input_tensor,
        optional_intermediate_tensor,
        optional_output_tensor,
        scatter_dim,
        resolved_num_links,
        num_devices,
        memory_config.value_or(input_tensor.memory_config()),
        intermediate_memory_config,
        usable_topology,
        multi_device_global_semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        cluster_axis,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        resolved_compute_kernel_config);

    // Return the output tensor (index 1, intermediate is at index 0)
    return result.at(1);
}

ttnn::Tensor reduce_scatter_minimal_async_create_intermediate_buffer(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device is required to allocate the reduce_scatter intermediate buffer");

    // Mirror reduce_scatter_minimal_async's resolution so the sizing is identical to what the op derives.
    const int32_t rank = input_tensor.logical_shape().rank();
    const int32_t scatter_dim = (dim < 0) ? rank + dim : dim;
    const uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    const ttnn::ccl::Topology usable_topology = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);

    // fp32 inputs default to fp32 dest-acc (see reduce_scatter_minimal_async); this affects tile_granularity
    // and therefore the staging page size, so it must be resolved the same way here.
    auto resolved_compute_kernel_config = compute_kernel_config;
    if (!resolved_compute_kernel_config.has_value() && input_tensor.dtype() == DataType::FLOAT32) {
        resolved_compute_kernel_config = ttnn::DeviceComputeKernelConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
        };
    }
    const bool fp32_dest_acc_en = ttnn::get_fp32_dest_acc_en(resolved_compute_kernel_config);

    auto stage_spec = ttnn::experimental::ccl::reduce_scatter_ring_interm_staging_spec(
        input_tensor, usable_topology, scatter_dim, num_devices, fp32_dest_acc_en);
    TT_FATAL(
        stage_spec.has_value(),
        "reduce_scatter_minimal_async_create_intermediate_buffer only applies to the contiguous fast path "
        "(Ring topology, scatter dim != 0). For other configurations the intermediate has the input tensor "
        "shape and can be allocated directly.");

    return create_device_tensor(*stage_spec, mesh_device);
}

}  // namespace ttnn::experimental
