// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "all_gather_minimal_matmul_async.hpp"
#include "device/all_gather_minimal_matmul_async_device_operation.hpp"
#include "device/tuned/agmm_tuned_device_operation.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;

namespace ttnn {

std::vector<ttnn::Tensor> all_gather_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    const std::optional<float> scalar,
    const std::optional<ttnn::Tensor>& addcmul_input_tensor1,
    const std::optional<ttnn::Tensor>& addcmul_input_tensor2,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const ::ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const ttnn::ccl::Topology topology,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    std::optional<uint32_t> cluster_axis,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const bool force_transpose,
    uint32_t num_workers_per_link,
    uint32_t num_buffers_per_channel,
    int32_t chunks,
    int32_t dim,
    std::optional<uint32_t> fsdp_cluster_axis,
    const std::vector<GlobalSemaphore>& fsdp_multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_weight_buffer,
    std::optional<ttnn::ccl::Topology> fsdp_topology,
    bool fuse_swiglu,
    const std::vector<uint32_t>& chunk_sizes,
    std::optional<bool> transpose_core_grid,
    const std::optional<const operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>& matmul_program_config,
    const std::optional<ttnn::Tensor>& gather_table) {
    if (matmul_program_config.has_value()) {
        // Tuned Blackhole Fabric2D path: 2D multicast matmul with the TP all-gather inside its program.
        TT_FATAL(
            tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig()),
            "all_gather_minimal_matmul_async: matmul_program_config (tuned path) needs a Fabric2D / torus fabric");
        TT_FATAL(cluster_axis.has_value(), "all_gather_minimal_matmul_async tuned path: cluster_axis is required");
        TT_FATAL(
            persistent_output_buffer.has_value() && barrier_semaphore.has_value() && gather_table.has_value(),
            "all_gather_minimal_matmul_async tuned path: persistent_output_buffer (gathered), barrier_semaphore "
            "(ready) and gather_table are required");
        TT_FATAL(
            !bias_tensor.has_value() && !scalar.has_value() && !fused_activation.has_value() && !fuse_swiglu &&
                chunks == 1 && !fsdp_cluster_axis.has_value() && !persistent_weight_buffer.has_value(),
            "all_gather_minimal_matmul_async tuned path: bias, addcmul, activation, SwiGLU, chunks and FSDP are not "
            "supported");
        std::vector<uint32_t> counters;
        counters.reserve(multi_device_global_semaphore.size());
        for (const auto& counter : multi_device_global_semaphore) {
            counters.push_back(static_cast<uint32_t>(counter.address()));
        }
        const auto kernel_config = init_device_compute_kernel_config(
            input_tensor.device()->arch(),
            compute_kernel_config,
            tt::tt_metal::MathFidelity::HiFi2,
            /*default_approx_mode=*/false,
            /*default_fp32_acc=*/true,
            /*default_l1_acc=*/true);
        return {ttnn::experimental::prim::agmm_tuned(
            ttnn::experimental::prim::AgmmTunedParams{
                .program_config = matmul_program_config.value(),
                .compute_kernel_config = kernel_config,
                .output_mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
                .output_dtype = dtype.has_value() ? dtype.value() : input_tensor.dtype(),
                .cluster_axis = cluster_axis.value(),
                .num_links = num_links,
                .ready_semaphore_address = static_cast<uint32_t>(barrier_semaphore->address()),
                .counter_addresses = std::move(counters)},
            ttnn::experimental::prim::AgmmTunedInputs{
                .input = input_tensor,
                .weight = weight_tensor,
                .gathered = persistent_output_buffer.value(),
                .gather_table = gather_table.value()})};
    }
    return ttnn::prim::all_gather_minimal_matmul_async(
        input_tensor,
        weight_tensor,
        bias_tensor,
        scalar,
        addcmul_input_tensor1,
        addcmul_input_tensor2,
        std::move(fused_activation),
        config,
        multi_device_global_semaphore,
        topology,
        memory_config,
        dtype,
        compute_kernel_config,
        persistent_output_buffer,
        num_links,
        cluster_axis,
        barrier_semaphore,
        force_transpose,
        num_workers_per_link,
        num_buffers_per_channel,
        chunks,
        dim,
        fsdp_cluster_axis,
        fsdp_multi_device_global_semaphore,
        persistent_weight_buffer,
        fsdp_topology,
        fuse_swiglu,
        chunk_sizes,
        transpose_core_grid);
}

std::vector<std::vector<uint32_t>> all_gather_minimal_matmul_async_gather_table(
    const MeshDevice& mesh_device,
    int32_t cluster_axis,
    uint32_t num_links,
    const CoreCoord& matmul_grid,
    const std::vector<GlobalSemaphore>& slice_counters) {
    TT_FATAL(cluster_axis >= 0, "all_gather_minimal_matmul_async_gather_table: cluster_axis must be nonnegative");
    std::vector<uint32_t> counters;
    counters.reserve(slice_counters.size());
    for (const auto& counter : slice_counters) {
        counters.push_back(static_cast<uint32_t>(counter.address()));
    }
    return ttnn::experimental::prim::agmm_tuned_gather_table_rows(
        mesh_device, static_cast<uint32_t>(cluster_axis), num_links, matmul_grid.x, counters);
}

}  // namespace ttnn
