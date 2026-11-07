// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_matmul_prefill.hpp"

#include "device/all_gather_matmul_prefill_device_operation.hpp"

namespace ttnn::experimental {

Tensor all_gather_matmul_prefill(
    const Tensor& input_tensor,
    const Tensor& weight,
    const Tensor& gathered,
    const tt::tt_metal::GlobalSemaphore& ready_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& slice_counters,
    int32_t cluster_axis,
    const operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig& program_config,
    uint32_t num_links,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType>& dtype) {
    TT_FATAL(cluster_axis >= 0, "all_gather_matmul_prefill: cluster_axis must be nonnegative");
    std::vector<uint32_t> counters;
    counters.reserve(slice_counters.size());
    for (const auto& counter : slice_counters) {
        counters.push_back(static_cast<uint32_t>(counter.address()));
    }
    const auto kernel_config = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/true);
    return ttnn::experimental::prim::all_gather_matmul_prefill(
        ttnn::experimental::prim::AllGatherMatmulPrefillParams{
            .program_config = program_config,
            .compute_kernel_config = kernel_config,
            .output_mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
            .output_dtype = dtype.value_or(input_tensor.dtype()),
            .cluster_axis = static_cast<uint32_t>(cluster_axis),
            .num_links = num_links,
            .ready_semaphore_address = static_cast<uint32_t>(ready_semaphore.address()),
            .counter_addresses = std::move(counters)},
        ttnn::experimental::prim::AllGatherMatmulPrefillInputs{
            .input = input_tensor, .weight = weight, .gathered = gathered});
}

}  // namespace ttnn::experimental
