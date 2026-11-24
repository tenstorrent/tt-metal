// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "router_front_end.hpp"

#include "device/router_front_end_device_operation.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

std::tuple<Tensor, Tensor> router_front_end(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<tt::tt_metal::MemoryConfig>& scores_memory_config,
    const std::optional<tt::tt_metal::MemoryConfig>& rows_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // Without a config: HiFi4 with fp32 accumulation, which the softmax needs both of.
    const auto kernel_config = init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);
    const tt::tt_metal::MemoryConfig dram{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    return ttnn::prim::router_front_end(
        input, weight, scores_memory_config.value_or(dram), rows_memory_config.value_or(dram), kernel_config);
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
