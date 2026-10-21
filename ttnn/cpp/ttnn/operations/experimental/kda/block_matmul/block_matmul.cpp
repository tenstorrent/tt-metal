// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "block_matmul.hpp"

#include "device/block_matmul_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor block_matmul(
    const ttnn::Tensor& a,
    const ttnn::Tensor& w,
    const tt::tt_metal::CoreRange& cores,
    uint32_t k_block,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::device_operation::launch<ttnn::experimental::prim::BlockMatmulOperation>(
        ttnn::experimental::prim::BlockMatmulParams{
            .cores = tt::tt_metal::CoreRangeSet(cores),
            .k_block = k_block,
            .output_mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
            .compute_kernel_config = ttnn::init_device_compute_kernel_config(
                a.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, false, true)},
        ttnn::experimental::prim::BlockMatmulInputs{.a = a, .w = w})[0];
}

}  // namespace ttnn::experimental::kda
