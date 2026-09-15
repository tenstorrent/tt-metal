// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pack_convolution_carry.hpp"
#include "device/pack_convolution_carry_device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor pack_convolution_carry(
    const ttnn::Tensor& input,
    const ttnn::Tensor& wrap_indicator,
    uint32_t wrap_row,
    uint32_t history_rows,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE && input.buffer() != nullptr,
        "pack_convolution_carry: input must be an allocated device tensor");
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    auto outputs = ttnn::experimental::prim::pack_convolution_carry(
        input, wrap_indicator, wrap_row, history_rows, output_memory_config, kernel_config);
    return outputs[0];
}

}  // namespace ttnn::experimental::kda
