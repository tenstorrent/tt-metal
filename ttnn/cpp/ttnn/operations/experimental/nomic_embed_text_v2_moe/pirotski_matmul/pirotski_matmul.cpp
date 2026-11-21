// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pirotski_matmul.hpp"

#include "device/pirotski_matmul_device_operation.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

Tensor pirotski_matmul(
    const Tensor& input_tensor,
    const Tensor& weight,
    const ttnn::experimental::prim::MinimalMatmulConfig& config,
    tt::tt_metal::DataType dtype,
    const std::optional<Tensor>& bias,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    bool resident_weights,
    uint32_t strip_m_tiles,
    bool strip_read_ahead,
    bool read_ahead) {
    // minimal_matmul's defaults, so the two ops compare bit for bit on the same arguments.
    const auto kernel_config = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/true);
    return ttnn::prim::pirotski_matmul(
        input_tensor,
        weight,
        bias,
        config,
        dtype,
        // A sharded input's layout does not fit the output's N; its buffer type does.
        memory_config.value_or(tt::tt_metal::MemoryConfig(
            tt::tt_metal::TensorMemoryLayout::INTERLEAVED, input_tensor.memory_config().buffer_type())),
        kernel_config,
        resident_weights,
        strip_m_tiles,
        strip_read_ahead,
        read_ahead);
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
