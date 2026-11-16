// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_bh.hpp"

#include <vector>

#include "device/sdpa_decode_bh_device_operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::llama31_8b_v3 {

ttnn::Tensor sdpa_decode_bh(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    const ttnn::Tensor& cur_pos_tensor,
    const ttnn::operations::transformer::SDPAProgramConfig& program_config,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    bool multi_pos) {
    const auto& q = input_tensor_q.logical_shape();
    const auto& k = input_tensor_k.logical_shape();
    TT_FATAL(
        input_tensor_q.device()->arch() == tt::ARCH::BLACKHOLE && input_tensor_q.layout() == Layout::TILE &&
            q.rank() == 4 && q[0] == 1 && q[2] == 32 && (q[3] == 128 || q[3] == 64) && k.rank() == 4 && k[1] == 8 &&
            k[2] == 32 && k[3] == q[3] && input_tensor_k.dtype() == DataType::BFLOAT8_B &&
            input_tensor_v.dtype() == DataType::BFLOAT8_B && program_config.k_chunk_size == 0,
        "sdpa_decode_bh only covers the Llama-3.1-8B (head_dim 128) and Llama-3.2-1B (head_dim 64) paged decode on "
        "Blackhole");
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);
    return sdpa_decode_bh_detail::sdpa_decode(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cur_pos_tensor,
        page_table_tensor,
        std::nullopt,
        std::nullopt,
        true,
        true,
        std::vector<uint32_t>(),
        scale,
        std::nullopt,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        program_config,
        kernel_config_val,
        0,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        0,
        multi_pos);
}

}  // namespace ttnn::experimental::llama31_8b_v3
