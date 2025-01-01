// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "speculative_sdpa_decode.hpp"

#include "device/speculative_sdpa_decode_op.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace {
inline uint32_t get_chunk_size(uint32_t s) {
    /*
    # find maximum power of 2 divisor of s
    for i in range(1, s):
        if s % (2**(i+1)) != 0:
            break
    */
    uint32_t i = 1;
    for (; i < s; i++) {
        if (s % (1 << (i + 1)) != 0) {
            break;
        }
    }
    return std::min(512, 1 << i);
}
}  // namespace

namespace ttnn::operations::experimental::transformer {

using SDPAProgramConfig = ttnn::operations::transformer::SDPAProgramConfig;

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>
ExecuteSpeculativeScaledDotProductAttentionDecode::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    std::optional<float> lambda_,
    const bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::vector<uint32_t>& cur_pos,
    const std::optional<const Tensor>& cur_pos_tensor,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<Tensor>& priority_tensor) {
    auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                    ? input_tensor_q.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    uint32_t s = input_tensor_k.get_logical_shape()[-2];
    uint32_t k_chunk_size = get_chunk_size(s);
    if (program_config.has_value() && program_config.value().k_chunk_size > 0) {
        k_chunk_size = program_config.value().k_chunk_size;
        // assert chunk size must be power of 2 and multiple of 32
        TT_FATAL(
            (k_chunk_size & (k_chunk_size - 1)) == 0,
            "User provided k_chunk_size must be power of 2, got: {}",
            k_chunk_size);
        TT_FATAL(k_chunk_size % 32 == 0, "User provided k_chunk_size must be multiple of 32, got: {}", k_chunk_size);
    } else {
        TT_FATAL(
            k_chunk_size % 32 == 0,
            "Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: {}",
            k_chunk_size);
    }

    // get chunk size and then pass to sdpa decode as an attribute for prgm cache
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    auto output_tensors = operation::run(
        SpeculativeScaledDotProductAttentionDecode{
            .lambda_ = lambda_,
            .is_causal = is_causal,
            .cur_pos = cur_pos,
            .scale = scale,
            .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
            .program_config = program_config,
            .compute_kernel_config = kernel_config_val,
            .k_chunk_size = k_chunk_size,
            .paged_attention = false},
        {input_tensor_q, input_tensor_k, input_tensor_v},
        {cur_pos_tensor, std::nullopt, attn_mask, priority_tensor},
        {},
        queue_id);

    TT_FATAL(output_tensors.size() == 4, "Expected 4 output tensors, got: {}", output_tensors.size());
    return std::make_tuple(
        std::move(output_tensors[0]),
        std::move(output_tensors[1]),
        std::move(output_tensors[2]),
        std::move(output_tensors[3]));
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>
ExecuteSpeculativeScaledDotProductAttentionDecode::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    std::optional<float> lambda_,
    const bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::vector<uint32_t>& cur_pos,
    const std::optional<const Tensor>& cur_pos_tensor,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<Tensor>& priority_tensor) {
    return invoke(
        DefaultQueueId,
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        lambda_,
        is_causal,
        attn_mask,
        cur_pos,
        cur_pos_tensor,
        scale,
        memory_config,
        program_config,
        compute_kernel_config,
        priority_tensor);
}

}  // namespace ttnn::operations::experimental::transformer
