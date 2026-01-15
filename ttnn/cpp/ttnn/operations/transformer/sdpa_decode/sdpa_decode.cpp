// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode.hpp"

#include <optional>
#include <utility>

#include "device/sdpa_decode_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
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

namespace ttnn::operations::transformer {

ttnn::Tensor ExecuteScaledDotProductAttentionDecode::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::vector<uint32_t>& cur_pos,
    const std::optional<const Tensor>& cur_pos_tensor,
    const std::optional<const Tensor>& attention_sink,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    uint32_t s = input_tensor_k.logical_shape()[-2];
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
    return ttnn::prim::sdpa_decode(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cur_pos_tensor,
        std::nullopt,
        attn_mask,
        attention_sink,
        is_causal,
        false,
        cur_pos,
        scale,
        sliding_window_size,
        memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        program_config,
        kernel_config_val,
        k_chunk_size,
        std::nullopt,
        std::nullopt,
        std::nullopt);
}

ttnn::Tensor ExecutePagedScaledDotProductAttentionDecode::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    const bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::optional<const Tensor>& cur_pos_tensor,
    const std::optional<const Tensor>& attention_sink,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();

    // Use k_chunk_size as override; if k_chunk_size == 0, figure it out in kernels
    // uint32_t k_chunk_size = get_chunk_size(s);
    uint32_t k_chunk_size = 0;
    if (program_config.has_value() && program_config.value().k_chunk_size > 0) {
        k_chunk_size = program_config.value().k_chunk_size;
        // assert chunk size must be power of 2 and multiple of 32
        TT_FATAL(
            (k_chunk_size & (k_chunk_size - 1)) == 0,
            "User provided k_chunk_size must be power of 2, got: {}",
            k_chunk_size);
        TT_FATAL(k_chunk_size % 32 == 0, "User provided k_chunk_size must be multiple of 32, got: {}", k_chunk_size);
    }

    // get chunk size and then pass to sdpa decode as an attribute for prgm cache
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa_decode(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cur_pos_tensor,
        page_table_tensor,
        attn_mask,
        attention_sink,
        is_causal,
        true,
        std::vector<uint32_t>(),
        scale,
        sliding_window_size,
        memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        program_config,
        kernel_config_val,
        k_chunk_size,
        std::nullopt,
        std::nullopt,
        std::nullopt);
}

ttnn::Tensor ExecuteFlashMultiLatentAttentionDecode::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::vector<uint32_t>& cur_pos,
    const std::optional<const Tensor>& cur_pos_tensor,
    const std::optional<const Tensor>& attention_sink,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();
    uint32_t s = input_tensor_k.logical_shape()[-2];
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

    return ttnn::prim::sdpa_decode(
        input_tensor_q,
        input_tensor_k,
        std::nullopt,
        cur_pos_tensor,
        std::nullopt,
        attn_mask,
        attention_sink,
        is_causal,
        false,
        cur_pos,
        scale,
        sliding_window_size,
        memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        program_config,
        kernel_config_val,
        k_chunk_size,
        std::nullopt,
        true,
        head_dim_v);
}

ttnn::Tensor ExecutePagedFlashMultiLatentAttentionDecode::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const ttnn::Tensor& page_table_tensor,
    const bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::optional<const Tensor>& cur_pos_tensor,
    const std::optional<const Tensor>& attention_sink,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch = input_tensor_q.storage_type() == StorageType::DEVICE
                                     ? input_tensor_q.device()->arch()
                                     : ttnn::GetDefaultDevice()->arch();

    // Use k_chunk_size as override; if k_chunk_size == 0, figure it out in kernels
    // uint32_t k_chunk_size = get_chunk_size(s);
    uint32_t k_chunk_size = 0;
    if (program_config.has_value() && program_config.value().k_chunk_size > 0) {
        k_chunk_size = program_config.value().k_chunk_size;
        // assert chunk size must be power of 2 and multiple of 32
        TT_FATAL(
            (k_chunk_size & (k_chunk_size - 1)) == 0,
            "User provided k_chunk_size must be power of 2, got: {}",
            k_chunk_size);
        TT_FATAL(k_chunk_size % 32 == 0, "User provided k_chunk_size must be multiple of 32, got: {}", k_chunk_size);
    }

    // get chunk size and then pass to sdpa decode as an attribute for prgm cache
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::sdpa_decode(
        input_tensor_q,
        input_tensor_k,
        std::nullopt,
        cur_pos_tensor,
        page_table_tensor,
        attn_mask,
        attention_sink,
        is_causal,
        true,
        std::vector<uint32_t>(),
        scale,
        sliding_window_size,
        memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        program_config,
        kernel_config_val,
        k_chunk_size,
        std::nullopt,
        true,
        head_dim_v);
}

}  // namespace ttnn::operations::transformer
