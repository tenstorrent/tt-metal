// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding.hpp"

#include "device/rotary_embedding_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"

namespace ttnn::operations::experimental::transformer {

ttnn::Tensor RotaryEmbeddingOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const std::optional<uint32_t> token_index,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(input_tensor.get_legacy_shape()[-1] % (tt::constants::TILE_WIDTH * 2) == 0,
             "Input X dimension ({}) must be divisible by {} for tiling.",
             input_tensor.get_legacy_shape()[-1],
             tt::constants::TILE_WIDTH * 2);

    uint32_t seq_len = input_tensor.get_legacy_shape()[-2];
    uint32_t B = input_tensor.get_legacy_shape()[0];
    uint32_t X = input_tensor.get_legacy_shape()[-1];

    TT_FATAL(cos_cache.get_legacy_shape() == sin_cache.get_legacy_shape(),
             "Cosine and Sine cache dimensions must match. Cos cache dimensions: {}, Sin cache dimensions: {}.",
             cos_cache.get_legacy_shape(),
             sin_cache.get_legacy_shape());

    TT_FATAL(cos_cache.get_legacy_shape()[0] == 1 && cos_cache.get_legacy_shape()[1] == 1 &&
                 cos_cache.get_legacy_shape()[-1] == X,
             "Cosine cache dimensions must match input dimensions. Expected (1, 1, {}), but got {}.",
             X,
             cos_cache.get_legacy_shape());

    if (token_index.has_value()) {
        seq_len = input_tensor.get_legacy_shape()[0];
        TT_FATAL(seq_len == 1,
                 "When token index is provided, sequence length must be 1. Current sequence length: {}.",
                 seq_len);

        TT_FATAL(cos_cache.get_legacy_shape()[-2] >= token_index,
                 "Cosine cache dimensions must cover the token index. Token index: {}, Cos cache dimension: {}.",
                 token_index.value(),
                 cos_cache.get_legacy_shape()[-2]);
    } else {
        TT_FATAL(
            cos_cache.get_legacy_shape()[-2] >= seq_len,
            "Cosine cache dimensions must cover the sequence length. Sequence length: {}, Cos cache dimension: {}.",
            seq_len,
            cos_cache.get_legacy_shape()[-2]);
    }

    auto arch = input_tensor.storage_type() == StorageType::DEVICE
                    ? input_tensor.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

    tt::tt_metal::MemoryConfig default_memory_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        default_memory_config = input_tensor.memory_config();
    }

    tt::tt_metal::LegacyShape input_pad_shape =
        ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
    ttnn::operations::experimental::auto_format::FormatParams input_format_params = {
        .pad_shape = input_pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};

    tt::tt_metal::LegacyShape cos_pad_shape =
        ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(cos_cache.get_legacy_shape());
    ttnn::operations::experimental::auto_format::FormatParams cos_format_params = {
        .pad_shape = cos_pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};

    tt::tt_metal::LegacyShape sin_pad_shape =
        ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(sin_cache.get_legacy_shape());
    ttnn::operations::experimental::auto_format::FormatParams sin_format_params = {
        .pad_shape = sin_pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};

    return operation::run_with_autoformat(
               RotaryEmbedding{seq_len, token_index, memory_config.value_or(default_memory_config), kernel_config_val},
               {input_tensor, cos_cache, sin_cache},
               {input_format_params, cos_format_params, sin_format_params},
               {Layout::TILE})
        .at(0);
}

}  // namespace ttnn::operations::experimental::transformer
