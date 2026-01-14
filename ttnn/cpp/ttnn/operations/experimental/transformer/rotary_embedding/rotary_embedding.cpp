// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.hpp"

#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"

namespace ttnn::operations::experimental::transformer {

ttnn::Tensor RotaryEmbeddingOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const std::optional<uint32_t> token_index,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    using namespace tt::constants;

    using tt::tt_metal::PadValue;
    TT_FATAL(
        input_tensor.padded_shape()[-1] % (TILE_WIDTH * 2) == 0,
        "Input X dimension ({}) must be divisible by {} for tiling.",
        input_tensor.padded_shape()[-1],
        TILE_WIDTH * 2);

    uint32_t seq_len = input_tensor.padded_shape()[-2];
    uint32_t X = input_tensor.padded_shape()[-1];

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(
        cos_cache.padded_shape() == sin_cache.padded_shape(),
        "Cosine and Sine cache dimensions must match. Cos cache dimensions: {}, Sin cache dimensions: {}.",
        cos_cache.padded_shape(),
        sin_cache.padded_shape());

    TT_FATAL(
        cos_cache.padded_shape()[0] == 1 && cos_cache.padded_shape()[1] == 1 && cos_cache.padded_shape()[-1] == X,
        "Cosine cache dimensions must match input dimensions. Expected (1, 1, {}), but got {}.",
        X,
        cos_cache.padded_shape());

    if (token_index.has_value()) {
        seq_len = input_tensor.padded_shape()[0];
        TT_FATAL(
            seq_len == 1,
            "When token index is provided, sequence length must be 1. Current sequence length: {}.",
            seq_len);

        TT_FATAL(
            cos_cache.padded_shape()[-2] >= token_index,
            "Cosine cache dimensions must cover the token index. Token index: {}, Cos cache dimension: {}.",
            token_index.value(),
            cos_cache.padded_shape()[-2]);
    } else {
        TT_FATAL(
            cos_cache.padded_shape()[-2] >= seq_len,
            "Cosine cache dimensions must cover the sequence length. Sequence length: {}, Cos cache dimension: {}.",
            seq_len,
            cos_cache.padded_shape()[-2]);
    }

    auto arch = input_tensor.device()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

    tt::tt_metal::MemoryConfig default_memory_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        default_memory_config = input_tensor.memory_config();
    }

    auto padded_shape_input = ttnn::operations::data_movement::pad_to_tile_shape(input_tensor.padded_shape());
    auto padded_shape_cos = ttnn::operations::data_movement::pad_to_tile_shape(cos_cache.padded_shape());
    auto padded_shape_sin = ttnn::operations::data_movement::pad_to_tile_shape(sin_cache.padded_shape());
    Tensor formatted_input =
        ttnn::tilize_with_val_padding(input_tensor, padded_shape_input, PadValue(0.0f), input_tensor.memory_config());
    Tensor formatted_cos =
        ttnn::tilize_with_val_padding(cos_cache, padded_shape_cos, PadValue(0.0f), cos_cache.memory_config());
    Tensor formatted_sin =
        ttnn::tilize_with_val_padding(sin_cache, padded_shape_sin, PadValue(0.0f), sin_cache.memory_config());

    return ttnn::prim::rotary_embedding(
        formatted_input,
        formatted_cos,
        formatted_sin,
        seq_len,
        token_index,
        memory_config.value_or(default_memory_config),
        kernel_config_val);
}

}  // namespace ttnn::operations::experimental::transformer
