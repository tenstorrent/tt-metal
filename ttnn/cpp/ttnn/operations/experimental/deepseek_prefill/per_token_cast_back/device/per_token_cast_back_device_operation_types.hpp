// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::per_token_cast_back {

struct PerTokenCastBackParams {
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::MemoryConfig output_memory_config;
    // Masked decompress mode (see per_token_cast_back.hpp). When false the op behaves exactly as
    // before. When true, input_e4m3 is a dispatch buffer and only this device's valid expert-region
    // rows are decompressed; each valid row's per-128-block fp32 scales are read from the metadata
    // tail (fields 5..) — input_scale is unused in this mode.
    bool masked = false;
    uint32_t experts_per_chip = 0;
    uint32_t dispatch_group_size = 0;
};

struct PerTokenCastBackInputs {
    const Tensor& input_e4m3;
    const Tensor& input_scale;
    // Masked-mode routing tensors (all set together, or all nullopt). metadata carries the per-token
    // fp32 scale tail (fields 5..) written by the scaled dispatch path.
    std::optional<Tensor> expert_token_counts = std::nullopt;
    std::optional<Tensor> expert_region_offsets = std::nullopt;
    std::optional<Tensor> metadata = std::nullopt;
};

}  // namespace ttnn::experimental::prim::per_token_cast_back
