// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::per_token_cast_back {

struct PerTokenCastBackParams {
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::MemoryConfig output_memory_config;
    // Masked decompress mode (see per_token_cast_back.hpp). When false the op behaves exactly as
    // before. When true, the three optional tensors in PerTokenCastBackInputs are populated and the
    // op decompresses only this device's valid expert-region rows, gathering scale by token_idx.
    bool masked = false;
    uint32_t experts_per_chip = 0;
    uint32_t dispatch_group_size = 0;
};

struct PerTokenCastBackInputs {
    const Tensor& input_e4m3;
    const Tensor& input_scale;
    // Masked-mode routing tensors (all set together, or all nullopt).
    std::optional<Tensor> expert_token_counts = std::nullopt;
    std::optional<Tensor> expert_region_offsets = std::nullopt;
    std::optional<Tensor> metadata = std::nullopt;
};

}  // namespace ttnn::experimental::prim::per_token_cast_back
