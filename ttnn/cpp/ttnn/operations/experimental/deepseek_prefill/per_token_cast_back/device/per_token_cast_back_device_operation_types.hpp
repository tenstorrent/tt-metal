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
    // op decompresses only this device's valid expert-region rows, reading each token's per-128-block
    // scales from the tail of its metadata row (input_scale is unused).
    bool masked = false;
    uint32_t experts_per_chip = 0;
    uint32_t dispatch_group_size = 0;
};

struct PerTokenCastBackInputs {
    const Tensor& input_e4m3;
    // Plain-path per-token scales. Unused in masked mode, where scales ride in the metadata tail.
    const Tensor& input_scale;
    // Masked-mode routing tensors (all set together, or all nullopt).
    std::optional<Tensor> expert_token_counts = std::nullopt;
    std::optional<Tensor> expert_region_offsets = std::nullopt;
    std::optional<Tensor> metadata = std::nullopt;
};

}  // namespace ttnn::experimental::prim::per_token_cast_back
