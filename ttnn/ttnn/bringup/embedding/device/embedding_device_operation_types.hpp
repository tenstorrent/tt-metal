// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::bringup {

enum class EmbeddingsType { GENERIC, PADDED, BINARY };
enum class EmbeddingsIndexType { UINT32, BFP16 };

struct EmbeddingParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    bool tilized = false;
    EmbeddingsType embeddings_type = EmbeddingsType::GENERIC;
    std::optional<uint32_t> pad_token;
    // Bring-up option max_grid (deepseek_v41_flash_d_p FIX.15, agent rule 18: at most 110 Tensix cores an op on a
    // Galaxy): an interleaved output's work split runs on the rectangle (0, 0) .. (max_grid.x - 1, max_grid.y - 1)
    // of the device's compute grid. Default (none) = the whole grid (the source op). Part of the program hash.
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt;
};

// The device's compute grid cut to max_grid (FIX.15)
inline tt::tt_metal::CoreCoord embedding_grid(tt::tt_metal::CoreCoord g, const EmbeddingParams& p) {
    if (p.max_grid.has_value()) {
        g.x = std::max<std::size_t>(1, std::min<std::size_t>(g.x, p.max_grid->x));
        g.y = std::max<std::size_t>(1, std::min<std::size_t>(g.y, p.max_grid->y));
    }
    return g;
}

struct EmbeddingInputs {
    Tensor input_tensor_arg;
    Tensor weight_arg;
    std::optional<Tensor> optional_output_tensor;
};

}  // namespace ttnn::prim::bringup
