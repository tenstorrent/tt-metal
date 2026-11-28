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
    // Bring-up option server order (deepseek_v41_flash_d_p NC.9; default none = the source op). Set: the input is the
    // engine's ids in the server's row order gathered over the mesh rows ([SP, 1, W] uint32, every chip the same),
    // and each chip's reader picks its natural-order rows of the chunk at ``server_start`` itself (mesh row r: natural
    // positions [r W, (r + 1) W)), positions >= ``server_valid`` and ids > clamp_max read as clamp_max: the output is
    // ttnn.embedding of tt/page_copy.IdsReorder's ids, without that kernel or its ids tensor. clamp_max is part of
    // the program hash; server_start / server_valid are per-call values (runtime args, never in the hash).
    std::optional<uint32_t> clamp_max = std::nullopt;
    uint32_t server_start = 0;
    uint32_t server_valid = 0;

    static constexpr auto attribute_names =
        std::forward_as_tuple("output_mem_config", "tilized", "embeddings_type", "pad_token", "max_grid", "clamp_max");
    auto attribute_values() const {
        return std::forward_as_tuple(output_mem_config, tilized, embeddings_type, pad_token, max_grid, clamp_max);
    }
};

// NC.9 server order: the reader's per-call values of the chunk at ``server_start`` (tt/page_copy.IdsReorder's)
struct ServerOrderValues {
    uint32_t iu = 0;     // (start mod W) / 32
    uint32_t cs = 0;     // (start / W) mod SP: the mesh row holding the chunk's start
    uint32_t valid = 0;  // natural positions below it are ids, the rest pad
};
inline ServerOrderValues server_order_values(const EmbeddingParams& p, uint32_t sp, uint32_t w) {
    return ServerOrderValues{
        .iu = (p.server_start % w) / 32,
        .cs = (p.server_start / w) % sp,
        .valid = std::min<uint32_t>(p.server_valid, sp * w),
    };
}

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
