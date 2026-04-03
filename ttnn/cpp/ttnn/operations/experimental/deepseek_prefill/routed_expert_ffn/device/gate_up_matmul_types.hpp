// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

// Parameters captured at program-cache time (determine whether to recompile).
struct GateUpMatmulParams {
    // Tile dimensions
    uint32_t M_tiles;
    uint32_t K_tiles;
    uint32_t N_tiles;
    // Block tiling
    uint32_t M_block_tiles;
    uint32_t K_block_tiles;
    uint32_t N_block_tiles;
    uint32_t subblock_h;
    uint32_t subblock_w;
    // Output
    MemoryConfig output_mem_config;
    DataType output_dtype;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "M_tiles",
        "K_tiles",
        "N_tiles",
        "M_block_tiles",
        "K_block_tiles",
        "N_block_tiles",
        "subblock_h",
        "subblock_w",
        "output_mem_config",
        "output_dtype");

    auto attribute_values() const {
        return std::forward_as_tuple(
            M_tiles,
            K_tiles,
            N_tiles,
            M_block_tiles,
            K_block_tiles,
            N_block_tiles,
            subblock_h,
            subblock_w,
            output_mem_config,
            output_dtype);
    }
};

struct GateUpMatmulInputs {
    Tensor x;          // (M, K) — input activations
    Tensor gate_proj;  // (K, N) — gate projection weights
    Tensor up_proj;    // (K, N) — up projection weights
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
