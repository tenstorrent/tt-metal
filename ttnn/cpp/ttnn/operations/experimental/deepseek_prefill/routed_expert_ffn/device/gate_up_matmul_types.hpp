// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

// Parameters captured at program-cache time (determine whether to recompile).
struct GateUpMatmulParams {
    // Tile dimensions — the full matrix dimensions expressed in tiles (tile = 32x32 elements).
    // A change to any of these triggers recompilation because loop bounds are compile-time constants.
    uint32_t Mt;  // Number of tiles along M (rows of the activation matrix x).
    uint32_t Kt;  // Number of tiles along K (shared inner dimension: columns of x, rows of weight matrices).
    uint32_t Nt;  // Number of tiles along N (columns of the weight matrices gate_proj / up_proj).

    // Block tiling — how the Mt×Kt×Nt iteration space is partitioned across Tensix cores.
    // Each core processes one (M_block_size × K_block_size) × (K_block_size × N_block_size) matmul.
    // subblock_h × subblock_w is the innermost register-file tile that the compute engine accumulates
    // in a single pass; it must evenly divide M_block_size × N_block_size respectively.
    uint32_t M_block_size;  // Tiles along M assigned to a single core per iteration.
    uint32_t K_block_size;  // Tiles along K consumed per inner-loop iteration (controls L1 reuse).
    uint32_t N_block_size;  // Tiles along N assigned to a single core per iteration.
    uint32_t subblock_h;    // Tiles along M in the innermost compute subblock (<= M_block_size).
    uint32_t subblock_w;    // Tiles along N in the innermost compute subblock (<= N_block_size).

    // Output configuration — determines where and in what format the result is written.
    MemoryConfig output_mem_config;  // DRAM / L1 placement and sharding strategy for the output tensor.
    DataType output_dtype;           // Element type of the output (e.g. bfloat16, float32).

    static constexpr auto attribute_names = std::forward_as_tuple(
        "Mt",
        "Kt",
        "Nt",
        "M_block_size",
        "K_block_size",
        "N_block_size",
        "subblock_h",
        "subblock_w",
        "output_mem_config",
        "output_dtype");

    auto attribute_values() const {
        return std::forward_as_tuple(
            Mt,
            Kt,
            Nt,
            M_block_size,
            K_block_size,
            N_block_size,
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
