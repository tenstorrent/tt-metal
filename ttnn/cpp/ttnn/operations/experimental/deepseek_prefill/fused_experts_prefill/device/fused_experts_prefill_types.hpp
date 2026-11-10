// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill {

// Core geometry: a 12x10 worker rectangle cut into groups that each run one expert at a time. The group shape is
// chosen per call (see the program factory): 4x2 groups of 8 cores (3 x 5 = 15 groups), or 3x2 groups of 6
// (4 x 5 = 20 groups) when the expert's I tiles split evenly over 6 cores but not over 8.
inline constexpr uint32_t kGridX = 12;
inline constexpr uint32_t kGridY = 10;
struct GroupShape {
    uint32_t x;
    uint32_t y;
};
inline constexpr GroupShape kGroupShapes[] = {{4, 2}, {3, 2}};
inline constexpr uint32_t kMaxCoresPerGroup = 8;

// Tile rows of one expert processed per chunk. Bounded by L1 (the gathered SwiGLU activations scale
// with it); an expert with more rows runs as several chunks, re-reading its weights per chunk.
inline constexpr uint32_t kChunkTiles = 8;

// Tile rows processed together against one pass over a weight slot (M is the innermost matmul loop:
// every weight tile is unpacked once and multiplied into `kMBlock` DST accumulators). Bounded by DST
// (fp32 accumulation: 4 tiles) and by the L1 held by the tilized x block.
inline constexpr uint32_t kMBlock = 2;

// Tiles per row-major x segment: a token row is read / tilized as kt / kRmChunkTiles segments of
// kRmChunkTiles * 32 elements.
inline constexpr uint32_t kRmChunkTiles = 16;
// Max row-major x segments the reader keeps in flight per barrier (cb_rm holds exactly one such group;
// the actual group is the largest divisor of kt / kRmChunkTiles that is <= this).
inline constexpr uint32_t kRmGroupMax = 4;

// Tokens are addressed by the routing entry's token field (kTokenBits wide, see
// kernels/fused_experts_prefill_routing_entry.hpp). The cap decides what a prefill chunk costs, because one
// call reads EVERY expert's weights: a chunk split into two calls pays twice for the same weights.
//
// 1024 is the size that costs nothing extra. At 384 experts and top-6, T tokens put 6T / 384 on an average
// expert, and an expert's rows are rounded up to whole tile rows:
//   T =  512 -> 8 tokens/expert  -> 1.000 tile rows/expert (every expert one row, 16 of 32 rows used)
//   T = 1024 -> 16 tokens/expert -> 1.000 tile rows/expert (still one row, 16 of 32 rows used)
//   T = 2048 -> 32 tokens/expert -> ~1.45 tile rows/expert (the spread pushes many experts to a second row)
// So 512 -> 1024 doubles the tokens for exactly the same WEIGHT traffic, while 2048 already re-reads ~45% of
// them. The op's cost is not otherwise flat in T: the routing prologue runs on every core, and the x rows and
// partials are O(T) (see the reader's comment). Measured at TP4, 1024 tokens cost ~1.35x what 512 does, so the
// win from this cap is that a chunk is ONE call instead of two -- each call re-reading all the weights -- not
// that the extra tokens are free. L1 agrees: the routing lists in cb_meta (one list of T entries per owned
// expert, 20 experts in a group) take ~82 KiB of the 1536 KiB at 1024, but at 2048 the 3x2 group's best
// resident tier (dpp = 14, m_chunk = 8) goes from 1500 KiB to 1612 KiB and the factory would drop a tier and
// stream weights per M-block.
inline constexpr uint32_t kMaxTokens = 1024;
// Selected experts per token: the per-token selection lives on the reader's stack, and the slot index
// takes kSlotBits of a routing entry.
inline constexpr uint32_t kMaxTopK = 16;

// Non-tensor parameters of the prefill routed-expert FFN.
struct operation_attributes_t {
    uint32_t intermediate_size{};
    // Clamp limit of the SwiGLU: silu(min(gate, limit)) * clamp(up, -limit, +limit).
    float swiglu_limit{};
    uint32_t top_k{};
    float routed_scaling_factor{};
    float routing_eps{};
    tt::tt_metal::MemoryConfig output_memory_config{};
};

// All tensors flowing in/out of the operation. The routing is computed ON DEVICE by every core (each
// core needs the token lists of its own experts only, but derives them from the full routing).
//
//   x_tok:             [1, 1, T, H] ROW_MAJOR BFLOAT16, DRAM interleaved (T a multiple of 32, <= kMaxTokens).
//   routing_scores:    [1, 1, T, E] TILE BFLOAT16, the UNBIASED scores that become the weights.
//   routing_indices:   [1, 1, T, top_k] TILE UINT16 / BFLOAT16 -- the router's selected expert ids; or
//   ranking_scores:    [1, 1, T, E] TILE BFLOAT16 -- the row ranked on device (exactly one of the two).
//   gate_up_weights:   one [H, 2I] tensor per expert in the DECODE layout: gate/up columns interleaved
//                      per 32-column tile, DRAM ND-sharded [H, 64] (one I-tile per shard).
//   down_weights:      one [I, H] tensor per expert in the DECODE layout: DRAM ND-sharded [I, 64].
struct tensor_args_t {
    const Tensor& x_tok;
    const Tensor& routing_scores;
    std::optional<Tensor> routing_indices;
    std::optional<Tensor> ranking_scores;
    std::vector<Tensor> gate_up_weights;
    std::vector<Tensor> down_weights;
};

// The device op writes the routed FFN output per (slot, token): [1, top_k, T, H] ROW_MAJOR bfloat16,
// slot j of token t at row j * T + t holding w[t, j] * FFN_e(x_t); the wrapper reduces over the slots.
using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill
