// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "head_program.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

struct MoEHeadParams {
    // The head's structure: the planned layout, the model, the switches its kernels see.
    HeadProgramConfig config;
    // MoEBlockResources::plan_key(): the config, dims, mesh shape and fabric the plans were built from.
    ttsl::hash::hash_t plan_key = 0;
    // Hash-excluded: the host model (made from config on this mesh), the persistent words' addresses (head_words'
    // order) and the arena (an L1 tensor - not a tensor arg: the op rebinds its CBs and args on every call).
    std::shared_ptr<const HeadModel> model;
    std::vector<uint32_t> words;
    Tensor arena{};
    // This call's split (MoEBlockConfig.chunk_splits: a padded chunk's real rows depend on it - one cached program
    // each). The head's row blocks are config.shape.row_blocks (x and y hold every block's rows).
    uint32_t split = 0;

    static constexpr auto attribute_names = std::forward_as_tuple("plan_key");
    auto attribute_values() const { return std::forward_as_tuple(plan_key); }
};

struct MoEHeadInputs {
    Tensor x{};             // [1, 1, rows, H / TP] bf16 TILE, DRAM: this chip's shard of the residual
    Tensor w{};             // the shared expert's weights (prepare_moe_head_weights)
    Tensor w_router{};      // W_r^T [E, H / TP] bf16 TILE
    Tensor router_bias{};   // [E, 32] fp32 TILE
    Tensor router_index{};  // [E, 32] uint16 TILE
    // The resources' persistent DRAM tensors the head writes (outputs) or exchanges through.
    Tensor y{};                                                  // the shared expert's reduced output (returned)
    Tensor rows{};                                               // the owners' joint routing rows
    Tensor dispatch_buffer{};                                    // the dispatch buffer
    Tensor route{};                                              // the owned tokens' route words
    Tensor slots{};                                              // the routing exchange's slots
    Tensor counts{}, regions{}, token_index{}, table{}, plan{};  // the combine plan the routing planner writes
    // A padded chunk: main's chunked-prefill metadata scalars actual_start / actual_end (one uint32 each in DRAM,
    // replicated), read on device each call; the chunk in main's rotated layout over the SP rows. Padding tokens are
    // not routed (no route words, not planned). Absent: every row is real.
    std::optional<Tensor> actual_start, actual_end;
    // Or the forward's metadata words (common/prefill/prefill_chunk.py: at forward_chunks 1 (slot, start, end), above
    // {num_chunks, 0} + forward_chunks (slot, start, end) records; one uint32 DRAM tensor, replicated), read on device
    // every call: MoEBlockConfig.forward_chunks chunks packed in the forward, each its own real rows.
    std::optional<Tensor> metadata;
};

}  // namespace ttnn::operations::experimental::moe_block::moe_head
