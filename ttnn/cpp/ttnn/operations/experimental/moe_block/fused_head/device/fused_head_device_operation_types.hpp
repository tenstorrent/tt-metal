// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>

#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "head_program.hpp"

namespace ttnn::operations::experimental::moe_block::fused_head {

// Where the address arguments of the cached programs sit (one HeadPatches a chip): filled by the build of the
// programs, or - a cache hit on programs another resources object built - on that hit, once.
struct HeadPatchStore {
    std::mutex mutex;
    std::vector<HeadPatches> chips;
    // per cached program (each chip's is its own Program): the addresses its args were last patched with
    std::map<const tt::tt_metal::Program*, HeadLastAddresses> last;
};

struct FusedHeadParams {
    // The head's structure: the planned layout, the model, the switches its kernels see.
    HeadProgramConfig config;
    // MoEBlockResources::plan_key(): the config, dims, mesh shape and fabric the plans were built from.
    ttsl::hash::hash_t plan_key = 0;
    // Hash-excluded: the host model (made from config on this mesh), the persistent words' addresses (head_words'
    // order), the arena (an L1 tensor - not a tensor arg: its CBs and args are rebound by the op only when it moves)
    // and the patch store.
    std::shared_ptr<const HeadModel> model;
    std::vector<uint32_t> words;
    Tensor arena{};
    std::shared_ptr<HeadPatchStore> patches;

    static constexpr auto attribute_names = std::forward_as_tuple("plan_key");
    auto attribute_values() const { return std::forward_as_tuple(plan_key); }
};

struct FusedHeadInputs {
    Tensor x{};                    // [1, 1, rows, H / TP] bf16 TILE, DRAM: this chip's shard of the residual
    std::optional<Tensor> x_full;  // the DRAM fallback: x gathered over TP [1, 1, rows, H]
    Tensor w{};                    // the shared expert's weights (prepare_fused_head_weights)
    Tensor w_router{};             // W_r^T [E, H / TP] bf16 TILE
    Tensor router_bias{};          // [E, 32] fp32 TILE
    Tensor router_index{};         // [E, 32] uint16 TILE
    // The resources' persistent DRAM tensors the head writes (outputs) or exchanges through.
    Tensor y{};                                        // the shared expert's reduced output (returned)
    Tensor rows{};                                     // the owners' joint routing rows
    Tensor xbuf{};                                     // the dispatch buffer
    Tensor route{};                                    // the owned tokens' route words
    Tensor slots{};                                    // the routing exchange's slots
    Tensor fslots{};                                   // its forwarding slots
    Tensor counts{}, regions{}, token_index{}, table{}, plan{};  // the combine plan the routing planner writes
    // A padded chunk: main's chunked-prefill metadata scalars actual_start / actual_end (one uint32 each in DRAM,
    // replicated), read on device each call; the chunk in main's rotated layout over the SP rows. Padding tokens are
    // not routed (no route words, not planned). Absent: every row is real.
    std::optional<Tensor> actual_start, actual_end;
};

}  // namespace ttnn::operations::experimental::moe_block::fused_head
