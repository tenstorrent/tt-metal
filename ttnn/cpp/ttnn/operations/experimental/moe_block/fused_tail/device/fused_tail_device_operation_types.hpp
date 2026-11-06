// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include <tt-metalium/base_types.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "fused_tail_plan.hpp"

namespace ttnn::operations::experimental::moe_block::fused_tail {

// Which program: TailEngine (a latent model: gather the latent over TP, rmsnorm, up projection, + se) or TailA2A (no
// latent: tilize the owner rows, + se, TP all-to-all).
enum class TailKind : uint8_t { Engine = 0, A2A = 1 };

struct FusedTailParams {
    TailKind kind = TailKind::A2A;
    // MoEBlockResources::plan_key(): the config, dims, mesh shape and fabric the plans were built from.
    ttsl::hash::hash_t plan_key = 0;
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi2;  // the engine's up projection
    // The engine's ring slots: chosen from the free L1 when its plan was built, so hashed on its own.
    uint32_t nslot = 0;
    // Token loops in one call (MoEBlockConfig.token_loops k): the call runs k loops' rows, one loop's after the
    // other's in y_own, shared_expert and the output; one program a chip whose kernels loop (mb_loop.hpp). 1: one.
    uint32_t loops = 1;
    // A padded forward (TailA2A): the call's first loop of the forward (MoEBlockConfig splits x token loops), the
    // forward's chunks (MoEBlockConfig.forward_chunks) and the SP row's tokens - its padding row tiles are skipped.
    uint32_t loop = 0;
    uint32_t fwd_chunks = 1;
    uint32_t row_tokens = 0;
    MemoryConfig output_memory_config;
    // The plans (owned by the resources, built once). Hash-excluded: plan_key, nslot and the tensor specs determine
    // them.
    std::shared_ptr<const TailEnginePlan> engine_plan;
    std::shared_ptr<const TailA2APlan> a2a_plan;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "kind",
        "plan_key",
        "math_fidelity",
        "nslot",
        "loops",
        "loop",
        "fwd_chunks",
        "row_tokens",
        "output_memory_config");
    auto attribute_values() const {
        return std::forward_as_tuple(
            kind, plan_key, math_fidelity, nslot, loops, loop, fwd_chunks, row_tokens, output_memory_config);
    }
};

struct FusedTailInputs {
    Tensor y_own;                         // the combine's owner rows [1, 1, S/N, Hr], ROW_MAJOR, DRAM
    std::optional<Tensor> shared_expert;  // engine: [1, 1, S/SP, H/TP]; a2a: [1, 1, S/N, H] or none; TILE, DRAM
    std::optional<Tensor> w_up;           // engine: prepare_fused_tail_weights' W' (DRAM, bank-sharded regions)
    // The resources' persistent words (tensor args, so a cache hit rebinds their addresses)
    Tensor ready{};                 // the fence's ready word on every port core (the per-row-tile words are per-call CBs)
    std::optional<Tensor> arrival;  // a2a: the ports' arrival counters
    std::optional<Tensor> staging;     // a2a: z [1, 1, S/N, H] TILE, DRAM (the blocks for other chips)
    std::optional<Tensor> output;      // preallocated output
    std::optional<Tensor> loop_words;  // token loops in one call: the loop words of every core (mb_loop.hpp)
    // A padded forward (fused_head's): one chunk's actual_start / actual_end, or the forward's metadata words
    std::optional<Tensor> actual_start;
    std::optional<Tensor> actual_end;
    std::optional<Tensor> metadata;
};

}  // namespace ttnn::operations::experimental::moe_block::fused_tail
