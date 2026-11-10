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

#include "moe_tail_plan.hpp"

namespace ttnn::operations::experimental::moe_block::moe_tail {

// Which program: TailEngine (a latent model: gather the latent over TP, rmsnorm, up projection, + the shared expert) or
// the all-to-all (no latent: tilize the owner rows, + the shared expert, TP all-to-all).
enum class TailKind : uint8_t { Engine = 0, AllToAll = 1 };

struct MoETailParams {
    TailKind kind = TailKind::AllToAll;
    // MoEBlockResources::plan_key(): the config, dims, mesh shape and fabric the plans were built from.
    ttsl::hash::hash_t plan_key = 0;
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi2;  // the engine's up projection
    // The engine's ring slots: chosen from the free L1 when its plan was built, so hashed on its own.
    uint32_t ring_slots = 0;
    // The head's row blocks (MoEHeadOutputs.row_blocks): the call runs that many blocks' rows, block after block in
    // y_own, shared_expert and the output (the owner rows in the head's block layout); the kernels loop over them.
    uint32_t row_blocks = 1;
    // A padded forward (the all-to-all): the call's first row loop over the chunk (split x row_blocks), the forward's
    // chunks (MoEBlockConfig.forward_chunks) and the SP row's tokens - its padding row tiles are skipped.
    uint32_t first_block = 0;
    uint32_t forward_chunks = 1;
    uint32_t row_tokens = 0;
    MemoryConfig output_memory_config;
    // The plans (owned by the resources, built once). Hash-excluded: plan_key, ring_slots and the tensor specs
    // determine them.
    std::shared_ptr<const TailEnginePlan> engine_plan;
    std::shared_ptr<const TailAllToAllPlan> all_to_all_plan;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "kind",
        "plan_key",
        "math_fidelity",
        "ring_slots",
        "row_blocks",
        "first_block",
        "forward_chunks",
        "row_tokens",
        "output_memory_config");
    auto attribute_values() const {
        return std::forward_as_tuple(
            kind,
            plan_key,
            math_fidelity,
            ring_slots,
            row_blocks,
            first_block,
            forward_chunks,
            row_tokens,
            output_memory_config);
    }
};

struct MoETailInputs {
    Tensor y_own;                         // the combine's owner rows [1, 1, S/N, Hr], ROW_MAJOR, DRAM
    std::optional<Tensor> shared_expert;  // engine: [1, 1, S/SP, H/TP]; all-to-all: [1, 1, S/N, H] or none; TILE, DRAM
    std::optional<Tensor> w_up;           // engine: prepare_moe_tail_weights' W' (DRAM, bank-sharded regions)
    // The resources' persistent words (tensor args, so a cache hit rebinds their addresses)
    Tensor ready{};  // the fence's ready word on every port core (the per-row-tile words are per-call CBs)
    std::optional<Tensor> arrival;  // all-to-all: the ports' arrival counters
    std::optional<Tensor> staging;  // all-to-all: z [1, 1, S/N, H] TILE, DRAM (the blocks for other chips)
    std::optional<Tensor> output;   // preallocated output
    // A padded forward (moe_head's): one chunk's actual_start / actual_end, or the forward's metadata words
    std::optional<Tensor> actual_start;
    std::optional<Tensor> actual_end;
    std::optional<Tensor> metadata;
};

}  // namespace ttnn::operations::experimental::moe_block::moe_tail
