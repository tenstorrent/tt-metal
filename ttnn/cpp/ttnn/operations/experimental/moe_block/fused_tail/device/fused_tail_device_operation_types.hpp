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
    MemoryConfig output_memory_config;
    // The plans (owned by the resources, built once). Hash-excluded: plan_key, nslot and the tensor specs determine
    // them.
    std::shared_ptr<const TailEnginePlan> engine_plan;
    std::shared_ptr<const TailA2APlan> a2a_plan;

    static constexpr auto attribute_names =
        std::forward_as_tuple("kind", "plan_key", "math_fidelity", "nslot", "output_memory_config");
    auto attribute_values() const {
        return std::forward_as_tuple(kind, plan_key, math_fidelity, nslot, output_memory_config);
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
};

}  // namespace ttnn::operations::experimental::moe_block::fused_tail
