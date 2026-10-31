// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "combine_host.hpp"
#include "combine_program.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

// The combine's op-specific configuration (the MoE block's CombineTuning / tie rule / snap come from MoEBlockConfig).
struct CombineConfig {
    std::optional<uint32_t> n_acc;      // the accumulator count the fused head planned (pins combine_auto's A)
    std::optional<CombineMode> mode;    // ... and its mode
    uint32_t arena_bytes = 0;           // the combine's L1 budget a core; 0: the reference's sizing at the first call
    uint32_t flat_static_bytes = 0;     // (the reference's sizing: the flat program's static CBs, flat plan)
    uint32_t depth = 8;                 // landing pool slots per source chip (average)
    std::optional<uint32_t> piece_max;  // default max(7168, 2 Hr)

    bool operator==(const CombineConfig&) const = default;
};

struct RoutedExpertsCombineParams {
    CombineConfig combine;
    uint32_t S = 0;  // home tokens a chip
    uint32_t W = 0;  // the routed width Hr
    bool dispatch_in_program = false;
    // The flat expert's side (SEAM(flat): from its plan; the flat config itself joins the attributes when the flat
    // builder is integrated): its free cells, its y writers and their words.
    std::vector<tt::tt_metal::CoreCoord> cells;
    FlatHandoff flat;
    // Structural key of the resources (config, dims, mesh, fabric): hashed instead of the config.
    uint64_t plan_key = 0;
    // The combine's plan (cells, layout, LUT), a function of the fields above and the resources (not hashed).
    std::shared_ptr<const CombinePlan> plan;

    static constexpr auto attribute_names = std::forward_as_tuple("S", "W", "dispatch_in_program", "plan_key");
    auto attribute_values() const { return std::forward_as_tuple(S, W, dispatch_in_program, plan_key); }
};

struct RoutedExpertsCombineInputs {
    Tensor plan;                // the device plan [1, 1024] uint32 (the fused head's planner)
    Tensor table;               // the hand-off table [rows_cap / 32, 64] uint32 (the fused head's planner)
    Tensor out;                 // y_own [1, 1, S, Hr] bf16 row-major (persistent, the resources')
    Tensor bar;                 // the start barrier / zero gate words (persistent, every worker core)
    Tensor hcache;              // the senders' routed header cache (persistent DRAM)
    std::optional<Tensor> lut;  // per-chip accumulator roles: the senders' accumulator LUT (persistent DRAM)
    // SEAM(flat): xg, counts, regions, token_index, gidx, w_gu, w_d, w_rd, done, relay_words join here.
};

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
