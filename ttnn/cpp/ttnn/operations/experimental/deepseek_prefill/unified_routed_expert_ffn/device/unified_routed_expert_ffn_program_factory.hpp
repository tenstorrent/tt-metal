// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "unified_routed_expert_ffn_types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

struct UnifiedRoutedExpertFfnProgramFactory {
    // The L1-fitted (per_core_M, in0_block_w_gu) create_descriptor would build with now: the
    // circular buffers are sized to the free L1, so the program-cache key includes this plan.
    static std::pair<uint32_t, uint32_t> select_l1_plan(
        const UnifiedRoutedExpertFfnParams& operation_attributes, const UnifiedRoutedExpertFfnInputs& tensor_args);

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UnifiedRoutedExpertFfnParams& operation_attributes,
        const UnifiedRoutedExpertFfnInputs& tensor_args,
        Tensor& tensor_return_value);

    // For a caller that builds this program into its own (hybrid_routed_expert_ffn overlapped with combine).
    struct EmbeddedOptions {
        // The grid's first row: the caller keeps the rows above (combine takes rows 0-1).
        uint32_t origin_y = 0;
        // Experts above this many tokens are walked first, the rest after (0: one walk in local order). At least
        // 96, so the first walk never ends in a partial eighth wave, which leaves rows' rings behind.
        uint32_t big_first_threshold = 0;
        // The writers report each expert to combine's collector (HYB_EXPERT_DONE_RT_BASE) as soon as they are
        // done with it, at its place in adaptive_chunk::predicted_order.
        bool report_predicted_order = false;
        // Each whole-grid expert's chunks run from the one holding the rows combine's walks open on (the ring chip
        // diametrically opposite, URE_FAR_RT_BASE), fanning out from there (adaptive_chunk::chunk_at), so both
        // ring directions get rows to send from the first chunk on. The table rides past the counts.
        bool far_chunk_table = false;
    };
    static tt::tt_metal::ProgramDescriptor create_embedded_descriptor(
        const UnifiedRoutedExpertFfnParams& operation_attributes,
        const UnifiedRoutedExpertFfnInputs& tensor_args,
        Tensor& tensor_return_value,
        const EmbeddedOptions& options);

    // The wave planner configuration the kernels would be built with now, which is what anything predicting
    // this program's finish order (adaptive_chunk::predicted_order) must plan with.
    struct WavePlannerConfig {
        uint32_t per_core_M_max = 0;
        bool partial_eighth = false;
        // URE_TAPER_TAIL of the program overlapped with combine (EmbeddedOptions::report_predicted_order), the
        // only one whose finish order is predicted.
        uint32_t taper_tail = 0;
    };
    // Overlapped with combine, the last this many experts of a walk run at most as pairs (adaptive_chunk::WavePlanner,
    // URE_TAPER_TAIL): a whole eighth or quad wave finishing last leaves combine all of its tokens to ship after
    // the routed expert is done, which on a hot chip is most of the op's tail.
    static constexpr uint32_t kOverlapTaperTail = 4;
    static WavePlannerConfig wave_planner_config(
        const UnifiedRoutedExpertFfnParams& operation_attributes, const UnifiedRoutedExpertFfnInputs& tensor_args);

    // TILE in-place calls store x and output as the same buffer twice in tensor_args, so
    // resolve_bindings bails and the adapter would rebuild this factory on every cache hit.
    // Common runtime args are the only per-dispatch addresses; per-core args stay hashed.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const UnifiedRoutedExpertFfnParams& operation_attributes,
        const UnifiedRoutedExpertFfnInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
