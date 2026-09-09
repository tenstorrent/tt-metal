// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "unified_routed_expert_ffn_program_factory.hpp"
#include "unified_routed_expert_ffn_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

// Grouped variant of UnifiedRoutedExpertFfnProgramFactory (selected when
// UnifiedRoutedExpertFfnParams::num_row_groups > 0).
//
// The GRID_X x GRID_Y core grid is split into num_row_groups row groups of R =
// GRID_Y / num_row_groups M-rows. Every group runs its own expert loop over the
// experts a device-side greedy balance (group_assign::build_plan, computed from the
// resident per-expert token counts) hands it, so up to num_row_groups experts run
// concurrently. Within a group the data movement is the legacy scheme (x multicast
// along the row, weights read by the group's first row and multicast down the
// column group, activated multicast along the row); with R == 1 there is no weight
// multicast at all and every core streams its own weight slice from DRAM — gate on
// NoC 0 (reader) and up + down on NoC 1 (writer, UP_SPLIT + DOWN_SPLIT) — which is
// what lets the whole grid approach the DRAM bandwidth ceiling.
struct UnifiedRoutedExpertFfnGroupedProgramFactory {
    using shared_variables_t = UnifiedRoutedExpertFfnSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const UnifiedRoutedExpertFfnParams& operation_attributes,
        const UnifiedRoutedExpertFfnInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const UnifiedRoutedExpertFfnParams& operation_attributes,
        const UnifiedRoutedExpertFfnInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
