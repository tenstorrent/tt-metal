// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "unified_routed_expert_ffn_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

// Returns true when the unified routed-expert op should take the Wormhole
// execution path (8x8 compute grid, 2880-dim-tuned config) instead of the
// Blackhole path (11x8 grid, 7168/2048-dim config). True on real Wormhole_B0,
// or on ANY arch when the TT_UNIFIED_REXPERT_FORCE_WH passthrough env var is
// set — the latter is a temporary knob that lets us validate the WH kernel on
// Blackhole hardware (where the only difference is the smaller grid + config;
// the dataflow/compute kernels are shared). Both the host-side chunk picker
// and the program factory consult this so they agree on per_core_M / grid.
bool unified_routed_expert_use_wh_path(const ttnn::Tensor& x);

struct UnifiedRoutedExpertFfnSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

struct UnifiedRoutedExpertFfnProgramFactory {
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
