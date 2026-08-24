// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

namespace ttnn::operations::matmul::registry {

Resolution resolve(const Mode mode, const Eligibility& eligibility) noexcept {
    if (mode == Mode::Off) {
        return {.reason = ResolutionReason::Disabled};
    }
    if (eligibility.call.domain == OperationDomain::IneligibleSharedCaller) {
        return {.reason = ResolutionReason::IneligibleOperationDomain};
    }
    const bool is_addmm = eligibility.call.domain == OperationDomain::Addmm;
    if (is_addmm != (eligibility.call.alpha_f32_bits.has_value() && eligibility.call.beta_f32_bits.has_value())) {
        return {.reason = ResolutionReason::MalformedOperationSemantics};
    }
    if (eligibility.has_program_config || eligibility.has_compute_kernel_config || eligibility.has_user_core_grid) {
        return {.reason = ResolutionReason::ExplicitOverride};
    }
    const bool unsupported_bias = eligibility.has_bias && eligibility.call.domain != OperationDomain::Linear;
    const bool unsupported_activation =
        eligibility.has_activation && eligibility.call.domain != OperationDomain::Linear;
    const bool unsupported_transpose =
        (eligibility.transpose_a || eligibility.transpose_b) && eligibility.call.domain != OperationDomain::Linear;
    if (unsupported_bias || unsupported_activation || eligibility.has_optional_output || eligibility.has_output_tile ||
        eligibility.has_global_cb || eligibility.has_sub_device || eligibility.input_a_sharded ||
        eligibility.input_b_sharded || eligibility.input_b_batched || unsupported_transpose) {
        return {.reason = ResolutionReason::UnsupportedSemantics};
    }

    // The generated certified table intentionally starts empty. Shadow and On
    // therefore both preserve the existing selector and cache identity.
    return {.reason = ResolutionReason::EmptyRegistry};
}

}  // namespace ttnn::operations::matmul::registry
