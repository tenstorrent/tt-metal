// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

namespace ttnn::operations::matmul::registry {

Resolution resolve(const Mode mode, const Eligibility& eligibility) noexcept {
    if (mode == Mode::Off) {
        return {.reason = ResolutionReason::Disabled};
    }
    if (eligibility.call_origin != CallOrigin::PublicMatmul) {
        return {.reason = ResolutionReason::IneligibleCallOrigin};
    }
    if (eligibility.has_program_config || eligibility.has_compute_kernel_config || eligibility.has_user_core_grid) {
        return {.reason = ResolutionReason::ExplicitOverride};
    }
    if (eligibility.has_bias || eligibility.has_activation || eligibility.has_optional_output ||
        eligibility.has_output_tile || eligibility.has_global_cb || eligibility.has_sub_device ||
        eligibility.input_a_sharded || eligibility.input_b_sharded || eligibility.input_b_batched ||
        eligibility.transpose_a || eligibility.transpose_b) {
        return {.reason = ResolutionReason::UnsupportedSemantics};
    }

    // The generated certified table intentionally starts empty. Shadow and On
    // therefore both preserve the existing selector and cache identity.
    return {.reason = ResolutionReason::EmptyRegistry};
}

}  // namespace ttnn::operations::matmul::registry
