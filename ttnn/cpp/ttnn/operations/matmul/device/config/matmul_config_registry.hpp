// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"

namespace ttnn::operations::matmul::registry {

enum class Mode { Off, Shadow, On };

// The shared bound_matmul helper also serves linear and addmm. Call origin is
// explicit so adding a registry lookup cannot silently widen its semantic domain.
enum class CallOrigin { PublicMatmul, IneligibleSharedCaller };

enum class ResolutionReason {
    Disabled,
    IneligibleCallOrigin,
    ExplicitOverride,
    UnsupportedSemantics,
    EmptyRegistry,
    CertifiedMatch,
};

struct Eligibility {
    CallOrigin call_origin = CallOrigin::IneligibleSharedCaller;
    bool has_program_config = false;
    bool has_compute_kernel_config = false;
    bool has_user_core_grid = false;
    bool has_bias = false;
    bool has_activation = false;
    bool has_optional_output = false;
    bool has_output_tile = false;
    bool has_global_cb = false;
    bool has_sub_device = false;
    bool input_a_sharded = false;
    bool input_b_sharded = false;
    bool input_b_batched = false;
    bool transpose_a = false;
    bool transpose_b = false;
};

struct Recipe {
    MatmulProgramConfig program_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct Resolution {
    ResolutionReason reason = ResolutionReason::Disabled;
    std::optional<Recipe> recipe = std::nullopt;
};

// Startup-frozen rollout control is added with the generated table. The first
// plumbing change is deliberately and unconditionally Off in production.
inline constexpr Mode current_mode() noexcept { return Mode::Off; }

// Allocation-free/device-free admission and empty-table lookup. A future table
// implementation may return a Recipe only after constructing and exactly matching
// the complete typed request key.
Resolution resolve(Mode mode, const Eligibility& eligibility) noexcept;

}  // namespace ttnn::operations::matmul::registry
