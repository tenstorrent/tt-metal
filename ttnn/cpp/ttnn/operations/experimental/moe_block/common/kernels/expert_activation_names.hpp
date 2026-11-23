// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The named compile-time args of the experts' GLU activation (common/kernels/expert_activation.hpp on the device,
// common/moe_block_config.hpp on the host): one name for each meaning, shared by both sides. Plain constexpr only (no
// includes): the kernels include it as is.
#pragma once

namespace moe_block_expert_activation_names {

inline constexpr const char* kKind = "expert_activation";  // the Kind (host: expert_activation_code)
// the constants, each the bit pattern of an fp32 (host: expert_activation_constant_args)
inline constexpr const char* kBetaBits = "expert_activation_beta_bits";
inline constexpr const char* kLinearBetaBits = "expert_activation_linear_beta_bits";
inline constexpr const char* kLimitBits = "expert_activation_limit_bits";
inline constexpr const char* kAlphaBits = "expert_activation_alpha_bits";

}  // namespace moe_block_expert_activation_names
