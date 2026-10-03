// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>

#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::operations::transformer::sdpa {

// SDPAProgramConfig::qk_math_fidelity / pv_math_fidelity on an op whose streaming kernel honours them: at most one
// phase may be overridden (the other keeps the compute kernel config's fidelity), and only on the streaming path.
inline void validate_phase_fidelity(
    const std::optional<SDPAProgramConfig>& program_config, bool use_streaming_compute, const char* op_name) {
    if (!program_config.has_value()) {
        return;
    }
    const bool qk_set = program_config->qk_math_fidelity.has_value();
    const bool pv_set = program_config->pv_math_fidelity.has_value();
    TT_FATAL(
        !(qk_set && pv_set),
        "{}: set at most one of qk_math_fidelity and pv_math_fidelity; the other phase keeps the compute kernel "
        "config's fidelity (use the compute kernel config to change both)",
        op_name);
    TT_FATAL(
        !(qk_set || pv_set) || use_streaming_compute,
        "{}: qk_math_fidelity / pv_math_fidelity need the streaming compute path (fp32_dest_acc_en=false)",
        op_name);
}

// For ops whose kernels ignore the per-phase fields.
inline void reject_phase_fidelity(const std::optional<SDPAProgramConfig>& program_config, const char* op_name) {
    TT_FATAL(
        !program_config.has_value() ||
            (!program_config->qk_math_fidelity.has_value() && !program_config->pv_math_fidelity.has_value()),
        "{} does not support qk_math_fidelity / pv_math_fidelity; set the fidelity through the compute kernel config",
        op_name);
}

// The streaming compute kernel's compile-time arg (compute_streaming.hpp): per phase MathFidelity + 1 when the phase's
// effective fidelity differs from the compute kernel config's, else 0 (the ordinary helpers); bit 16 when that config
// is LoFi.
inline uint32_t matmul_fidelity_ct_arg(
    std::optional<tt::tt_metal::MathFidelity> qk_math_fidelity,
    std::optional<tt::tt_metal::MathFidelity> pv_math_fidelity,
    tt::tt_metal::MathFidelity compute_fidelity) {
    const auto code = [compute_fidelity](std::optional<tt::tt_metal::MathFidelity> fidelity) -> uint32_t {
        if (!fidelity.has_value() || *fidelity == compute_fidelity) {
            return 0u;
        }
        TT_FATAL(*fidelity != tt::tt_metal::MathFidelity::Invalid, "MathFidelity::Invalid is not a matmul fidelity");
        return static_cast<uint32_t>(*fidelity) + 1u;
    };
    const uint32_t compute_lofi = compute_fidelity == tt::tt_metal::MathFidelity::LoFi ? 1u : 0u;
    return code(qk_math_fidelity) | (code(pv_math_fidelity) << 8) | (compute_lofi << 16);
}

}  // namespace ttnn::operations::transformer::sdpa
