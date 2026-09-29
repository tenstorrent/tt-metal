// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <cstring>
#include <optional>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

namespace gdn_detail {
// R10B experiment hook (plan_0928/P2_R10B): parse HiFi4|HiFi3|HiFi2|LoFi from the env var named
// `name`. Returns nullopt when `name` is null or the var is unset/empty. The fused prim reads it
// through gdn_resolve_fidelity when its attributes are built, so the value is part of the program
// hash (a process may change it between calls).
inline std::optional<tt::tt_metal::MathFidelity> gdn_fidelity_env_override(const char* name) {
    if (name == nullptr) {
        return std::nullopt;
    }
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        return std::nullopt;
    }
    using tt::tt_metal::MathFidelity;
    if (std::strcmp(v, "HiFi4") == 0) {
        return MathFidelity::HiFi4;
    }
    if (std::strcmp(v, "HiFi3") == 0) {
        return MathFidelity::HiFi3;
    }
    if (std::strcmp(v, "HiFi2") == 0) {
        return MathFidelity::HiFi2;
    }
    if (std::strcmp(v, "LoFi") == 0) {
        return MathFidelity::LoFi;
    }
    TT_FATAL(false, "{}={}: expected one of HiFi4|HiFi3|HiFi2|LoFi", name, v);
    return std::nullopt;
}
}  // namespace gdn_detail

// The math fidelity of one fused compute kernel (prep or scan), resolved when the fused prim's
// attributes are built (the result is hashed). Precedence: the experiment env var `env_var` when set
// (QWEN36_FLA_PREP_FID / QWEN36_FLA_SCAN_FID) > `requested` (the ChunkGdnFusedProgramConfig field) >
// the compute_kernel_config's own fidelity, which must then be HiFi4 (today's fixed behaviour).
inline tt::tt_metal::MathFidelity gdn_resolve_fidelity(
    const DeviceComputeKernelConfig& cfg, const char* env_var, std::optional<tt::tt_metal::MathFidelity> requested) {
    using tt::tt_metal::MathFidelity;
    if (const auto env = gdn_detail::gdn_fidelity_env_override(env_var)) {
        return *env;
    }
    if (requested.has_value()) {
        TT_FATAL(
            *requested == MathFidelity::HiFi4 || *requested == MathFidelity::HiFi3 ||
                *requested == MathFidelity::HiFi2 || *requested == MathFidelity::LoFi,
            "chunk_gated_delta_rule: a program-config math fidelity must be HiFi4, HiFi3, HiFi2 or LoFi (got {})",
            static_cast<int>(*requested));
        return *requested;
    }
    TT_FATAL(
        cfg.math_fidelity == MathFidelity::HiFi4,
        "chunk_gated_delta_rule runs HiFi4 unless its program config (prep_math_fidelity / scan_math_fidelity) or an "
        "experiment env var sets another fidelity; compute_kernel_config asked for math_fidelity {} (HiFi4 = {})",
        static_cast<int>(cfg.math_fidelity),
        static_cast<int>(MathFidelity::HiFi4));
    return cfg.math_fidelity;
}

// The one compute configuration for every chunk_gated_delta_rule program (mono, phased, fused). The
// kernels keep the recurrent state and the WY inverse in fp32 DEST and nothing else has been
// validated, so other arithmetic is rejected; dst_full_sync_en passes through, other knobs are unused.
//
// `fidelity` unset (mono, phased): math_fidelity must be HiFi4 or this TT_FATALs (today's fixed
// behaviour). `fidelity` set (the fused prim, from gdn_resolve_fidelity): that fidelity for THIS
// kernel only. fp32_dest_acc_en and math_approx_mode are required exactly as before in both cases.
inline tt::tt_metal::ComputeConfigDescriptor gdn_compute_config(
    const DeviceComputeKernelConfig& cfg, std::optional<tt::tt_metal::MathFidelity> fidelity = std::nullopt) {
    using tt::tt_metal::MathFidelity;
    TT_FATAL(
        (fidelity.has_value() || cfg.math_fidelity == MathFidelity::HiFi4) && cfg.fp32_dest_acc_en &&
            !cfg.math_approx_mode,
        "chunk_gated_delta_rule runs HiFi4 with fp32 destination accumulation and no approx mode on every "
        "path (its recurrent state and WY inverse are fp32) unless the fused program config or an experiment env "
        "var sets the fidelity; compute_kernel_config asked for math_fidelity {} (HiFi4 = {}), fp32_dest_acc_en {}, "
        "math_approx_mode {}",
        static_cast<int>(cfg.math_fidelity),
        static_cast<int>(MathFidelity::HiFi4),
        cfg.fp32_dest_acc_en,
        cfg.math_approx_mode);
    return tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = fidelity.value_or(cfg.math_fidelity),
        .fp32_dest_acc_en = cfg.fp32_dest_acc_en,
        .dst_full_sync_en = cfg.dst_full_sync_en,
        .math_approx_mode = cfg.math_approx_mode};
}

}  // namespace ttnn::prim
