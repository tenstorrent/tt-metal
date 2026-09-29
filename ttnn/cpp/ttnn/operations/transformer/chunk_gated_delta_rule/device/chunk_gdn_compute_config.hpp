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
// `name`. Returns nullopt when `name` is null or the var is unset/empty, in which case
// gdn_compute_config's caller gets today's fixed behaviour exactly (see below). NOTE: env vars are
// not part of the program hash, so a process must pick ONE fidelity setting for its whole lifetime
// -- a program compiled under a different setting could otherwise be served back from the cache.
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

// The one compute configuration for every chunk_gated_delta_rule program (mono, phased, fused). The
// kernels keep the recurrent state and the WY inverse in fp32 DEST at HiFi4 and nothing else has been
// validated, so other arithmetic is rejected; dst_full_sync_en passes through, other knobs are unused.
//
// `fidelity_env_var`, when non-null, names an env var (R10B experiment hook, plan_0928/P2_R10B) that
// may override math_fidelity for THIS kernel only. Unset/null (the default) reproduces today's
// behaviour exactly: math_fidelity must be HiFi4 or this TT_FATALs. fp32_dest_acc_en and
// math_approx_mode are never touched by the override -- they stay required exactly as before.
inline tt::tt_metal::ComputeConfigDescriptor gdn_compute_config(
    const DeviceComputeKernelConfig& cfg, const char* fidelity_env_var = nullptr) {
    using tt::tt_metal::MathFidelity;
    const std::optional<MathFidelity> fid_override = gdn_detail::gdn_fidelity_env_override(fidelity_env_var);
    TT_FATAL(
        (fid_override.has_value() || cfg.math_fidelity == MathFidelity::HiFi4) && cfg.fp32_dest_acc_en &&
            !cfg.math_approx_mode,
        "chunk_gated_delta_rule runs HiFi4 with fp32 destination accumulation and no approx mode on every "
        "path (its recurrent state and WY inverse are fp32) unless an experiment env var overrides fidelity; "
        "compute_kernel_config asked for math_fidelity {} (HiFi4 = {}), fp32_dest_acc_en {}, math_approx_mode {}",
        static_cast<int>(cfg.math_fidelity),
        static_cast<int>(MathFidelity::HiFi4),
        cfg.fp32_dest_acc_en,
        cfg.math_approx_mode);
    return tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = fid_override.value_or(cfg.math_fidelity),
        .fp32_dest_acc_en = cfg.fp32_dest_acc_en,
        .dst_full_sync_en = cfg.dst_full_sync_en,
        .math_approx_mode = cfg.math_approx_mode};
}

}  // namespace ttnn::prim
