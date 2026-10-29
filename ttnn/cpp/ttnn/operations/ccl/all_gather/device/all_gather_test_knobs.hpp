// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// TEST ONLY, DO NOT UPSTREAM: environment overrides of all_gather's tuning knobs, for sweeps. Unset, every knob keeps
// its built-in value. None of them is in the program-cache hash: keep the environment constant within a process.
//   TT_AG_FORCE=unicast|multicast   TT_AG_K   TT_AG_READ_BYTES   TT_AG_WORKERS   TT_AG_MUX_BUFS (1-255)
//   TT_AG_SIG_PER_ITER   TT_AG_SIG_MIN   TT_AG_SIG_MAX   TT_AG_NO_ROW_GROUP=1   TT_AG_LOG=1 (one AGCFG line per build)
//   TT_AG_UC_RD_NOC   TT_AG_UC_WR_NOC   TT_AG_MC_RD_NOC   TT_AG_MC_WR_NOC   TT_AG_MUX_NOC (0 or 1)

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string_view>

#include <tt-metalium/kernel_types.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::ccl::ag_test {

inline std::optional<std::string_view> env_str(const char* name) {
    const char* v = std::getenv(name);
    return (v == nullptr || *v == '\0') ? std::nullopt : std::optional<std::string_view>(v);
}

inline std::optional<uint32_t> env_u32(const char* name) {
    const auto v = env_str(name);
    if (!v) {
        return std::nullopt;
    }
    char* end = nullptr;
    const unsigned long x = std::strtoul(v->data(), &end, 0);
    TT_FATAL(*end == '\0' && x <= UINT32_MAX, "{}={} is not a uint32", name, *v);
    return static_cast<uint32_t>(x);
}

inline tt::tt_metal::NOC env_noc(const char* name, tt::tt_metal::NOC fallback) {
    const auto v = env_u32(name);
    TT_FATAL(!v || *v <= 1, "{} must be 0 or 1", name);
    return v ? static_cast<tt::tt_metal::NOC>(*v) : fallback;
}

inline bool log_enabled() { return env_u32("TT_AG_LOG").value_or(0) != 0; }

}  // namespace ttnn::operations::ccl::ag_test
