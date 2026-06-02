// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace agmm_profiling_config {

inline bool matmul_isolation_enabled() {
    const char* v = std::getenv("AGMM_MATMUL_ISOLATION");
    return v && std::string(v) == "1";
}

// Append new profiling stages here as {name, "1"} pairs.
inline std::vector<std::pair<std::string, std::string>> get_profiling_defines() {
    std::vector<std::pair<std::string, std::string>> defines;
    if (matmul_isolation_enabled()) {
        defines.emplace_back("MATMUL_ISOLATION_MODE", "1");
    }
    return defines;
}

}  // namespace agmm_profiling_config
