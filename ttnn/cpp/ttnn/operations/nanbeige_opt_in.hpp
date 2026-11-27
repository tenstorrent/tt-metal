// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <string_view>

// Opt-in switches for the agentic-models nanbeige_4_3_33b package (its tt_metal patches in
// agentic-models patches.json). The package's serving process sets them; when unset, every
// program that reads them is built exactly as before.
namespace ttnn::nanbeige {

inline bool env_flag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::string_view(value) == "1";
}

// TTNN_NANBEIGE_SAMPLING_TOP64=1: ttnn.sampling compiles 64 candidates (top_k up to 64).
inline bool sampling_top64() { return env_flag("TTNN_NANBEIGE_SAMPLING_TOP64"); }

// TTNN_NANBEIGE_PRECISION=1: Blackhole FP32-preserving paths in RoPE, RMSNorm, line reduce-scatter,
// chunked SDPA prefill, paged SDPA decode and the BF16 -> BFP8 paged KV-cache update.
inline bool precision() { return env_flag("TTNN_NANBEIGE_PRECISION"); }

}  // namespace ttnn::nanbeige
