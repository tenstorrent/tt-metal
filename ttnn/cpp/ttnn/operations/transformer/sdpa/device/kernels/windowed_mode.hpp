// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/**
 * Shared host/device encoding of the windowed (block-diagonal, cu_window_seqlens) SDPA mode.
 *
 * Passed to the reader, writer and compute kernels as a compile-time arg. Windowed causal is
 * realized entirely by the windowed geometry and mask generator (the K range stops at each Q
 * chunk's diagonal; the mask is block-diagonal AND lower-triangular), so the kernels' own causal
 * paths (lightweight causal mask, zigzag balancing) stay off in every windowed mode.
 */
enum class WindowedMode : uint32_t {
    None = 0,           // Regular SDPA: no cu_window_seqlens.
    Bidirectional = 1,  // Token t in window [cu[i], cu[i+1]) attends to the whole window.
    Causal = 2,         // Token t in window [cu[i], cu[i+1]) attends to cu[i]..t.
};

// True for both windowed modes: cu_window_seqlens drives the K range and the generated mask.
constexpr bool is_windowed_mode(WindowedMode mode) { return mode != WindowedMode::None; }
