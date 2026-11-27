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

// Which cores load cu_window_seqlens, and how much of it: the reader and the writer of every core of the grid, a
// tile's worth of bytes each from the tensor's one page. With the define SDPA_WINDOWS_READ_BY_WORKING_CORES, which
// the host sets for the LLK 2.0 compute kernel's route, only a core with a Q chunk reads the tensor, and only its
// values: the reads of the cores without work do not queue at the tensor's DRAM bank in front of the others.
#ifdef SDPA_WINDOWS_READ_BY_WORKING_CORES
constexpr bool windows_read_by_working_cores = true;
#else
constexpr bool windows_read_by_working_cores = false;
#endif

// True when a core with that many Q chunks loads cu_window_seqlens (and the per-device Q origin with it).
constexpr bool loads_window_lengths(uint32_t q_chunks_of_core) {
    return !windows_read_by_working_cores || q_chunks_of_core != 0;
}

// Bytes a core reads of cu_window_seqlens, which has `eles` 32-bit values; `tile_bytes` is its landing buffer's size.
constexpr uint32_t window_lengths_read_bytes(uint32_t tile_bytes, uint32_t eles) {
    return windows_read_by_working_cores ? eles * static_cast<uint32_t>(sizeof(uint32_t)) : tile_bytes;
}
