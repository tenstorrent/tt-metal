// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Typed L1 read/write helpers for element-level gather operations.
// Optimized: no volatile (L1 stable during gather), no runtime switch.
#pragma once

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

// NoC reads the writers keep in flight before a barrier while loading the input row.
// Any depth up to the input CB's is correct: each writer clamps its burst against that CB's own
// tile count -- Wt_input for the row-buffered plans, chunk_tiles for the streaming one -- so a
// burst can neither overrun the CB nor straddle its end, since every block starts at the CB base.
// Unlike WRITE_BATCH no host-side CB sizing is derived from it, so it is shared here rather than
// threaded as a compile-time arg.
constexpr uint32_t kGatherReadBatchTiles = 4;

template <typename T>
FORCE_INLINE uint32_t read_data_from_type(const uint32_t l1_addr, const uint32_t count) {
    // l1_addr is a firmware L1 offset. The tt_l1_ptr attribute is the standard
    // marker for an L1-address-space pointer on silicon; it also lets the emule
    // JIT source-patcher recognise this deref of a passed-in offset (a bare
    // parameter, not an inline get_*_ptr call) and rebase it onto the emulated L1
    // bridge. Without it the cast escapes every patcher pattern and emule derefs
    // the raw offset (SIGSEGV at a small unmapped address).
    tt_l1_ptr T* ptr = reinterpret_cast<tt_l1_ptr T*>(l1_addr);
    return ptr[count];
}

FORCE_INLINE uint32_t
get_value_from_tile(const uint32_t l1_read_addr, const uint32_t count, const uint32_t data_format_size) {
    if (data_format_size == 2) {
        return read_data_from_type<uint16_t>(l1_read_addr, count);
    }
    if (data_format_size == 4) {
        return read_data_from_type<uint32_t>(l1_read_addr, count);
    }
    return read_data_from_type<uint8_t>(l1_read_addr, count);
}

template <typename T>
FORCE_INLINE void write_data_from_type(const uint32_t l1_addr, const uint32_t count, const uint32_t value) {
    // See read_data_from_type: tt_l1_ptr marks the L1 address space and lets the
    // emule patcher rebase the passed-in offset onto the emulated L1 bridge.
    tt_l1_ptr T* ptr = reinterpret_cast<tt_l1_ptr T*>(l1_addr);
    ptr[count] = value;
}

FORCE_INLINE void write_value_to_tile(
    const uint32_t l1_read_addr, const uint32_t count, const uint32_t data_format_size, const uint32_t value) {
    if (data_format_size == 2) {
        write_data_from_type<uint16_t>(l1_read_addr, count, value);
        return;
    }
    if (data_format_size == 4) {
        write_data_from_type<uint32_t>(l1_read_addr, count, value);
        return;
    }
    write_data_from_type<uint8_t>(l1_read_addr, count, value);
}
