// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Rebuilds a merged dataflow kernel's full per-core runtime-argument list from the compact one the
// host sends (compact_runtime_args in hybrid_half_merge.cpp).
//
// Most of the routed expert's per-core arguments are the same on every core: semaphore ids,
// multicast rectangles, sender tables. Sent per core they are most of the program's dispatch
// commands, and the routed expert overlapped with combine carries a copy of the program per chip,
// so untraced their bytes are host time. Most of the rest differ only from row to row of cores, or
// only from column to column. The host keeps per core only what differs otherwise, puts each shared
// value once in the common arguments and each row's or column's value in a common table, and
// describes the original list there too: HYB_RTX_COUNT words at common argument HYB_RTX_CRT, one
// per original index, encoded as hybrid_rt_table.hpp lays out.
//
// Called first thing in kernel_main, it writes the original list to this RISC's local memory and
// points rta_l1_base at it, so every get_arg_val after it -- either half's, the shims', the
// helpers' -- reads the original indices unchanged.

#pragma once

#ifdef HYB_RTX_COUNT

#include "api/dataflow/dataflow_api.h"
#include "hybrid_rt_table.hpp"

inline void hybrid_expand_runtime_args() {
    static uint32_t full_args[HYB_RTX_COUNT];
    const uint32_t row = ::get_absolute_logical_y();
    const uint32_t column = ::get_absolute_logical_x();
    for (uint32_t i = 0; i < HYB_RTX_COUNT; ++i) {
        const uint32_t entry = ::get_common_arg_val<uint32_t>(HYB_RTX_CRT + i);
        const uint32_t slot = entry & kHybridRtSlotMask;
        uint32_t value;
        switch (entry >> kHybridRtKindShift) {
            case kHybridRtShared: value = ::get_common_arg_val<uint32_t>(slot); break;
            case kHybridRtPerRow: value = ::get_common_arg_val<uint32_t>(slot + row); break;
            case kHybridRtPerColumn: value = ::get_common_arg_val<uint32_t>(slot + column); break;
            case kHybridRtLinearInRow:
                value = ::get_common_arg_val<uint32_t>(slot) + ::get_common_arg_val<uint32_t>(slot + 1) * row;
                break;
            case kHybridRtLinearInColumn:
                value = ::get_common_arg_val<uint32_t>(slot) + ::get_common_arg_val<uint32_t>(slot + 1) * column;
                break;
            default: value = ::get_arg_val<uint32_t>(slot); break;
        }
        full_args[i] = value;
    }
    rta_l1_base = full_args;
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT)
    rta_count = HYB_RTX_COUNT;
#endif
}

#else

inline void hybrid_expand_runtime_args() {}

#endif
