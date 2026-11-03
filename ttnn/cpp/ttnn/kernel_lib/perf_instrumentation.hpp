// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Permanent per-stage perf instrumentation for kernels.
//
// MaybeDeviceZoneScope("<stage>") is a direct alias for DeviceZoneScopedN: an RAII device-profiler zone from the
// macro site to the closing brace of the enclosing block. It compiles to nothing when the device profiler is off
// (TT_METAL_DEVICE_PROFILER unset), so it is free in production builds.
//
// Durability contract: zones placed with this macro are PERMANENT instrumentation of an op's stage boundaries —
// do not remove them in later refactors; extend them to new paths instead. A zone times everything inside its
// braces, including a helper's internal CB waits (see .claude/references/device-zone-scope-attribution.md), and
// each execution spends 2 of the ~250 optional profiler markers per RISC per dispatch: keep zones out of hot
// per-tile loops, and gate per-block zones on a few iterations when the loop is long.
//
// Compute kernels include this header; dataflow kernels get the profiler transitively via dataflow_api.h (and may
// include it too).
#pragma once

#include "tools/profiler/kernel_profiler.hpp"

#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)

// MaybeDeviceZoneScopeIf(name, active): the same zone, recorded only when `active` (a runtime bool) is true. Use it
// for per-block zones inside long loops: sample a few blocks instead of exhausting the marker budget, which drops
// every later zone silently.
#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && !defined(ARCH_QUASAR)
namespace kernel_profiler {
template <uint32_t timer_id>
struct profileScopeIf {
    bool start_marked = false;
    inline __attribute__((always_inline)) explicit profileScopeIf(bool active) {
        if (active && bufferHasRoom<DoingDispatch::NOT_DISPATCH>()) {
            stackSize += PROFILER_L1_MARKER_UINT32_SIZE;
            start_marked = true;
            mark_time_at_index_inlined(wIndex, timer_id);
            wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }
    inline __attribute__((always_inline)) ~profileScopeIf() {
        if (start_marked) {
            mark_time_at_index_inlined(wIndex, get_const_id(timer_id, ZONE_END));
            wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
            stackSize -= PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }
};
}  // namespace kernel_profiler
#define MaybeDeviceZoneScopeIf(name, active)                                      \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                                  \
    auto constexpr hash_if = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScopeIf<hash_if> zone_if = kernel_profiler::profileScopeIf<hash_if>(active);
#else
#define MaybeDeviceZoneScopeIf(name, active) (void(sizeof(active)))
#endif
