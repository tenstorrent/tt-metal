// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// perf_instrumentation.hpp - permanent per-stage device zones for kernel_lib consumers.
//
//   MaybeDeviceZoneScope("stage_name");   // RAII: times the enclosing block
//
// Durability contract: these zones are PERMANENT instrumentation, not debug code. Leave them in when a
// kernel is optimized or refactored and extend them to every new path, so per-stage observability never
// regresses. The macro is free when the device profiler is off (DeviceZoneScopedN compiles to nothing
// without PROFILE_KERNEL), so there is no runtime cost in production builds.
//
// Reading the numbers: a zone times whatever is inside its braces, including a helper's internal
// cb_wait_front / cb_reserve_back - a zone around a helper call is OCCUPANCY (wait + work), not payload.
// Put waits in their own zones before ranking stages, and confirm a dominant stage with a payload ablation.
// Budget: ~125 zone executions per RISC-V processor per dispatch; exhaustion is SILENT (later zones vanish),
// so keep zones at block granularity, never per tile. See .claude/references/device-zone-scope-attribution.md.
//
// Compute kernels do not get the profiler through dataflow_api.h, hence this header; dataflow kernels may
// include it too (harmless).

// Copy of ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp from the llk_helper_library branch;
// main does not have it yet.
// Namespace gn_kernel_lib keeps it apart from a future copy on main.

#pragma once

#include "api/debug/kernel_profiler.hpp"

#ifndef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#endif

// MaybeDeviceZoneScopeIf(ENABLED, "stage_name"): the same zone, gated by a COMPILE-TIME bool. Use it for a zone
// inside a loop whose trip count the host knows: the host sets ENABLED (a CT arg) only when every execution fits
// the marker budget, so a long loop never silently exhausts it and hides the zones that follow. ENABLED = false
// is an empty struct (no markers, no code), as is the whole macro with the profiler off.
#if defined(PROFILE_STREAMING)
// Not supported by the streaming profiler: these template-gated zones use the DRAM profiler's hash ids.
#define MaybeDeviceZoneScopeIf(ENABLED, name)
#elif defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL)
namespace gn_perf_instrumentation {
template <bool Enabled, uint32_t timer_id>
struct GatedScope {
    inline __attribute__((always_inline)) GatedScope() {}
    inline __attribute__((always_inline)) ~GatedScope() {}
};
template <uint32_t timer_id>
struct GatedScope<true, timer_id> : kernel_profiler::profileScope<timer_id> {};
}  // namespace gn_perf_instrumentation
#define MaybeDeviceZoneScopeIf(ENABLED, name)                                  \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    gn_perf_instrumentation::GatedScope<(ENABLED), hash> zone;
#else
#define MaybeDeviceZoneScopeIf(ENABLED, name)
#endif
