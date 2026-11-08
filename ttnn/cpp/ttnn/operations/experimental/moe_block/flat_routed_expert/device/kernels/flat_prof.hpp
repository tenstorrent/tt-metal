// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Device-profiler instrumentation of the flat expert's kernels (compiled only with the device profiler, PROFILE_KERNEL;
// nothing otherwise): FP_ZONE a kernel-wide zone, FP_SUM a per-phase cycle sum (wall clock, cycles) reported once at
// the kernel's end as timestamped data (FP_EMIT), so the marker count stays fixed whatever the routing.
#pragma once
#include <cstdint>

#if defined(PROFILE_KERNEL) && !defined(SE_PROF_LIGHT)  // (SE_PROF_LIGHT: zones only, no cycle sums)
#include "api/debug/kernel_profiler.hpp"
FORCE_INLINE uint32_t fp_now() {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
}
#define FP_ZONE(name) DeviceZoneScopedN(name)
#define FP_VAR(v) uint32_t v = 0
#define FP_T0(t) const uint32_t t = fp_now()
#define FP_ADD(v, t) v += fp_now() - t
#define FP_EMIT(name, v) DeviceTimestampedData(name, v)
// a wait's cycles counted only when it spun (> 64 cycles: not the call's own cost), and how many did
#define FP_ADD_SPIN(v, n, t)              \
    {                                     \
        const uint32_t d_ = fp_now() - t; \
        if (d_ > 64) {                    \
            v += d_;                      \
            ++n;                          \
        }                                 \
    }
#elif defined(PROFILE_KERNEL)
#include "api/debug/kernel_profiler.hpp"
#define FP_ZONE(name) DeviceZoneScopedN(name)
#define FP_VAR(v)
#define FP_T0(t)
#define FP_ADD(v, t)
#define FP_EMIT(name, v)
#define FP_ADD_SPIN(v, n, t)
#else
#define FP_ZONE(name)
#define FP_VAR(v)
#define FP_T0(t)
#define FP_ADD(v, t)
#define FP_EMIT(name, v)
#define FP_ADD_SPIN(v, n, t)
#endif
