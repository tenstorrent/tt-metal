// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Stall counters for the unified routed-expert kernels (device profiler builds with URE_PROF only).
// URE_T(i, stmt) adds the wall-clock cycles stmt took on this RISC to counter i; URE_EMIT() writes the
// counters as timestamped data at the end of the kernel (zone names URE-00 .. URE-15, see perf/A9-report.md).
// Each TRISC thread compiles its own copy, so a compute-kernel counter exists once per thread.

#include <cstdint>

#if defined(URE_PROF) && defined(PROFILE_KERNEL)
#include "api/debug/kernel_profiler.hpp"
namespace ure_prof {
inline uint32_t acc[16] = {};
inline bool ev_on = false;
FORCE_INLINE uint32_t now() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
}  // namespace ure_prof
#define URE_T(i, ...)                                 \
    do {                                              \
        const uint32_t ure_t0 = ure_prof::now();      \
        __VA_ARGS__;                                  \
        ure_prof::acc[i] += ure_prof::now() - ure_t0; \
    } while (0)
#define URE_T0(i) const uint32_t ure_s##i = ure_prof::now()
#define URE_T1(i) ure_prof::acc[i] += ure_prof::now() - ure_s##i
// One timestamped event (kind, K-block) of the traced expert's first chunk (URE_EV_EXPERT, default none).
#ifndef URE_EV_EXPERT
#define URE_EV_EXPERT 0xffffffffu
#endif
#define URE_EV_SEL(expert, chunk) ure_prof::ev_on = ((expert) == URE_EV_EXPERT) && ((chunk) == 0)
#define URE_EV(kind, kb)                                           \
    do {                                                           \
        if (ure_prof::ev_on) {                                     \
            DeviceTimestampedData("URE-EV", ((kind) << 8) | (kb)); \
        }                                                          \
    } while (0)
#define URE_EMIT1(i, name) DeviceTimestampedData(name, ure_prof::acc[i])
#define URE_EMIT()               \
    do {                         \
        URE_EMIT1(0, "URE-00");  \
        URE_EMIT1(1, "URE-01");  \
        URE_EMIT1(2, "URE-02");  \
        URE_EMIT1(3, "URE-03");  \
        URE_EMIT1(4, "URE-04");  \
        URE_EMIT1(5, "URE-05");  \
        URE_EMIT1(6, "URE-06");  \
        URE_EMIT1(7, "URE-07");  \
        URE_EMIT1(8, "URE-08");  \
        URE_EMIT1(9, "URE-09");  \
        URE_EMIT1(10, "URE-10"); \
        URE_EMIT1(11, "URE-11"); \
        URE_EMIT1(12, "URE-12"); \
        URE_EMIT1(13, "URE-13"); \
        URE_EMIT1(14, "URE-14"); \
        URE_EMIT1(15, "URE-15"); \
    } while (0)
#else
#define URE_T(i, ...) \
    do {              \
        __VA_ARGS__;  \
    } while (0)
#define URE_T0(i) (void)0
#define URE_T1(i) (void)0
#define URE_EMIT() (void)0
#define URE_EV_SEL(expert, chunk) (void)0
#define URE_EV(kind, kb) (void)0
#endif
