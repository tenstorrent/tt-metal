// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Per-stage device zones for kernel perf breakdowns.
//
//   { MaybeDeviceZoneScope("reader_read"); ...stage... }
//
// DURABILITY CONTRACT: zones placed with this macro are PERMANENT instrumentation of the op's stage boundaries.
// Never delete them when optimizing; extend them to every new path instead. They are free unless enabled:
//
//   - compiled to nothing unless the device profiler is on (PROFILE_KERNEL) AND the kernel is built with the
//     KERNEL_PERF_ZONES define (an op passes it to its kernels on request, e.g. from an env var on the host);
//   - so a profiler-on headline measurement (eval/ccl_perf.py) is not perturbed by them: no code growth (kernel-config
//     ring / trace-overlap budget), no marker traffic, no exhaustion of the per-RISC marker budget on busy RISCs.
//
// When enabled, it is DeviceZoneScopedN(name): a stopwatch on the enclosing block, whatever is inside (CB waits of
// helpers included) — see .claude/references/device-zone-scope-attribution.md before placing one. Budget: ~125 zone
// executions per RISC per dispatch; overflow is silent (zones after it simply vanish).

//
// Compute kernels: a zone compiles into all three TRISCs. Where that does not fit the kernel-config buffer, build with
// KERNEL_PERF_ZONES_TRISC=0 (unpack: the only thread whose CB waits block on data), 1 (math) or 2 (pack: blocks on
// space) to keep the zones on one thread only.

#if defined(KERNEL_PERF_ZONES_TRISC) &&                         \
    ((defined(TRISC_UNPACK) && KERNEL_PERF_ZONES_TRISC != 0) || \
     (defined(TRISC_MATH) && KERNEL_PERF_ZONES_TRISC != 1) || (defined(TRISC_PACK) && KERNEL_PERF_ZONES_TRISC != 2))
#define KERNEL_PERF_ZONES_OFF_THIS_THREAD
#endif

#if defined(PROFILE_KERNEL) && defined(KERNEL_PERF_ZONES) && !defined(KERNEL_PERF_ZONES_OFF_THIS_THREAD)
#include "tools/profiler/kernel_profiler.hpp"
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#else
#define MaybeDeviceZoneScope(name)
#endif
