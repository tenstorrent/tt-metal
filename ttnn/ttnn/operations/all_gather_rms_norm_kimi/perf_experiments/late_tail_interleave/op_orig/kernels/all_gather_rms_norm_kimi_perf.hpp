// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Per-stage device zones (PERMANENT instrumentation, Perf 1). MaybeDeviceZoneScope is the kernel_lib name for a
// DeviceZoneScopedN zone (ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp where that header exists). Here the zones
// compile in only under AGRN_PERF_ZONES (the host sets it when AGRN_PERF_ZONES=1 at op build time): the ccl_perf
// harness runs 500 traced calls with the profiler on, and zone markers there overflow the profiler DRAM buffer and
// drop whole programs from its count. Zones sit outside per-chunk loops (the 125-zone-per-RISC budget), and a wait is
// zoned apart from the work it gates (.claude/references/device-zone-scope-attribution.md). Breakdown driver and
// analyzers: perf/p1_zone_driver.py.txt, perf/p1_zone_analyze.py.txt, perf/p1_tail_analyze.py.txt.
#pragma once

#if __has_include("ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp")
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#endif
#include "tools/profiler/kernel_profiler.hpp"

#ifdef AGRN_PERF_ZONES
#ifndef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#endif
#else
#undef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name)
#endif
