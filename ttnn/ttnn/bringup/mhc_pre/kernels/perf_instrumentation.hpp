// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Per-stage device-zone instrumentation for kernel sources (dataflow and compute).
//
//     {
//         MaybeDeviceZoneScope("reader_issue");
//         ... the stage ...
//     }   // the zone stops at this brace
//
// MaybeDeviceZoneScope(name) is DeviceZoneScopedN(name) — an RAII stopwatch on the enclosing
// block — but ONLY when the kernel is compiled with BOTH the device profiler (PROFILE_KERNEL) and
// the opt-in define KERNEL_PERF_ZONES. Otherwise it expands to an unevaluated expression and costs
// nothing. The opt-in exists because every zone costs two profiler markers (a wall-clock read plus
// two L1 stores each) ON the kernel's critical path: measured on tilize (WH B0, Perf 1), ~12 zones
// on a ~2.3 us op added 330-510 ns (+14 %) to DEVICE KERNEL DURATION — the very number a perf gate
// reads. So a plain `--profile` run measures the production kernel, and a perf investigation opts
// in (the op's host code passes KERNEL_PERF_ZONES as a kernel define; for tilize set the env var
// TT_METAL_KERNEL_PERF_ZONES=1). A different define set is a different JIT hash, so toggling it
// always recompiles.
//
// DURABILITY CONTRACT: zones placed with this macro are PERMANENT observability, not debug
// scaffolding. Do not remove them when a stage is optimized; extend them to the new path so
// per-stage breakdowns stay comparable across revisions.
//
// Attribution caveats (.claude/references/device-zone-scope-attribution.md):
//   * a zone times everything inside its braces, including any cb_wait_front / cb_reserve_back
//     (directly or inside a helper), so a starved stage reads the same as an expensive one;
//   * a zone costs 2 of the 250 optional profiler markers per RISC-V processor per dispatch,
//     per EXECUTION: zones inside per-tile loops exhaust the budget and the remainder of the
//     kernel silently vanishes from the profile.

#include "tools/profiler/kernel_profiler.hpp"

#if defined(PROFILE_KERNEL) && defined(KERNEL_PERF_ZONES)
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#else
#define MaybeDeviceZoneScope(name) ((void)sizeof(name))
#endif
