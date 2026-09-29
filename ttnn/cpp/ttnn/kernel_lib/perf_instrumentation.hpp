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

// Opt-in stall accumulators (dataflow kernels only): sum the wall-clock cycles a kernel spends in
// one kind of wait across a whole loop, then emit ONE timestamped-data marker per accumulator, so
// a per-packet wait is measurable without spending a zone (2 markers) per iteration.
//
//     MaybePerfAccum(acc_slot_wait);          // declare, zeroed
//     for (...) {
//         MaybePerfBegin(acc_slot_wait);
//         ... the wait ...
//         MaybePerfEnd(acc_slot_wait);
//     }
//     MaybePerfReport("writer_slot_wait", acc_slot_wait);
//
// Compiled in only under the device profiler AND -DKERNEL_LIB_PERF_STALLS: each Begin/End pair
// is two wall-clock reads on the loop's critical path, measured at +2% device time on
// high_bw_all_reduce when it was on by default. Otherwise every macro is an unevaluated no-op.
#if defined(PROFILE_KERNEL) && defined(KERNEL_LIB_PERF_STALLS)
#define MaybePerfAccum(acc) \
    uint32_t acc = 0;       \
    uint32_t acc##_t0 = 0
#define MaybePerfBegin(acc) acc##_t0 = *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
#define MaybePerfEnd(acc) \
    acc += *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L) - acc##_t0
#define MaybePerfReport(name, acc) DeviceTimestampedData(name, static_cast<uint64_t>(acc))
#else
#define MaybePerfAccum(acc) ((void)0)
#define MaybePerfBegin(acc) ((void)0)
#define MaybePerfEnd(acc) ((void)0)
#define MaybePerfReport(name, acc) ((void)sizeof(name))
#endif
