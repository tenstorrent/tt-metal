// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// perf_instrumentation — permanent per-stage device zones for op kernels.
//
//   MaybeDeviceZoneScope("stage_name");
//
// Opens an RAII device-profiler zone for the rest of the enclosing block. With the device profiler
// on (PROFILE_KERNEL, i.e. a `--profile` run) it is exactly `DeviceZoneScopedN` and the zone lands in
// generated/profiler/.logs/profile_log_device.csv as ZONE_START / ZONE_END rows per core and RISC.
// With the profiler off the macro expands to nothing — no code, no data, no L1 — so the
// instrumentation is free in production and is meant to stay in the kernel permanently.
//
// Durability contract: zones placed with this macro are the op's per-stage observability. They are
// never removed by a later refinement; a new code path extends them.
//
// Placement rules (see .claude/references/device-zone-scope-attribution.md):
//   * a zone times everything inside its braces, waits included — split `cb_wait_front` /
//     `cb_reserve_back` / NoC barriers from the payload so occupancy is not mistaken for cost;
//   * a compute-kernel zone is recorded on all three TRISCs; only unpack can block on tiles arriving
//     and only pack on space, math stalls land wherever its stream happens to be;
//   * the budget is ~125 zone executions per RISC per dispatch and exhaustion is silent — keep zones
//     out of per-tile loops.
//
// Dataflow kernels get the profiler transitively through dataflow_api.h; compute kernels include
// this header.

#pragma once

#include "tools/profiler/kernel_profiler.hpp"

#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL)
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#else
#define MaybeDeviceZoneScope(name)
#endif
