// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Permanent per-stage instrumentation for kernels.
//
//   MaybeDeviceZoneScope("stage_name");   // RAII: times the enclosing block
//
// A direct alias for DeviceZoneScopedN: compiles to nothing unless the device profiler is enabled
// (PROFILE_KERNEL), so it is free in production builds. Durability contract: zones placed at an op's stage
// boundaries are PERMANENT (perf tournaments rank stages from them) — do not remove them when the code
// around them changes; move them with the stage. Keep zones out of per-tile / per-row loops: each execution
// spends 2 of the 250 optional markers per RISC per dispatch, and running out is silent
// (.claude/references/device-zone-scope-attribution.md §7).
#pragma once

#include "api/debug/kernel_profiler.hpp"

// MAYBE_DEVICE_ZONE_SCOPE_OFF (a kernel define): compile the stage zones out of a profiled build, to measure
// what the zones themselves cost under the profiler (an A/B of the same kernel with and without them).
#if defined(MAYBE_DEVICE_ZONE_SCOPE_OFF) && !defined(MaybeDeviceZoneScope)
#define MaybeDeviceZoneScope(name)
#endif

#ifndef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#endif
