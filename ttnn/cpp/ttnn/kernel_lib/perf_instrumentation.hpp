// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// perf_instrumentation.hpp — permanent per-stage device zones for kernel_lib consumers.
//
//   MaybeDeviceZoneScope("stage_name");   // RAII: times the enclosing block
//
// Durability contract: these zones are PERMANENT instrumentation, not debug code. Leave them in when a
// kernel is optimized or refactored and extend them to every new path, so per-stage observability never
// regresses. The macro is free when the device profiler is off (DeviceZoneScopedN compiles to nothing
// without PROFILE_KERNEL), so there is no runtime cost in production builds.
//
// Reading the numbers: a zone times whatever is inside its braces, including a helper's internal
// cb_wait_front / cb_reserve_back — a zone around a helper call is OCCUPANCY (wait + work), not payload.
// Put waits in their own zones before ranking stages, and confirm a dominant stage with a payload ablation.
// Budget: ~125 zone executions per RISC-V processor per dispatch; exhaustion is SILENT (later zones vanish),
// so keep zones at block granularity, never per tile. See .claude/references/device-zone-scope-attribution.md.
//
// Compute kernels do not get the profiler through dataflow_api.h, hence this header; dataflow kernels may
// include it too (harmless).

#pragma once

#include "api/debug/kernel_profiler.hpp"

#ifndef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#endif
