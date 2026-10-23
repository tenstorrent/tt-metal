// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Permanent per-stage device-zone instrumentation for kernels.
//
//   MaybeDeviceZoneScope("stage_name");   // RAII: times the enclosing block
//
// Durability contract: zones placed with this macro are PERMANENT. They are free when the device profiler is off
// (DeviceZoneScopedN compiles to nothing unless PROFILE_KERNEL is defined), so they must never be removed to "clean
// up" a kernel; perf passes extend them to every new path instead. A zone measures everything inside its braces,
// including CB waits inside helpers (occupancy, not payload) — see .claude/references/device-zone-scope-attribution.md.
// Budget: ~125 zone executions per RISC per dispatch; recording past it stops silently.
//
// Compute kernels include this header; dataflow kernels get the profiler via dataflow_api.h but may include it too.

#pragma once

#include "tools/profiler/kernel_profiler.hpp"

#ifndef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#endif
