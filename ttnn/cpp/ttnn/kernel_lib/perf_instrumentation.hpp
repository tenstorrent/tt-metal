// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Permanent per-stage perf instrumentation for kernels.
//
// MaybeDeviceZoneScope("<stage>") is a direct alias for DeviceZoneScopedN: a RAII zone that times
// its enclosing block when the device profiler is on and compiles to nothing otherwise.
//
// Durability contract: zones placed with this macro are PERMANENT instrumentation. They are free
// when the profiler is off, so never remove them to "clean up" a kernel — later perf passes rank
// stages off them. A zone times everything inside its braces, including CB waits a helper does
// internally (occupancy, not payload), and each execution spends 2 of the 250 per-RISC markers —
// keep zones out of per-tile loops. See .claude/references/device-zone-scope-attribution.md.
//
// Dataflow kernels already get the profiler through dataflow_api.h; compute kernels include this.

#pragma once

#include "tools/profiler/kernel_profiler.hpp"

#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
