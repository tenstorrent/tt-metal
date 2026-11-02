// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// MaybeDeviceZoneScope("<stage>") — a permanent per-stage device-profiler zone.
//
// Durability contract: instrumentation placed with this macro is PERMANENT. It is a direct alias of
// DeviceZoneScopedN and compiles to nothing when the device profiler is off (no PROFILE_KERNEL), so
// it costs nothing in production builds; do not remove it when optimizing a kernel.
//
// A zone times everything inside its braces, waits included (occupancy, not payload). Each
// execution spends 2 of the ~250 optional markers per RISC per dispatch; the overflow is silent.
// See .claude/references/device-zone-scope-attribution.md.
//
// Dataflow kernels get the profiler via dataflow_api.h; compute kernels include this header.

#include "api/debug/kernel_profiler.hpp"

#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
