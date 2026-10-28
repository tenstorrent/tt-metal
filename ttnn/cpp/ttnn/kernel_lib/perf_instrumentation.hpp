// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Permanent per-stage device-zone instrumentation for kernels.
//
//   MaybeDeviceZoneScope("stage_name");   // RAII: times until the enclosing block's closing brace
//
// Durability contract: these scopes are PERMANENT. They are free when the device profiler is off
// (DeviceZoneScopedN expands to a no-op without PROFILE_KERNEL), so a perf pass, a refinement or a
// verifier must never remove them. Extend them to every new stage / path instead.
//
// Reading the numbers: a zone times everything inside its braces, including a helper's internal
// cb_wait_front / cb_reserve_back, so a per-stage number is OCCUPANCY, not payload. Split wait from
// work where the question needs it, and confirm a dominant stage with a payload ablation.
// The profiler's per-RISC marker budget (250 markers = ~125 zone executions) is exhausted SILENTLY:
// never put a zone in a per-tile / per-chunk loop. See .claude/references/device-zone-scope-attribution.md.
//
// Dataflow kernels get the profiler transitively via dataflow_api.h; compute kernels need this include.

#include <api/debug/kernel_profiler.hpp>

#ifndef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#endif
