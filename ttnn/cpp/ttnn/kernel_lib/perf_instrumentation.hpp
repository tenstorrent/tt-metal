// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Permanent per-stage device-zone instrumentation for kernels.
//
//   MaybeDeviceZoneScope("stage_name");   // RAII: times the rest of the enclosing block
//
// Durability contract: these zones are PERMANENT. The macro is a direct alias of
// DeviceZoneScopedN and compiles to nothing when the device profiler is off, so leaving it in a
// kernel costs nothing at runtime. Never strip it from a kernel after a perf pass: the next pass
// re-uses the same stage boundaries to re-measure the critical path.
//
// What a zone measures (read .claude/references/device-zone-scope-attribution.md before placing
// one): the wall-clock between the macro site and the closing brace, INCLUDING any CB wait /
// reserve or NoC barrier inside (helpers wait internally). It is occupancy, not payload cost.
// Budget: ~125 zone executions per RISC per dispatch; past that, zones silently vanish — keep
// zones outside per-tile loops.
//
// Dataflow kernels get the profiler transitively through dataflow_api.h; compute kernels include
// this header.

#pragma once

#include "api/debug/kernel_profiler.hpp"

#ifndef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#endif
