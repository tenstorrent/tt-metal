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

#include "tools/profiler/kernel_profiler.hpp"

#ifndef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#endif
