// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Permanent per-stage device-zone instrumentation for kernels.
//
//   {
//       MaybeDeviceZoneScope("reader_issue");
//       ... stage body ...
//   }
//
// DURABILITY CONTRACT: zones placed with MaybeDeviceZoneScope are PERMANENT observability, not debug
// scaffolding. Do not remove them in cleanups, refactors or verifier passes. The macro is free when the
// device profiler is off (DeviceZoneScopedN compiles to a no-op without PROFILE_KERNEL), so it costs
// nothing in production builds.
//
// Rules (see .claude/references/device-zone-scope-attribution.md):
//   - a zone times everything inside its braces, including CB waits: keep waits in their own zone;
//   - two zones cannot share one C++ scope (the macro declares locals): give each its own braces;
//   - 250 markers per RISC per dispatch (2 per zone execution); running out is silent — keep zones out of
//     per-tile loops.
//
// Compute kernels include this header; dataflow kernels may include it too (dataflow_api.h already pulls
// in the profiler).

#pragma once

#include "tools/profiler/kernel_profiler.hpp"

#ifndef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#endif
