// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast stage zones (PERMANENT instrumentation — never remove a G6_ZONE).
//
// `G6_ZONE("<stage>")` is `DeviceZoneScopedN` when the program is built with the define
// G6_TYPECAST_ZONES (host: environment variable G6_TYPECAST_ZONES=1, see the program descriptor), and
// nothing otherwise. Why gated and not always on: this tree has no perf_instrumentation.hpp, and a
// DeviceZoneScopedN costs ~51 cycles per execution whenever the device profiler is on — which is exactly
// when DEVICE KERNEL DURATION is measured. Perf 1 measured the per-block zones of this op on the
// critical path of the 1-block-per-core cells (see changelog, Perf 1). Breakdown runs turn them on;
// timed runs (A/B vs stock) leave them off.
#pragma once

#if defined(G6_TYPECAST_ZONES)
#include "api/debug/kernel_profiler.hpp"
#define G6_ZONE(name) DeviceZoneScopedN(name)
#else
#define G6_ZONE(name)
#endif
