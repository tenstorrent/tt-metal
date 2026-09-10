// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// writer_starvation bench — zone gate.
//
// The candidate kernels in this bench put a zone INSIDE a per-push-group loop
// (that is the whole point: the wait has to be observed per group, not per
// block). At `block_width_tiles = 16` with a per-2-tile push and several
// tile-rows per barrier that is well past the profiler's 250-marker-per-RISC
// budget, and budget exhaustion is SILENT — it would truncate the `*-KERNEL`
// span the wall-clock number itself is read from.
//
// So the zones are OFF by default (the wall-clock sweep runs unzoned and is
// therefore marker-safe on every shape) and turned on by a kernel define for
// the single focus-shape capture, where the count is small and bounded.
#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifdef TILIZE_BENCH_ZONES
#define BenchZone(name) MaybeDeviceZoneScope(name)
#else
#define BenchZone(name) ((void)0)
#endif
