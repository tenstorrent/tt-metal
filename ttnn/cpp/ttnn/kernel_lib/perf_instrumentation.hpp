// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PERMANENT per-stage device instrumentation for TTNN kernels.
//
// ---------------------------------------------------------------------------
// THE DURABILITY CONTRACT
// ---------------------------------------------------------------------------
// `MaybeDeviceZoneScope("<stage>")` brackets ONE stage of a kernel's per-iteration chain and
// reports that stage's device time to the Tracy device profiler. It is designed to be
// **checked in and left in forever**:
//
//   * When the profiler is OFF (the shipped configuration — `PROFILE_KERNEL` undefined) the macro
//     expands to `(void(sizeof(name)))`, i.e. it emits **NO instructions, no L1 traffic and no
//     register pressure**. The shipped kernel binary is byte-identical to one with no zones at all.
//   * When the profiler is ON it emits the same 2 timestamp writes `DeviceZoneScopedN` does.
//
// Therefore: **NEVER DELETE A ZONE TO "CLEAN UP" A KERNEL.** Removing one costs nothing at runtime
// and destroys the op's per-stage observability, which is the only thing that makes a later perf
// round targetable instead of guesswork. When you add a new fast path, ADD its zone.
//
// ---------------------------------------------------------------------------
// WHY THIS HEADER EXISTS (rather than including the profiler directly)
// ---------------------------------------------------------------------------
// Dataflow kernels get `tools/profiler/kernel_profiler.hpp` transitively through
// `dataflow_api.h`, but a COMPUTE kernel must not see the dataflow API — so a compute TU that
// wants a zone has to include the profiler header itself, and the correct path is not obvious.
// This header is the one place that knows it, so all three kernels of an op can spell the same
// thing.
//
// ---------------------------------------------------------------------------
// USE
// ---------------------------------------------------------------------------
//   #include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
//   ...
//   {
//       MaybeDeviceZoneScope("reader_read");     // one stage of the chain
//       for (...) { noc_async_read(...); }
//       noc_async_read_barrier();
//   }
//
// Name zones after the STAGE, not the code (`compute_reduce`, not `the_add_loop`), and keep the
// names stable across refinements so two rounds' measurements are comparable.
//
// ---------------------------------------------------------------------------
// LIMITS worth knowing before you read a profile
// ---------------------------------------------------------------------------
//   * **125 zone RECORDS per core per dispatch**, silently dropped beyond. A zone inside a loop
//     costs one record PER ITERATION, so `zones_per_iteration * trip_count <= 125` must hold for
//     the shape you profile. When it does not, profile the small-trip-count shape (or the drop
//     shows up as a truncated tail, never as an error).
//   * Zones share SRAM with DPRINT and the Watcher — they cannot be enabled together
//     (`unset TT_METAL_DPRINT_CORES`, no `--dev`).
//   * Zone results land in `generated/profiler/.logs/profile_log_device.csv`
//     (parse with `tools/tracy/process_device_log.py`); the per-op
//     `ops_perf_results_*.csv` carries only the whole-kernel duration.

#pragma once

#include "tools/profiler/kernel_profiler.hpp"

// One stage of a kernel's chain. Free when the profiler is off — see the durability contract above.
//
// NOTE on scoping: the underlying `DeviceZoneScopedN` declares two names (`hash`, `zone`) in the
// ENCLOSING scope, so exactly one zone may live per `{ }` block. Give each stage its own braces.
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
