// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
//  MaybeDeviceZoneScope — PERMANENT per-stage device-profiler instrumentation
// ============================================================================
//
// One macro, usable from BOTH kernel families:
//
//     {
//         MaybeDeviceZoneScope("reader_read_x");
//         ... the stage's payload ...
//     }
//
// The zone's duration lands in
//   generated/profiler/.logs/profile_log_device.csv
// (parse with tools/tracy/process_device_log.py) whenever a run is profiled
// (`scripts/run_safe_pytest.sh --profile ...`).
//
// ---------------------------------------------------------------------------
//  DURABILITY CONTRACT — read this before deleting a zone
// ---------------------------------------------------------------------------
//  1. **It is FREE when the profiler is off.** Kernels are compiled without
//     `PROFILE_KERNEL` in every non-profiled run, and then this macro expands
//     to `(void(sizeof(name)))` — no code, no registers, no L1, no cycles.
//     There is therefore never a perf reason to remove a zone.
//  2. **It is PERMANENT.** These zones are the op's per-stage observability.
//     A later refinement that deletes them makes the next perf round guess
//     where the time goes instead of measuring it. Extend the set when you add
//     a stage; never strip it. If you replace a stage's implementation, move
//     its zone onto the new code — do not drop it.
//  3. **Name a STAGE, not a line.** The name is the unit a perf breakdown
//     ranks: `reader_read_x`, `compute_reduce`, `writer_gather`. Names are
//     hashed at compile time (16-bit), so keep them short and distinct.
//  4. **One zone per C++ scope.** The underlying macro declares a
//     `constexpr hash` in the enclosing scope, so two zones in the same braces
//     will not compile. Give each its own `{ ... }`.
//  5. **Budget: 125 zones per RISC per program launch**, silently dropped
//     beyond that. Zones inside a loop cost one entry per iteration — on a
//     shape with very many row-blocks the tail is truncated. That is fine for
//     ranking (the head is representative) but do not read an absolute total
//     off a truncated capture.
//  6. **Cannot co-run with DPRINT / Watcher** (shared L1 scratch). Profile with
//     `unset TT_METAL_DPRINT_CORES` and without `--dev`.
//
// ---------------------------------------------------------------------------
//  Ablation
// ---------------------------------------------------------------------------
// This header instruments; it does not ablate. To attribute overlapped time,
// stub a stage's PAYLOAD behind a compile-time flag while leaving its CB
// reserve/push/wait/pop and loop trip counts (and its zone) in place, and peel
// stages off CUMULATIVELY — see /perf-measure.

#pragma once

#include "tools/profiler/kernel_profiler.hpp"

// `DeviceZoneScopedN` is already a no-op (`(void(sizeof(name)))`) when
// `PROFILE_KERNEL` is undefined, which is every non-profiled build. The alias
// exists so that op kernels declare their intent ("a permanent stage marker")
// rather than reaching for the raw profiler macro, and so this one place can
// carry the contract above.
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
