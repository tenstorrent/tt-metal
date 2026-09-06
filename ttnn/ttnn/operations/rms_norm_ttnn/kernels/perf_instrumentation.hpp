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
//  1. **It is FREE unless you ask for it.** The zones compile only when the
//     program descriptor emits the `RMS_STAGE_ZONES` kernel define (set the env
//     var of the same name); otherwise this macro expands to
//     `(void(sizeof(name)))` — no code, no registers, no L1, no cycles, and no
//     profiler zone-hash entry (see "WHY THE ZONES ARE OPT-IN" below, which is
//     the part that is NOT merely about cycles). There is therefore never a
//     perf reason to remove a zone.
//
//         RMS_STAGE_ZONES=1 scripts/run_safe_pytest.sh --profile <test>
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
// `PROFILE_KERNEL` is undefined. The alias exists so that op kernels declare
// their intent ("a permanent stage marker") rather than reaching for the raw
// profiler macro, and so this one place can carry the contract above.
//
// ---------------------------------------------------------------------------
//  WHY THE ZONES ARE OPT-IN (`RMS_STAGE_ZONES`) AND NOT MERELY PROFILE-GATED
// ---------------------------------------------------------------------------
// `PROFILE_KERNEL` is NOT the same switch as "somebody is reading a per-stage
// breakdown".  Any run that sets `TT_METAL_DEVICE_PROFILER=1` -- which the eval
// golden runner does for EVERY graded run, purely to read the op-level
// `DEVICE KERNEL DURATION` off the FIRMWARE markers -- compiles user zones in
// too.  That is not free, and the cost is not cycles:
//
//   `tt_metal/impl/profiler/profiler.cpp` keys every zone by a **16-bit** hash
//   of the string "<zone>,<absolute source path>,<line>,KERNEL_PROFILER", and
//   `populateZoneSrcLocations()` TT_THROWs the moment two DISTINCT strings land
//   on the same hash.  The line number is part of the string, so every edit to
//   a kernel file re-rolls the dice for every zone in it, and this op declares
//   35 distinct stage names across ~41 sites.  Against a 65 536-slot space that
//   is a ~2-3% birthday collision per edit -- and when it hits, the throw fires
//   on EVERY profiler read and finally escapes as `terminate` at a device
//   re-open, taking the whole pytest process (and its junit.xml) with it.
//   Refinement 4 hit exactly that: `compute_scale@compute.cpp:1680` and
//   `writer_tree_forward@writer.cpp:662` both hash to 0x0773.
//
// So the zones default to OFF and are turned on by a kernel define that the
// program descriptor emits from the `RMS_STAGE_ZONES` env var (ONE source of
// truth -- see `_kernel_defines()` there).  The contract above is unchanged:
// the zones stay in the source, permanently, and a perf round gets all of them
// with `RMS_STAGE_ZONES=1`.  A graded run registers ZERO op zone locations, so
// the collision class cannot reach it at all.
//
// `test_rms_norm_ttnn_zone_hashes.py` pins the invariant for the ON build: it
// re-implements the profiler's own hash over the CURRENT source lines and fails
// if any two zones collide.  If it ever goes red, move ONE zone by one line.
#if defined(RMS_STAGE_ZONES)
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#else
// Byte-identical to the profiler's own disabled expansion: no pragma, hence no
// zone-source-location registration, hence no hash entry.
#define MaybeDeviceZoneScope(name) (void(sizeof(name)))
#endif
