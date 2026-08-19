// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
// PERMANENT per-stage device instrumentation for TTNN kernels.
// ============================================================================
//
// `MaybeDeviceZoneScope("<stage>")` is an RAII stopwatch on the enclosing block.
// It is a thin, self-documenting alias for Metalium's `DeviceZoneScopedN`, and
// its whole reason to exist is the DURABILITY CONTRACT below.
//
// ---------------------------------------------------------------------------
// DURABILITY CONTRACT — never delete a MaybeDeviceZoneScope
// ---------------------------------------------------------------------------
// When the device profiler is OFF (the default for every functional run, every
// golden test, and every shipped build) the macro expands to a construct with
// no runtime cost: the profiler's own build-mode switch compiles the marker
// writes out entirely.  So instrumentation is NOT something to add for a perf
// pass and strip afterwards — stripping it costs nothing at runtime and destroys
// the op's per-stage observability for every later round.  Extend it when a new
// path is added; never remove it.
//
// ---------------------------------------------------------------------------
// WHAT A ZONE ACTUALLY MEASURES (read before placing one)
// ---------------------------------------------------------------------------
// A zone reports the wall-clock delta between two braces.  It has no model of
// work vs. waiting.  Consequences that decide where the braces go:
//
//  1. A zone that encloses `cb_wait_front` / `cb_reserve_back` — DIRECTLY or
//     inside a kernel_lib helper that does its own CB management — reports
//     wait + work.  A *starved* stage is then numerically indistinguishable
//     from an *expensive* one.  Such a number is OCCUPANCY, not cost.
//  2. Only UNPACK blocks on `cb_wait_front`/`cb_pop_front` and only PACK on
//     `cb_reserve_back`/`cb_push_back` (they are `UNPACK(...)` / `PACK(...)`
//     macros).  A compute zone therefore yields THREE different numbers about
//     three different threads; the math thread has no wait call to hoist and is
//     unconditionally wait + work.
//  3. A NoC region has two costs: ISSUING the transactions (RISC-serial, scales
//     with transaction count) and the BARRIER (the actual wait).  Zone them
//     separately — a barrier zone at ~0 does NOT mean the transfer was free, it
//     means the cost was paid in the issue loop.
//  4. Do NOT zone `tile_regs_acquire/commit/wait/release`.
//
// Preferred shape, when the wait can be hoisted out of the payload:
//
//     { MaybeDeviceZoneScope("wait_in");  cb_wait_front(cb_in, n); }
//     { MaybeDeviceZoneScope("work");     helper(...); }
//
// ---------------------------------------------------------------------------
// MARKER BUDGET — exhaustion is SILENT
// ---------------------------------------------------------------------------
// 250 optional markers per RISC per dispatch
// (`PROFILER_L1_OPTIONAL_MARKER_COUNT`, hostdevcommon/profiler_common.h).  A
// zone costs 2 markers, so ~125 zone EXECUTIONS per RISC — executions, not
// distinct names, so a zone inside a per-tile loop burns 2 markers per
// iteration.  Past the budget `profileScope`'s ctor writes nothing and its dtor
// skips the end marker: the zone is ABSENT, starts/ends stay balanced, and no
// warning is logged.  The report looks complete while everything after the
// cutoff is missing.  Before trusting a breakdown, check that the zones cover
// the whole `*-KERNEL` span and that no (core, RISC) sits at the cap.
//
// Zones share SRAM with DPRINT / Watcher and cannot run together with them.
//
// ---------------------------------------------------------------------------
// USAGE
// ---------------------------------------------------------------------------
//   dataflow kernels : this header is optional — `dataflow_api.h` already pulls
//                      the profiler in transitively.  Including it is still fine
//                      and is what documents the intent.
//   compute kernels  : include this header (it pulls in the profiler, which
//                      `compute_kernel_api.h` does not).
//
// One zone per BLOCK: `DeviceZoneScopedN` declares `hash` and `zone` in the
// enclosing scope, so two zones in the same scope are a redeclaration error.
// Give each zone its own braces.

#pragma once

#include "tools/profiler/kernel_profiler.hpp"

// The op-facing name.  Direct alias for DeviceZoneScopedN — same RAII scope,
// same zero cost when the profiler is off, same marker budget.
#ifndef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#endif
