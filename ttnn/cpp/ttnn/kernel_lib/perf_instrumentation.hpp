// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PERMANENT per-stage device instrumentation for TTNN kernels.
//
// ---------------------------------------------------------------------------
// DURABILITY CONTRACT
// ---------------------------------------------------------------------------
// `MaybeDeviceZoneScope("<stage>")` is a direct alias for `DeviceZoneScopedN`.
// When the device profiler is OFF (the default for every functional run) the
// macro expands to nothing that costs a single instruction — it is a
// compile-time-hashed RAII object whose body is compiled out
// (tt_metal/tools/profiler/kernel_profiler.hpp). Therefore:
//
//   * It is CHEAP: leaving it in costs zero device time in production.
//   * It is PERMANENT: once a stage boundary is instrumented, the zone STAYS.
//     Never delete a zone to "clean up" a kernel — the next perf round, and
//     every regression triage after it, starts from these numbers. A kernel
//     that loses its zones has to be re-instrumented from scratch, and the
//     re-instrumentation is where attribution mistakes get made.
//   * A new code path (a fused variant, a dual path, a fast case) MUST carry
//     the same zone names as the path it replaces, or per-stage observability
//     silently regresses to "the new path is one opaque number".
//
// ---------------------------------------------------------------------------
// WHAT A ZONE MEASURES (see .claude/references/device-zone-scope-attribution.md)
// ---------------------------------------------------------------------------
// A zone is a stopwatch on a block: whatever sits between the macro and the
// closing brace is inside the number, including any `cb_wait_front` /
// `cb_reserve_back` a helper performs internally. So a zone that encloses a
// wait reports OCCUPANCY, not cost — a starved stage and an expensive stage are
// numerically identical. Split them:
//
//     { MaybeDeviceZoneScope("wait_in");  cb_wait_front(cb_in, n); }
//     { MaybeDeviceZoneScope("work");     the_payload(...); }
//
// and split a NoC region into issue vs. barrier, because a barrier ~ 0 does not
// mean "hidden" — it means the RISC-serial issue cost was paid in whatever zone
// covered the issue loop:
//
//     { MaybeDeviceZoneScope("read_issue");   for (...) noc_async_read_tile(...); }
//     { MaybeDeviceZoneScope("read_barrier"); noc_async_read_barrier(); }
//
// A compute-kernel zone is recorded THREE times (unpack / math / pack); read
// each as that thread's occupancy, never as one "stage cost". Only unpack can
// block on tiles arriving and only pack on space, so a math-thread zone is
// unconditionally wait+work and cannot be split.
//
// ---------------------------------------------------------------------------
// MARKER BUDGET — exhaustion is SILENT
// ---------------------------------------------------------------------------
// 250 optional markers per RISC per dispatch (PROFILER_L1_OPTIONAL_MARKER_COUNT),
// and a zone costs 2 markers per EXECUTION (not per name). ~125 zone executions
// per RISC. Past the cap, zones are simply ABSENT — starts and ends stay
// balanced and nothing is logged, so a truncated profile looks complete and
// invents a dominant stage. Prefer zones OUTSIDE hot loops; before ranking
// anything, check that the recorded zones cover the whole `*-KERNEL` span for
// each RISC in `profile_log_device.csv`.
//
// Zones share SRAM with DPRINT / Watcher and cannot run together.
//
// ---------------------------------------------------------------------------
// USAGE
// ---------------------------------------------------------------------------
//   compute kernels : #include this header (it pulls kernel_profiler.hpp in).
//   dataflow kernels: dataflow_api.h already provides the profiler; including
//                     this header anyway is harmless and self-documenting.
//
// One zone per enclosing block: the underlying macro declares `hash` and `zone`
// in the current scope, so two zones in the same braces will not compile. Wrap
// each in its own `{ ... }`.

#pragma once

#include "tools/profiler/kernel_profiler.hpp"

// A named device zone that is free when the profiler is off, and PERMANENT.
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
