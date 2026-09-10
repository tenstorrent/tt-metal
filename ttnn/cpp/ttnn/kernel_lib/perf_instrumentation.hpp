// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Permanent, zero-cost device-side stage instrumentation for TTNN kernels.
//
// USAGE
//   #include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
//   {
//       MaybeDeviceZoneScope("reader_issue");
//       ...
//   }                                   // <-- the zone stops HERE
//
// Dataflow kernels get the profiler transitively through `dataflow_api.h`;
// compute kernels must include this header explicitly.
//
// DURABILITY CONTRACT — these scopes are PERMANENT.
//   * They compile to NOTHING unless the build defines `PROFILE_KERNEL`
//     (i.e. unless the run is under `run_safe_pytest.sh --profile`). There is
//     no runtime cost, no register pressure and no code-size cost in a normal
//     build, so there is never a perf reason to delete one.
//   * A later refinement that adds a stage MUST add its zone. Per-stage
//     observability is a property of the op, not of the pass that measured it
//     once; deleting a scope silently destroys the next round's breakdown.
//
// WHAT A ZONE ACTUALLY MEASURES (see .claude/references/device-zone-scope-attribution.md)
//   A zone is a stopwatch on a block — whatever is inside the braces is inside
//   the number, INCLUDING a `cb_wait_front` / `cb_reserve_back` a helper does
//   internally. So a zone around a stage that is merely STARVED reads exactly
//   like a stage that is EXPENSIVE. Consequences for where to put them:
//     * split the WAIT from the WORK when you need cost rather than occupancy;
//     * split a NoC region into ISSUE (RISC-serial, scales with transaction
//       count) and BARRIER (waiting for bytes to land) — a barrier that reads
//       ~0 does NOT mean the transfer was free, it means the issue loop paid;
//     * a compute zone yields THREE numbers (unpack / math / pack); only
//       unpack blocks on tiles arriving and only pack on space, so the math
//       thread's number is unconditionally wait+work and is occupancy only;
//     * do NOT zone `tile_regs_*` — the DEST handoff is not worth a marker.
//
// BUDGET — 250 optional markers per RISC per dispatch
// (`PROFILER_L1_OPTIONAL_MARKER_COUNT`), i.e. ~125 zone EXECUTIONS. It counts
// executions, not names: a zone in a per-tile loop burns 2 markers per
// iteration. Exhaustion is SILENT — the profiler simply stops recording and the
// report still looks complete. Before ranking anything off a breakdown, check
// that the last user zone's end reaches the `*-KERNEL` span for that RISC.
#pragma once

// `kernel_profiler.hpp` self-guards on `PROFILE_KERNEL`: off a profiled build
// every macro below degenerates to `(void(sizeof(name)))`. Dataflow kernels
// already pull it in through `dataflow_api.h`; including it again is a no-op.
#include "tools/profiler/kernel_profiler.hpp"

// A direct alias for `DeviceZoneScopedN`, named for the durability contract
// above so a reader can tell an instrumentation scope from an ad-hoc one.
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
