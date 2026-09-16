// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// perf_instrumentation.hpp — permanent per-stage device zones for TTNN op kernels.
//
//   MaybeDeviceZoneScope("stage_name");
//
// is an RAII stopwatch on the enclosing block (a direct alias of the profiler's DeviceZoneScopedN).
// It is meant to be left in production kernels at every stage boundary so a perf pass can read a
// per-stage breakdown off any `--profile` run without re-instrumenting.
//
// Durability contract
//   * Profiler OFF (the normal build, PROFILE_KERNEL undefined): the macro expands to a no-op
//     `(void(sizeof(name)))` — zero code, zero registers, zero L1. It is free. NEVER remove it.
//   * Profiler ON (`--profile` / TT_METAL_DEVICE_PROFILER=1): each executed zone writes 2 markers
//     (start + end) into the per-RISC L1 profiler buffer. The buffer holds 250 optional markers per
//     RISC per dispatch, i.e. ~125 zone EXECUTIONS per RISC; a zone inside a per-tile loop burns its
//     2 markers on every iteration. Running out is SILENT (later zones are simply absent, the report
//     still looks complete) — so put zones on stage boundaries, not on tile loops, and check that
//     the last user zone reaches the *-KERNEL span before trusting a breakdown.
//   * Profiler ON but zones unwanted (e.g. a whole-op number where the marker writes themselves are
//     a measurable share of a µs-scale kernel): compile with -DKERNEL_LIB_PERF_ZONES_OFF (a kernel
//     `defines` entry) and every MaybeDeviceZoneScope compiles out while the *-KERNEL spans and the
//     op CSV stay intact.
//
// What a zone measures — occupancy, not payload. Whatever sits inside the braces is inside the
// number, including a helper's internal cb_wait_front / cb_reserve_back, so a starved stage is
// indistinguishable from an expensive one. On the compute kernel the same zone is recorded three
// times (unpack / math / pack) and only the unpack (wait_front) and pack (reserve_back) threads can
// wait on a CB; a math-thread number is always wait + work. Split waits from work into separate
// zones, zone NoC issue loops separately from their barriers, and confirm any "stage X dominates"
// verdict with a payload ablation. Full guidance: .claude/references/device-zone-scope-attribution.md.
//
// Two zones in ONE scope collide (the macro declares `hash` and `zone`): give each its own block.
//
// Dataflow kernels get the profiler through dataflow_api.h; compute kernels do not, hence the include.

#pragma once

#include "tools/profiler/kernel_profiler.hpp"

#if defined(PROFILE_KERNEL) && !defined(KERNEL_LIB_PERF_ZONES_OFF)
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#else
#define MaybeDeviceZoneScope(name) (void(sizeof(name)))
#endif
