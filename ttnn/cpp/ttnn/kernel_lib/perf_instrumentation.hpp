// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file perf_instrumentation.hpp
 * @brief Permanent, opt-in per-stage profiling instrumentation for op kernels.
 *
 * Provides MaybeDeviceZoneScope(name) — a device zone scope that measures the
 * wall-clock of the enclosing stage (reader read, a compute phase, writer pack,
 * ...) ONLY when the kernel is built with the profiler on, and expands to
 * nothing otherwise.
 *
 * It wraps DeviceZoneScopedN (tt_metal/tools/profiler/kernel_profiler.hpp),
 * which already self-compiles to a no-op when PROFILE_KERNEL is undefined — so
 * this macro costs *nothing* in a normal (non-profiling) run. The profiler is
 * enabled by the device-profiler env vars (TT_METAL_DEVICE_PROFILER=1 + the
 * mid-run-dump / post-process vars); /perf-measure documents the exact set.
 *
 * DURABILITY CONTRACT — this is the load-bearing rule:
 *   These zone scopes are a PERMANENT FIXTURE. NEVER remove them from a kernel
 *   and NEVER leave a stage boundary un-instrumented. Because the gate makes
 *   them free when off, there is no cost to keeping them, and keeping them means
 *   the op stays re-profileable per-stage forever — anyone can flip the profiler
 *   on and read where the time goes, on any shape, at any time. This is the
 *   OPPOSITE of DEVICE_PRINT, which is stripped before a green commit. When a
 *   new (e.g. predicate-guarded) code path is added to a kernel, extend the
 *   instrumentation to cover it too, so observability never regresses.
 *
 * Usage (name is a string literal):
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
 *   ...
 *   {
 *       MaybeDeviceZoneScope("reader_read");   // scoped: measures this block
 *       // ... the reader's NoC reads for one stage ...
 *   }
 *
 * The zone is RAII-scoped: it times from the macro site to the end of the
 * enclosing block. Put each stage in its own block (or rely on the natural
 * scope) so the per-stage durations are cleanly separated.
 *
 * Dataflow kernels already pull in kernel_profiler.hpp transitively via
 * "api/dataflow/dataflow_api.h"; compute kernels get it from the include below.
 */

#include "tools/profiler/kernel_profiler.hpp"

// Real device zone scope under the profiler build; a no-op otherwise (the gate
// lives inside DeviceZoneScopedN — see kernel_profiler.hpp). One name to
// standardize on across ops so the durability contract above is greppable.
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
