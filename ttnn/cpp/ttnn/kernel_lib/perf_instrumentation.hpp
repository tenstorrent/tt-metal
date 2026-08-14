// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// MaybeDeviceZoneScope — the PERMANENT per-stage instrumentation macro.
//
// ## Durability contract
//
// This macro is meant to be written into a kernel ONCE and left there forever.
// It is a direct alias for `DeviceZoneScopedN`, which is itself compiled out to
// nothing whenever the device profiler is off (`PROFILE_KERNEL` unset — the
// default for every production JIT build). So:
//
//   * with the profiler OFF it costs ZERO instructions, ZERO L1 and ZERO ns;
//   * with the profiler ON (`TT_METAL_DEVICE_PROFILER=1`) it emits a
//     start/end marker pair and the stage shows up in
//     `generated/profiler/.logs/profile_log_device.csv`.
//
// Therefore **never delete a MaybeDeviceZoneScope to "clean up" or to "make the
// kernel faster"** — there is nothing to clean up and nothing to win. Removing
// it only destroys the op's per-stage observability, which the next perf round
// has to rebuild from scratch.
//
// ## How to place it (see .claude/references/device-zone-scope-attribution.md)
//
// A zone is a stopwatch on a block: whatever is inside the braces is inside the
// number, including a `cb_wait_front` hidden inside a helper. So a zone that
// encloses a wait reports OCCUPANCY, not cost. Split the scopes so waiting and
// working are different numbers:
//
//     { MaybeDeviceZoneScope("stage_wait");  cb_wait_front(cb, n); }
//     { MaybeDeviceZoneScope("stage_issue"); for (...) noc_async_read(...); }
//     { MaybeDeviceZoneScope("stage_barrier"); noc_async_read_barrier(); }
//
// Budget: 250 optional markers per RISC per dispatch
// (`PROFILER_L1_OPTIONAL_MARKER_COUNT`), i.e. ~125 zone EXECUTIONS. A zone in a
// per-block loop burns 2 markers per iteration and exhaustion is SILENT — the
// zones simply stop appearing partway through the kernel with no warning. Check
// that the recorded zones cover the whole `*-KERNEL` span before trusting a
// breakdown.
//
// Each expansion declares a `constexpr hash`, so two zones cannot live in the
// same C++ scope — always give a zone its own braces.
//
// Dataflow kernels get the profiler transitively through `dataflow_api.h`;
// compute kernels must include this header (which pulls it in explicitly).

#pragma once

#include "tools/profiler/kernel_profiler.hpp"

#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
