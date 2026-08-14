// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Permanent, zero-cost-when-off per-stage instrumentation for TTNN kernels.
//
// ---------------------------------------------------------------------------
// Durability contract
// ---------------------------------------------------------------------------
// `MaybeDeviceZoneScope("<stage>")` is a direct alias for `DeviceZoneScopedN`,
// which the profiler headers compile down to `(void(name))` when device
// profiling is not enabled (tt_metal/tools/profiler/kernel_profiler.hpp).  So a
// scope left in a shipped kernel costs NOTHING in a production build.
//
// Because it is free when off, the instrumentation is PERMANENT: it is part of
// the op, not a temporary probe.  Never delete a scope to "clean up" a kernel —
// deleting it destroys the op's per-stage observability and the next perf pass
// has to rediscover the breakdown from scratch.  When a new code path is added,
// EXTEND the scopes to cover it.
//
// ---------------------------------------------------------------------------
// How to place a scope (see .claude/references/device-zone-scope-attribution.md)
// ---------------------------------------------------------------------------
//  * A zone times whatever is inside its braces — including `cb_wait_front` /
//    `cb_reserve_back`, including a helper's INTERNAL CB management.  A zone
//    around a wait therefore reports OCCUPANCY, not payload cost.
//  * Split wait from work so the two answer different questions:
//        { MaybeDeviceZoneScope("stage_wait"); cb_wait_front(...); }
//        { MaybeDeviceZoneScope("stage");      work(...);          }
//  * For NoC, split ISSUE from BARRIER: a barrier zone reading ~0 does not mean
//    the transfer was free, it means the issue loop already paid for it.
//  * A compute-kernel zone is recorded THREE times (unpack / math / pack) and
//    the three numbers are about three different threads.
//  * Budget: 250 optional markers per RISC per dispatch = ~125 zone EXECUTIONS.
//    A zone inside a loop burns 2 markers per iteration and exhaustion is
//    SILENT (later zones simply vanish).  Verify the zones cover the whole
//    kernel span before ranking anything off them.
//
// Dataflow kernels get the profiler transitively through `dataflow_api.h`;
// compute kernels must include this header explicitly.

#pragma once

#include "tools/profiler/kernel_profiler.hpp"

// The one spelling every rms_norm-style op uses for a stage boundary.
#ifndef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#endif
