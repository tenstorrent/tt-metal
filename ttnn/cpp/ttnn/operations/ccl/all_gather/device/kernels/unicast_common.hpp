// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "gather_walk.hpp"
#include "gather_noc.hpp"
#include "gather_args.hpp"

// Store-and-forward AllGather: every fabric send is a single 1-hop unicast to the neighbor (HopSink in
// gather_fabric.hpp). Runs on any effectively-1D topology (both Fabric 1D and 2D).

////////////////////////////////////////////////////////////////
// data_valid semaphore protocol
//
// data_valid counts the chunks upstream has relayed into our output -- cumulative over the op. Chunks are
// walk chunks (page / k), so every count here is k times the page count. A chunk's
// absolute position is base_seqno + its seqno in the batch, with base_seqno = (iter-1) * slice_chunks +
// skip. The writer maintains the count (atomic-inc per chunks delivered); the reader waits on it with
// noc_semaphore_wait_min at the last chunk of each batch it reads, then a final wait for total_chunks.
//
// This semaphore is reused across cached invocations, so the reader subtracts total_chunks instead of
// clearing, which would drop credits posted by an upstream which is an invocation ahead. total_chunks
// must be exact: too many and a later reader passes early.
//
// Waiting on an absolute position (not a signal count) lets one reader path cover every case with no
// alignment or per-topology special-casing:
//   - full relay, and even-ring split prefix half (skip 0) / suffix half (skip = half): same per-batch
//     wait, differing only in base_seqno/take;
//   - sink stripe (a line endpoint's incoming, or a ring antipode): no relay wait, covered by the final
//     total_chunks wait;
//   - sink direction (num_iters == 0): only the total_chunks wait runs.
// So data_valid_granularity is a pure writer-side perf knob: the reader auto-paces to the writer's cadence.
////////////////////////////////////////////////////////////////
