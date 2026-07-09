// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ============================================================================
// OVERLAPING_TOKEN_WRITE — overlap multiple sender->EDM L1 token transfers
// ============================================================================
//
// Goal: in the real (non-mocked) writer_combine send loop, keep several worker->EDM
// payload+header NOC writes in flight at once so the only per-token stall is
// wait_for_empty_write_slot(), the same regime the mocked sender reached (~35-40 GB/s).
//
// This is developed in stages; every stage is gated by this single toggle so the whole
// feature can be turned off (0 => the original blocking path, byte-for-byte).
//
//   Stage 1: split the blocking fabric send into a non-blocking ISSUE
//            (perform_payload_send<false,false>, via fabric_send_noc_unicast_nonblocking)
//            plus the existing noc_async_writes_flushed() WAIT, called adjacently. No
//            timing change yet — this just decomposes the call so later stages can move
//            the wait.
//   Stage 2: give each send a NOC transaction id and wait on that specific trid
//            (still issued+waited adjacently). Requires a trid-tagged send on the fabric
//            worker adapter.
//   Stage 3: cycle a small pool (e.g. 8) of trids instead of reusing one (still adjacent).
//   Stage 3b: add a matching pool of packet headers (one per in-flight slot) so a header
//            is never overwritten while its NOC read is still in flight.
//   Stage 4: move each wait to BEFORE the issue that reuses its trid/header/CB slot, i.e.
//            lag the completion wait by the pool depth. Needs the cb_route_info CB deepened
//            to more than the pool depth (16 slots is a good starting point) and a
//            peek-ahead read of the CB (offset from get_read_ptr, wrapping fifo_limit)
//            since multiple slots are held un-popped.
//
// Set to 0 to restore the original one-at-a-time blocking send path.
#define OVERLAPING_TOKEN_WRITE 1

// Depth of the lockstep trid + packet-header pools (stage 3 / 3b). Each in-flight token uses trid
// (slot + 1) — trid 0 is avoided — and header pool[slot], cycling slot over [0, OVERLAP_POOL_DEPTH).
// At stage 3 the wait is still adjacent (one in flight); at stage 4 this becomes the max number of
// outstanding sends (the wait lags by this many). Must be <= 15 (Blackhole has 16 NOC trids 0..15,
// trid 0 reserved) and, at stage 4, strictly less than the cb_route_info depth. The factory grows the
// c_5 packet-header CB to (2 + OVERLAP_POOL_DEPTH) headers when OVERLAPING_TOKEN_WRITE is on.
#define OVERLAP_POOL_DEPTH 8

// Depth (in slots) of the cb_route_info (c_3) CB between the sender reader and sender writer when
// OVERLAPING_TOKEN_WRITE is on. NOTE: stage 4 uses per-iteration pop, so the overlap depth actually
// lives in the trid+header pool (c_5) + the EDM buffer, NOT here — c_3 does not need to be >= the pool
// depth for the pipeline. A deeper c_3 only decouples reader jitter and reduces reader-write vs
// sender-read L1 contention on hot slots (a measurement-quality knob). Default is 2x the pool depth
// (the "ideal" headroom). Each slot is ~14.4 KB (l1_align + aligned_output_page_size), and the sender
// core's L1 is dominated by c_18 receive_buf (k_s * 16 * ~14.3 KB, k_s <= 4). If program creation
// fails with an L1 out-of-memory error, reduce this (down to 2 is fine for correctness) or reduce
// SLOTS_PER_UNTILIZER in the factory.
#define OVERLAP_ROUTE_INFO_CB_SLOTS (2 * OVERLAP_POOL_DEPTH)

#define MOCK_COMBINE_INTERNALS 1
#define MOCK_COMBINE_TOKENS_PER_DEST 200
