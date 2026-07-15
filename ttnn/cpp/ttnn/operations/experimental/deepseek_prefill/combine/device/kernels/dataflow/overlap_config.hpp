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

// MOCK_COMBINE_INTERNALS: 0 = real dram->untilizer->reader_combine producer chain feeds c_3 (unmocked);
// 1 = writers self-generate synthetic tokens (perf/liveness only). With USE_RELAY, the relay is agnostic
// to this — it just drains its c_24 ring; only the SENDER's token source differs (real c_3 vs synthetic).
#define MOCK_COMBINE_INTERNALS 0
#define MOCK_COMBINE_TOKENS_PER_DEST 200

// ============================================================================
// SENDERS_ONLY_MOCK — STAGE 1: remove untilizer cores (no placement change)
// ============================================================================
//
// Incremental experiment, built in stages so a hang can be pinpointed early.
//   STAGE 1 (this): the factory allocates NO untilizer cores/kernels/CBs/semaphores. Senders keep
//     their DEFAULT positions (no placement change). Requires MOCK_COMBINE_INTERNALS=1: sender
//     writer runs synthetic fabric traffic; sender reader exits early (its existing mock return) and
//     never blocks on untilizers (with 0 untilizers the INIT_ZEROS output-zeroing wait is a no-op /
//     compiled out, and the untilizer polling loop is skipped).
//   Later stages will add custom sender placement under a SEPARATE toggle.
// HOST-FACTORY change -> needs a libttnn rebuild. Set to 0 (and re-comment MOCK above) to restore
// normal combine (byte-identical). Diagnostics/perf experiment, not a correctness path.
// NOTE: SENDERS_ONLY_MOCK / SENDER_ONE_TO_ROW_Y3 are OFF for the relay experiment (USE_RELAY below):
// the relay plan needs the full R-S-U-U-U-U chain, i.e. the untilizers must be present, and the relay
// experiment sets placement explicitly (superseding the SENDER_ONE_TO_ROW_Y3 predicate).
#define SENDERS_ONLY_MOCK 0

// STAGE 3: move ONE of the 2 senders to physical row y=3 (logical y=1), keeping its column; the other
// sender stays on y=2. No untilizers, no other changes. Verifies that a sender on a different physical
// row (its worker->eth NOC path + fabric connection originating from y=3) is safe in isolation. The op
// stays on the full worker grid, which provides the y=3 worker core + init/exit GlobalSemaphore
// coverage. Requires SENDERS_ONLY_MOCK=1. Set to 0 to keep both senders on y=2 (Stage 1 behavior).
#define SENDER_ONE_TO_ROW_Y3 0

// ============================================================================
// USE_RELAY — introduce a dedicated relay ("R") tensix core per combine row
// ============================================================================
//
// Motivation: the mandatory opposite-NOC rule (a core with both DM0+DM1 kernels cannot use the same
// NOC) means the sender core cannot issue its eth write on NOC_0 while reader_combine also runs on it.
// Breaking the untilizer->sender->eth chain into untilizer->sender->relay->eth moves the fabric send to
// a dedicated single-kernel core, which is then free to use NOC_0 for the eth write — without perturbing
// the sender/untilizer NOC assignments (kept identical to main).
//
// Developed in stages (see the relay plan):
//   STAGE 1 (this): add one relay core per row (order R-S-U-U-U-U), placed to the LEFT of the current
//     cores, with a deep L1 receive buffer (RELAY_SLOTS slots, each = route_info + one token payload)
//     and a single idle writer kernel pinned to NOC_0. The relay is unconnected — this stage only
//     validates placement, L1 fit, and grid growth; the op otherwise runs unchanged.
//   Later stages: move the sender->eth send to relay->eth (mock, then real), forward routing metadata
//     through the CB, and finally unmock the producers for functional parity with better perf.
//
// Explicit physical NOC0 placement (virtual==physical for unharvested BH tensix), num_cores==2:
//   y=2: R@x1, S@x2, U@x3, U@x4, U@x5, U@x6
//   y=3: R@x2, S@x7, U@x10, U@x11, U@x12, U@x13
//
// HOST-FACTORY change -> needs a libttnn rebuild. Set to 0 to remove the relay cores entirely.
#define USE_RELAY 1

// Slot count of the relay's L1 receive buffer. Each slot holds one token payload plus its routing
// metadata (l1_align route_info + aligned output page, ~14.4 KB/slot). 64 slots ~= 0.9 MB, comfortably
// within the relay core's L1 (it holds essentially nothing else). Reduce if program creation OOMs.
#define RELAY_SLOTS 64

// RELAY_TRID_OVERLAP_SEND — selects the relay's c_24 -> eth send flavor (writer_relay only):
//   1 (default): the OVERLAPING_TOKEN_WRITE trid-pipelined path. Each token is sent non-blocking
//       (fabric_send_noc_unicast_with_trid) with a pool of OVERLAP_POOL_DEPTH trids + headers, and the
//       completion wait (wait_on_flush_for_trid) lags by the pool depth so up to DEPTH sends overlap.
//       The slot credit is returned only AFTER that lagged wait retires the slot's read (see the loop),
//       so the sender never overwrites a slot whose payload read is still in flight. These trid helpers
//       (send_current_slot_non_blocking_with_trid, noc_async_write_barrier_with_trid on the fabric worker
//       NOC) were ADDED on this branch — they do NOT exist on main.
//   0: main-only flavor. Each token is sent with the blocking fabric_send_noc_unicast +
//       noc_async_writes_flushed() (exactly the writer_combine !OVERLAPING_TOKEN_WRITE path on main),
//       using a single reusable packet header. The flush completes the slot's payload read before the
//       credit is returned, so it is race-free without trids. Uses ONLY fabric utils present on main;
//       no overlap (one send in flight at a time). Use this to A/B correctness + perf against flavor 1.
// Kernel-only (JIT) toggle -> no libttnn rebuild. HW-only verification.
#define RELAY_TRID_OVERLAP_SEND 0

// RELAY_ROUNDROBIN_ROUTE — BW-spread experiment. When 1, the relay OVERRIDES each token's route +
// distance (from the CB slot) with round-robin values so traffic spreads across both active fabric
// directions and a range of hop-distances: route = i-th active direction alternating (i%num_active),
// distance = {1,2,3,3,2,1,4}[i%7], where i is the per-token index. Payload/page still come from the slot,
// so tokens are MISROUTED — output is garbage (perf/liveness only). Set 0 to route tokens for real.
#define RELAY_ROUNDROBIN_ROUTE 0
