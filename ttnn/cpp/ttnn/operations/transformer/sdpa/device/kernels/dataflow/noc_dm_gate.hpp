// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Compute-ceiling instrumentation for the ring-joint SDPA dataflow kernels.
//
// When RING_JOINT_DISABLE_NOC_DM is defined (set ONLY on the ring_joint reader/writer kernels by
// the program factory, gated on the TT_RING_JOINT_DISABLE_NOC_DM env var), the bulk NoC
// data-movement primitives below are compiled to no-ops. Cross-kernel synchronization is left
// fully intact:
//   - kept: cb_reserve_back / cb_push_back / cb_wait_front / cb_pop_front (compute handshake)
//   - kept: noc_semaphore_* (incl. noc_semaphore_set_multicast / set_remote — the chain handoff)
//   - kept: all *_barrier / *_flushed* / *_set_trid ordering primitives (cheap no-ops with nothing
//           in flight)
//   - dropped: noc_async_read / noc_async_read_page / noc_async_read_one_packet_set_state,
//              noc_async_write / noc_async_write_page / noc_async_write_multicast (the actual
//              DRAM/L1 reads, unicast writes, and K/V multicast payloads)
//
// The result: the compute kernel runs on whatever stale data is already in L1 (outputs are
// garbage) but is never throttled by data movement, so the measured kernel duration reflects the
// compute ceiling. This header is included AFTER dataflow_api.h so the real declarations are not
// clobbered; the function-like macros only rewrite *call sites* that follow.
//
// Note: noc_async_read_one_packet_with_state<true>(...) cannot be intercepted here (a function-like
// macro does not match an identifier followed by '<'); those two call sites are guarded inline.

#ifdef RING_JOINT_DISABLE_NOC_DM

#define noc_async_read(...) ((void)0)
#define noc_async_read_page(...) ((void)0)
#define noc_async_read_one_packet_set_state(...) ((void)0)
#define noc_async_write(...) ((void)0)
#define noc_async_write_page(...) ((void)0)
#define noc_async_write_multicast(...) ((void)0)

#endif  // RING_JOINT_DISABLE_NOC_DM
