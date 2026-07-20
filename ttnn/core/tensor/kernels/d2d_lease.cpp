// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tiny per-coord fabric-link lease helper for D2DStreamService, run on ONE worker core
// per coord. It NoC-accesses the persistent service core's `link_grant` word so the
// host-driven lease (wait_for_fabric_links / release_fabric_links) is a CQ-ordered,
// enqueued workload instead of a host-side PCIe poke. Being CQ-ordered is the whole
// point: a RELEASE enqueued BEFORE the producer workload grants the service its turn so
// the producer's handshake completes against an already-granted service; a WAIT enqueued
// AFTER fences the next op until the service hands the link back. The host never blocks
// on a PCIe loop.
//
//   LEASE_MODE == WAIT (0):    spin NoC-reading link_grant until it is 0 (the service
//                              has handed the links back), then exit. Enqueued after the
//                              producer, so any later CQ op is ordered after the link is
//                              free — no host Finish needed.
//   LEASE_MODE == RELEASE (1): NoC-write 1 to link_grant (grant the service one turn),
//                              then exit. Enqueued BEFORE the producer workload, so the
//                              grant is in place by the time the producer runs.
//
// link_grant stays a service-core L1 word (the service kernel reads/writes it locally
// on its hot path); only this helper reaches it cross-core over NoC. The service writes
// 0, this helper (RELEASE) writes 1 — strict alternation enforced by CQ order, so the
// two never write concurrently.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"

#define LEASE_MODE_WAIT 0
#define LEASE_MODE_RELEASE 1

#ifndef LEASE_MODE
#error "LEASE_MODE must be defined (0 = WAIT, 1 = RELEASE) via DataMovementConfig.defines"
#endif

// CT[0]: scratch CB (WAIT only — a 1-word L1 staging slot for the NoC read). Provided
// to both builds; unread by RELEASE.
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(0);

void kernel_main() {
    // RT: [0]=service-core NoC x, [1]=service-core NoC y, [2]=link_grant L1 addr.
    const uint32_t svc_noc_x = get_arg_val<uint32_t>(0);
    const uint32_t svc_noc_y = get_arg_val<uint32_t>(1);
    const uint32_t link_grant_addr = get_arg_val<uint32_t>(2);

    Noc noc;
    UnicastEndpoint service;  // service-core link_grant word, addressed via RT args below

#if LEASE_MODE == LEASE_MODE_WAIT
    // Spin until the service core has dropped the link (link_grant == 0).
    CircularBuffer scratch_cb(scratch_cb_index);
    const uint32_t scratch = scratch_cb.get_write_ptr();
    CoreLocalMem<uint32_t> scratch_mem(scratch);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    do {
        noc.async_read(
            service,
            scratch_mem,
            sizeof(uint32_t),
            {.noc_x = svc_noc_x, .noc_y = svc_noc_y, .addr = link_grant_addr},
            {});
        noc.async_read_barrier();
    } while (*p != 0u);
#else
    // Grant the service core one transfer (link_grant = 1). INLINE_L1: the target is an
    // L1 word (the 2.0 wrapper handles the Blackhole inline-L1 scratch quirk internally).
    noc.inline_dw_write<NocOptions::INLINE_L1>(
        service, 1u, {.noc_x = svc_noc_x, .noc_y = svc_noc_y, .addr = link_grant_addr});
    noc.async_writes_flushed();
#endif
}
