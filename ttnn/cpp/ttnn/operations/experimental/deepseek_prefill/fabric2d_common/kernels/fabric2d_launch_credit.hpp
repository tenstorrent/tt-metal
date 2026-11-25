// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The launch credit, shared by the dispatch_fabric2d and combine_fabric2d senders. Kernel-only. Both ops reuse
// each stream's forwarding region at the same offsets every launch, so a sender may write to the next chip only
// once that chip has started the same launch; the protocol is the same in both and so has one definition.
//
// Each sender grants one credit to the downstream chip's reverse stream core (the stream on that chip which
// sends back to this one) and takes one from its own counter. The caller passes its own compile-time
// arguments: which header to send from, where that core is and the counter's L1 address, uniform across the
// mesh.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "hostdev/fabric_edm_packet_header.hpp"

namespace fabric2d {

// Tell the chip across the cable that this chip has started the launch. Its sender on the stream back toward
// us waits for this before it writes to us. Being in this launch means every kernel of this op's previous
// launch on this chip, our readers included, has finished: a launch's go signal waits for every core of its
// sub-device to finish the one before.
//
// Sent before this kernel waits on the reader or on any other chip, so the credit never depends on this
// launch making progress.
//
// `pkt_hdr_addr` must already carry the route to the chip across the cable; this sets only the command fields,
// and the blocking send leaves the header free for its next use.
template <typename FabricSender>
void grant_launch_credit(
    FabricSender& fabric,
    uint32_t pkt_hdr_addr,
    uint32_t credit_noc_x,
    uint32_t credit_noc_y,
    uint32_t launch_credit_addr) {
    volatile PACKET_HEADER_TYPE* hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_addr);
    hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        get_noc_addr(credit_noc_x, credit_noc_y, launch_credit_addr), /*val=*/1, /*flush=*/true});
    fabric.wait_for_empty_write_slot();
    fabric.send_payload_flush_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
}

// Wait until the chip across the cable has started this launch, then take its credit. Taken by subtracting
// one rather than by zeroing: that chip can start the next launch, and grant the next credit, while this
// one is still running.
inline void take_launch_credit(uint32_t launch_credit_addr) {
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(launch_credit_addr), 1);
    noc_semaphore_inc(get_noc_addr(launch_credit_addr), 0u - 1u);
}

}  // namespace fabric2d
