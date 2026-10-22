// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"

#include <cstdint>

#include "packetizer.hpp"
#include "route.hpp"

using tt::tt_fabric::common::experimental::close_connections;
using tt::tt_fabric::common::experimental::open_connections;

// Pump: the fabric side of a builder/pump core. The builder leaves complete, routed packets in the CB; the pump
// sends them in CB order plus its kernel's per-slot work (local copy, signals). The host tells it how many units
// to expect, so it needs no per-slot metadata beyond the header.

// Waits for `count` on a local semaphore, then takes it back. Subtract, never clear: a peer one invocation
// ahead may already have posted its next increments.
FORCE_INLINE void sem_wait_consume(uint32_t sem, uint32_t count) {
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), count);
    noc_semaphore_inc(get_noc_addr(sem), uint32_t{0} - count);
    noc_async_atomic_barrier();
}

// A semaphore-inc header on one route. The inc is flushed: the receiving EDM lands every earlier write on
// this route before it, which is what lets a counter stand for delivered data.
class IncSender {
public:
    template <bool kMcast>
    void init(tt::tt_fabric::WorkerToFabricEdmSender& edm, const RouteArgs& route, uint32_t variant) {
        edm_ = &edm;
        hdr_ = PacketHeaderPool::allocate_header();
        set_route<kMcast>(hdr_, route, variant);
    }

    // `x, y, addr` name a semaphore in NOC0 coordinates on the receiving device.
    void send(uint32_t x, uint32_t y, uint32_t addr, uint32_t val) {
        // The previous send may still be reading this header.
        noc_async_writes_flushed();
        hdr_->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(x, y, addr, 0), val, true});
        edm_->wait_for_empty_write_slot();
        edm_->send_payload_flush_non_blocking_from_address(
            reinterpret_cast<uint32_t>(hdr_), sizeof(PACKET_HEADER_TYPE));
    }

private:
    tt::tt_fabric::WorkerToFabricEdmSender* edm_ = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_ = nullptr;
};

template <
    uint32_t cb,
    uint32_t slots,
    uint32_t slot_bytes,
    uint32_t payload_off,
    uint32_t hdr_bytes,
    uint32_t C,
    uint32_t A>
class Pump {
    static_assert(noc_index == tt::tt_fabric::edm_to_local_chip_noc, "local copy uses header addresses as-is");
    static_assert(sizeof(PACKET_HEADER_TYPE) == hdr_bytes, "host and kernel disagree on the header size");
    static_assert(C >= 1 && C <= tt::tt_fabric::RoutingPlaneConnectionManager::MaxConnections);

public:
    explicit Pump(tt::tt_fabric::RoutingPlaneConnectionManager& conns) : conns_(conns), base_(get_read_ptr(cb)) {}

    // Sends slots until `total` units have gone. on_slot(slot, units) runs once a slot is issued; on_idle()
    // runs before blocking on the builder.
    template <class OnSlot, class OnIdle>
    void run(uint32_t total, OnSlot&& on_slot, OnIdle&& on_idle) {
        for (uint32_t sent = 0; sent < total;) {
            const int32_t want = held_ ? 2 : 1;
            if (!cb_pages_available_at_front(cb, want)) {
                on_idle();
                cb_wait_front(cb, want);
            }
            // The CB handshake does not fence: without this a CPU read can see the slot's previous use.
            invalidate_l1_cache();
            const uint32_t slot = base_ + next_ * slot_bytes;
            next_ = next_ + 1 == slots ? 0 : next_ + 1;
            const uint32_t bytes = header(slot, 0)->payload_size_bytes;
            for (uint32_t c = 0; c < C; ++c) {
                conns_.get(c).sender.wait_for_empty_write_slot();
            }
            // Lagged pop: the previous slot is freed only now, so its readout overlapped our waits.
            release_held();
            for (uint32_t c = 0; c < C; ++c) {
                conns_.get(c).sender.send_current_slot_non_blocking(
                    slot + payload_off, bytes, reinterpret_cast<uint32_t>(header(slot, c)));
            }
            on_slot(slot, bytes / A);
            sent += bytes / A;
            held_ = true;
        }
    }

    // Writes a slot's payload to this device at the addresses in its header. Header addresses are in EDM-NOC
    // coordinates, which is our NOC. Non-posted, so finish() can wait for them to land.
    void local_copy(uint32_t slot) {
        for_each_segment(header(slot, 0), [&](uint32_t off, uint64_t addr, uint32_t len) {
            noc_async_write<NOC_MAX_BURST_SIZE>(slot + payload_off + off, addr, len);
        });
    }

    void finish() {
        release_held();
        noc_async_write_barrier();
    }

private:
    static volatile tt_l1_ptr PACKET_HEADER_TYPE* header(uint32_t slot, uint32_t c) {
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(slot + c * hdr_bytes);
    }

    void release_held() {
        if (held_) {
            noc_async_writes_flushed();
            noc_async_posted_writes_flushed();
            cb_pop_front(cb, 1);
            held_ = false;
        }
    }

    tt::tt_fabric::RoutingPlaneConnectionManager& conns_;
    const uint32_t base_;
    uint32_t next_ = 0;
    bool held_ = false;
};
