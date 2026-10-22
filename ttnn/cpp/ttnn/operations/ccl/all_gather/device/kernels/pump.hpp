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

// Non-posted writes issued so far, and whether the NOC has sent (read out of L1) everything up to such a mark.
// Unlike noc_async_writes_flushed(), this waits for one write, not for every write issued after it.
FORCE_INLINE uint32_t writes_issued() { return noc_nonposted_writes_num_issued[noc_index]; }
FORCE_INLINE bool writes_sent(uint32_t mark) {
    return static_cast<int32_t>(NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT) - mark) >= 0;
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
        while (!writes_sent(mark_)) {
        }
        hdr_->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(x, y, addr, 0), val, true});
        edm_->wait_for_empty_write_slot();
        edm_->send_payload_flush_non_blocking_from_address(
            reinterpret_cast<uint32_t>(hdr_), sizeof(PACKET_HEADER_TYPE));
        mark_ = writes_issued();
    }

private:
    tt::tt_fabric::WorkerToFabricEdmSender* edm_ = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_ = nullptr;
    uint32_t mark_ = writes_issued();
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
    static_assert(sizeof(PACKET_HEADER_TYPE) == hdr_bytes, "host and kernel disagree on the header size");
    static_assert(C >= 1 && C <= tt::tt_fabric::RoutingPlaneConnectionManager::MaxConnections);

    // The builder runs on the core's other NOC and reads slot i with trid 1 + i.
    static constexpr uint32_t kBuilderNoc = 1 - noc_index;

public:
    explicit Pump(tt::tt_fabric::RoutingPlaneConnectionManager& conns) : conns_(conns), base_(get_read_ptr(cb)) {}

    // Sends slots until `total` units have gone. on_slot(slot, units) runs once a slot is issued; on_idle() runs
    // before blocking on the builder. A sent slot is popped once its writes (on_slot's included) have left L1,
    // without waiting for the writes issued after it, so several sends can be in flight.
    template <class OnSlot, class OnIdle>
    FORCE_INLINE void run(uint32_t total, OnSlot&& on_slot, OnIdle&& on_idle) {
        for (uint32_t sent = 0; sent < total;) {
            if (!cb_pages_available_at_front(cb, held_ + 1)) {
                release_sent();
                on_idle();
                cb_wait_front(cb, held_ + 1);
            }
            // The builder pushes a slot with its reads in flight (slot_io.hpp): wait for them on its NOC.
            while (NOC_STATUS_READ_REG(kBuilderNoc, NIU_MST_REQS_OUTSTANDING_ID(1 + next_)) != 0) {
            }
            // The CB handshake does not fence: without this a CPU read can see the slot's previous use.
            invalidate_l1_cache();
            const uint32_t slot = base_ + next_ * slot_bytes;
            const uint32_t bytes = header(slot, C - 1)->payload_size_bytes;
            for (uint32_t c = 0; c < C; ++c) {
                conns_.get(c).sender.wait_for_empty_write_slot();
            }
            for (uint32_t c = 0; c + 1 < C; ++c) {
                conns_.get(c).sender.send_current_slot_non_blocking(
                    slot + payload_off, bytes, reinterpret_cast<uint32_t>(header(slot, c)));
            }
            // The last header sits right before the payload: one write carries the whole packet.
            conns_.get(C - 1).sender.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(header(slot, C - 1)), hdr_bytes + bytes);
            on_slot(slot, bytes / A);
            mark_[next_] = writes_issued();
            next_ = next_ + 1 == slots ? 0 : next_ + 1;
            ++held_;
            sent += bytes / A;
            release_sent();
        }
    }

    // Writes a landed slot's payload to this device at the addresses in its header (converted back from the
    // EDM's NOC to ours).
    FORCE_INLINE void local_copy(uint32_t slot) {
        for_each_segment(header(slot, C - 1), [&](uint32_t off, uint64_t addr, uint32_t len) {
            noc_async_write<NOC_MAX_BURST_SIZE>(slot + payload_off + off, edm_noc_addr(addr), len);
        });
    }

    void finish() {
        noc_async_writes_flushed();
        release_sent();
        noc_async_write_barrier();
    }

private:
    // Header c sits (C - c) headers before the payload (see route.hpp).
    static volatile tt_l1_ptr PACKET_HEADER_TYPE* header(uint32_t slot, uint32_t c) {
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            slot + slot_header_offset(c, C, hdr_bytes, payload_off));
    }

    // Pops, oldest first, every held slot whose writes have left L1: the builder may refill it.
    FORCE_INLINE void release_sent() {
        while (held_ != 0 && writes_sent(mark_[oldest_])) {
            cb_pop_front(cb, 1);
            oldest_ = oldest_ + 1 == slots ? 0 : oldest_ + 1;
            --held_;
        }
    }

    tt::tt_fabric::RoutingPlaneConnectionManager& conns_;
    const uint32_t base_;
    uint32_t next_ = 0, oldest_ = 0, held_ = 0;
    uint32_t mark_[slots];
};
