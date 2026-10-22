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

// Write trids on the pump's NOC: slot i's writes (its sends and local copy) use 1 + i, so a slot is free once its
// own trid has nothing left to send; semaphore-inc headers use kIncTrid. The trid is sticky on the write command
// buffer: whoever sets one puts 0 back when done (a kernel must exit with 0).
inline constexpr uint32_t kIncTrid = 0xF;

// Whether every write issued under `trid` has left L1. A write counts under its trid once the command buffer has
// taken it; every later write or trid change waits for that, so only the newest write can be uncounted.
FORCE_INLINE bool trid_sent(uint32_t trid) {
    return ncrisc_noc_nonposted_write_with_transaction_id_sent(noc_index, trid);
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
        // An earlier inc may still be reading this header (set_trid(0) made it count).
        while (!trid_sent(kIncTrid)) {
        }
        hdr_->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(x, y, addr, 0), val, true});
        edm_->wait_for_empty_write_slot();
        noc_async_write_set_trid(kIncTrid);
        edm_->send_payload_flush_non_blocking_from_address(
            reinterpret_cast<uint32_t>(hdr_), sizeof(PACKET_HEADER_TYPE));
        noc_async_write_set_trid(0);
    }

private:
    tt::tt_fabric::WorkerToFabricEdmSender* edm_ = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_ = nullptr;
};

// Ct: SlotCt (route.hpp).
template <class Ct>
class Pump {
    static constexpr uint32_t C = Ct::C;
    static constexpr uint32_t slots = Ct::slots;
    static_assert(C >= 1 && C <= tt::tt_fabric::RoutingPlaneConnectionManager::MaxConnections);
    static_assert(slots < kIncTrid, "slot trids 1..slots must not reach the inc trid");

    // The builder runs on the core's other NOC and reads slot i with trid 1 + i.
    static constexpr uint32_t kBuilderNoc = 1 - noc_index;

public:
    explicit Pump(tt::tt_fabric::RoutingPlaneConnectionManager& conns) : conns_(conns), base_(get_read_ptr(Ct::cb)) {}

    // Sends slots until `total` units have gone. on_slot(slot, units) runs once a slot is issued; on_idle() runs
    // before blocking on the builder. A sent slot is popped once its own writes (local_copy's included) have left
    // L1, so several sends can be in flight.
    template <class OnSlot, class OnIdle>
    FORCE_INLINE void run(uint32_t total, OnSlot&& on_slot, OnIdle&& on_idle) {
        for (uint32_t sent = 0; sent < total;) {
            if (!cb_pages_available_at_front(Ct::cb, held_ + 1)) {
                release_sent();
                on_idle();
                // Keep releasing: the builder may be waiting for a slot we hold.
                while (!cb_pages_available_at_front(Ct::cb, held_ + 1)) {
                    release_sent();
                }
            }
            // The builder pushes a slot with its reads in flight (slot_ring.hpp): wait for them on its NOC.
            while (NOC_STATUS_READ_REG(kBuilderNoc, NIU_MST_REQS_OUTSTANDING_ID(1 + next_)) != 0) {
            }
            // The CB handshake does not fence: without this a CPU read can see the slot's previous use.
            invalidate_l1_cache();
            const uint32_t slot = base_ + next_ * Ct::slot_bytes;
            const uint32_t bytes = header(slot, C - 1)->payload_size_bytes;
            for (uint32_t c = 0; c < C; ++c) {
                conns_.get(c).sender.wait_for_empty_write_slot();
            }
            noc_async_write_set_trid(1 + next_);
            for (uint32_t c = 0; c + 1 < C; ++c) {
                conns_.get(c).sender.send_current_slot_non_blocking(
                    slot + Ct::payload_off, bytes, reinterpret_cast<uint32_t>(header(slot, c)));
            }
            // The last header sits right before the payload: one write carries the whole packet.
            conns_.get(C - 1).sender.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(header(slot, C - 1)), Ct::hdr_bytes + bytes);
            on_slot(slot, bytes / Ct::A);
            next_ = next_ + 1 == slots ? 0 : next_ + 1;
            ++held_;
            sent += bytes / Ct::A;
            release_sent();
        }
        noc_async_write_set_trid(0);
    }

    // From on_slot: writes the slot's payload to this device at the addresses in its header (converted back from
    // the EDM's NOC to ours), under the slot's trid.
    FORCE_INLINE void local_copy(uint32_t slot) {
        noc_async_write_set_trid(1 + next_);
        for_each_segment(header(slot, C - 1), [&](uint32_t off, uint64_t addr, uint32_t len) {
            noc_async_write<NOC_MAX_BURST_SIZE>(slot + Ct::payload_off + off, edm_noc_addr(addr), len);
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
            slot + slot_header_offset(c, C, Ct::hdr_bytes, Ct::payload_off));
    }

    // Pops, oldest first, every held slot whose writes have left L1: the builder may refill it. The newest slot's
    // last write is counted only once the command buffer is ready.
    FORCE_INLINE void release_sent() {
        while (held_ != 0 && trid_sent(1 + oldest_) && (held_ > 1 || noc_cmd_buf_ready(noc_index, write_cmd_buf))) {
            cb_pop_front(Ct::cb, 1);
            oldest_ = oldest_ + 1 == slots ? 0 : oldest_ + 1;
            --held_;
        }
    }

    tt::tt_fabric::RoutingPlaneConnectionManager& conns_;
    const uint32_t base_;
    uint32_t next_ = 0, oldest_ = 0, held_ = 0;
};
