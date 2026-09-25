// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

#ifndef SDPA_DEFERRED_MCAST_PACKET_BYTES
#define SDPA_DEFERRED_MCAST_PACKET_BYTES NOC_MAX_BURST_SIZE
#endif
inline constexpr uint32_t kDeferredMcastPacketBytes = SDPA_DEFERRED_MCAST_PACKET_BYTES;
static_assert(kDeferredMcastPacketBytes > 0 && kDeferredMcastPacketBytes <= NOC_MAX_BURST_SIZE);

// Owned by one reader invocation. Command issuance and ownership of the L1 source are separate:
// issuing VALID does not release the source; only draining the writes does.
struct DeferredKMulticast {
    bool commands_pending = false;
    bool source_in_flight = false;
    uint32_t source_slot_addr = 0;
    uint32_t next_src_addr = 0;
    uint32_t bytes_left = 0;
    uint64_t dst_mcast_addr = 0;
    uint32_t num_dests = 0;
    uint32_t sem_src_addr = 0;
    uint64_t sem_dst_mcast_addr = 0;

    // At most one data packet per read request. The final packet and VALID are issued together.
    void try_issue_packet() { issue_pending_commands<false>(); }
    void issue_all_pending() { issue_pending_commands<true>(); }

    void progress_reads() {
        if (commands_pending) {
            do {
                try_issue_packet();
            } while (!noc_cmd_buf_ready(noc_index, read_cmd_buf));
        }
    }

    void drain(Noc noc) {
        if (source_in_flight) {
            issue_all_pending();
            noc.async_writes_flushed();
            source_in_flight = false;
        }
    }

    void drain_before_reuse(Noc noc, uint32_t addr) {
        if (source_in_flight && addr == source_slot_addr) {
            drain(noc);
        }
    }

private:
    template <bool wait_for_command_buffer>
    void issue_pending_commands() {
        if (!commands_pending) {
            return;
        }
        while (bytes_left > 0 && (wait_for_command_buffer || noc_cmd_buf_ready(noc_index, write_cmd_buf))) {
            const uint32_t size = bytes_left < kDeferredMcastPacketBytes ? bytes_left : kDeferredMcastPacketBytes;
            const bool last_packet = size == bytes_left;
            noc_async_write_multicast_one_packet(next_src_addr, dst_mcast_addr, size, num_dests, last_packet);
            next_src_addr += size;
            dst_mcast_addr += size;
            bytes_left -= size;
            if (last_packet) {
                // Use the data command buffer and VC for VALID too, terminating the linked final
                // packet immediately. No barrier or unrelated write may split this pair.
                noc_async_write_multicast_one_packet(sem_src_addr, sem_dst_mcast_addr, 4, num_dests, false);
                commands_pending = false;
            }
            if constexpr (!wait_for_command_buffer) {
                break;
            }
        }
    }
};
