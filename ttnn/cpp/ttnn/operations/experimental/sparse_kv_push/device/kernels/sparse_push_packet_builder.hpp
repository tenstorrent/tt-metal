// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "sparse_push_constants.hpp"

namespace sparse_push {

// Groups the rows this chip owns for one requester into packets of up to pages_per_packet rows, and hands this
// sender's packets to the writer (kData: the rows; kMeta: a PacketMeta), kBatch at a time behind one read barrier.
// A channel's packets are dealt round robin to its senders: every sender walks all of them in the same order, but
// reads and hands on only those whose turn matches its sender_index.
template <uint32_t page_bytes, uint32_t pages_per_packet, uint32_t banks, uint32_t out_pages, typename InputAccessor>
struct PacketBuilder {
    PacketBuilder(const InputAccessor& input, uint32_t slot_base, uint32_t sender_index, uint32_t senders) :
        input(input), slot_base(slot_base), sender_index(sender_index), senders(senders) {}

    // Starts this chip's run of rows for `requester`. The run occupies the requester's compact buffer from page
    // run_base; first_rank is the rank of the first row pushed (non-zero where a kRest push resumes after kPrefix).
    void begin_run(uint32_t requester_, uint32_t run_base_, uint32_t first_rank) {
        requester = requester_;
        run_base = run_base_;
        packet_index = first_rank / pages_per_packet;
        first_row_in_packet = first_rank % pages_per_packet;
        packet_rows = 0;
        begin_packet();
    }

    // Adds input row `row` (a local row of the pushed cache slot) to the open packet.
    void add_row(uint32_t row) {
        if (my_turn) {
            noc_async_read(input.get_noc_addr(slot_base + row), data_l1 + packet_rows * page_bytes, page_bytes);
        }
        if (first_row_in_packet + ++packet_rows == pages_per_packet) {
            end_packet();
            begin_packet();
        }
    }

    // Ends the run's last, partly filled packet.
    void end_run() {
        if (packet_rows) {
            end_packet();
        }
    }

    // Hands the remaining packets to the writer, then an empty one (0 rows), which ends it.
    void finish() {
        flush();
        cb_reserve_back(kData, 1);
        cb_reserve_back(kMeta, 1);
        reinterpret_cast<volatile tt_l1_ptr PacketMeta*>(get_write_ptr(kMeta))->rows = 0;
        cb_push_back(kData, 1);
        cb_push_back(kMeta, 1);
    }

private:
    // Starts the next packet; if it is this sender's turn, reserves its kData/kMeta slots for the row reads.
    void begin_packet() {
        my_turn = turn == sender_index;
        if (my_turn) {
            cb_reserve_back(kData, batched + 1);
            cb_reserve_back(kMeta, batched + 1);
            data_l1 = get_write_ptr(kData) + batched * pages_per_packet * page_bytes;
        }
    }

    // Ends the open packet: if ours, records where its rows go (kMeta) and batches it; then passes the turn on.
    void end_packet() {
        const uint32_t first_page =
            compact_page<pages_per_packet, banks>(run_base, packet_index * pages_per_packet + first_row_in_packet);
        // The packet's rows land on pages first_page + k * banks; an undersized output drops it.
        const bool fits = first_page + (packet_rows - 1) * banks < out_pages;
        ASSERT(fits);
        if (my_turn && !fits) {
            // Dropped: wait for its reads, which still land in the kData slot the next packet reuses.
            noc_async_read_barrier();
        } else if (my_turn) {
            auto* meta =
                reinterpret_cast<volatile tt_l1_ptr PacketMeta*>(get_write_ptr(kMeta) + batched * kMetaEntryBytes);
            meta->requester = requester;
            meta->first_page = first_page;
            meta->rows = packet_rows;
            if (++batched == kBatch) {
                flush();
            }
        }
        ++packet_index;
        packet_rows = 0;
        first_row_in_packet = 0;
        turn = (turn + 1) % senders;
    }

    // Hands the batched packets to the writer once their reads have landed.
    void flush() {
        if (batched) {
            noc_async_read_barrier();
            cb_push_back(kData, batched);
            cb_push_back(kMeta, batched);
            batched = 0;
        }
    }

    const InputAccessor& input;
    const uint32_t slot_base;  // first input page of the pushed cache slot
    const uint32_t sender_index;
    const uint32_t senders;
    uint32_t turn = 0;     // sender of the next packet; carries over between requesters
    uint32_t batched = 0;  // packets read but not yet handed to the writer
    // The open packet.
    uint32_t requester = 0;
    uint32_t run_base = 0;
    uint32_t packet_index = 0;         // in the run
    uint32_t first_row_in_packet = 0;  // non-zero only where a kRest push resumes mid-packet
    uint32_t packet_rows = 0;
    bool my_turn = false;  // this sender reads its rows and hands it on
    uint32_t data_l1 = 0;
};

}  // namespace sparse_push
