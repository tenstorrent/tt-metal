// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"

namespace sparse_push {

// Groups the rows this chip owns for one requester into packets and hands this sender's packets to the writer (kData:
// the rows; kMeta: a PacketMeta). The run's row of rank r goes to compact page run_base + r (see compact_page), so a
// unit of pages_per_packet * banks ranks spreads over every bank, and packet s of a unit takes the unit's ranks
// start + s, start + s + banks, ...: one bank's pages, at consecutive addresses there. A channel's units are dealt round
// robin to its senders: every sender walks all rows in the same order, but reads only its own units. A unit goes to the
// writer behind one read barrier as banks slots, packet s in slot s (kSkipRows where a partial unit has no packet s),
// so the slots stay aligned in the CBs.
template <uint32_t page_bytes, uint32_t pages_per_packet, uint32_t banks, uint32_t out_pages, typename InputAccessor>
struct PacketBuilder {
    static_assert(banks % kBatch == 0 && banks <= 32, "a unit's slots must fill whole writer batches");
    static constexpr uint32_t kUnit = pages_per_packet * banks;
    static constexpr uint32_t kPacketBytes = pages_per_packet * page_bytes;

    PacketBuilder(const InputAccessor& input, uint32_t slot_base, uint32_t sender_index, uint32_t senders) :
        input(input), slot_base(slot_base), sender_index(sender_index), senders(senders) {}

    // Starts this chip's run of rows for `requester`. The run occupies the requester's compact buffer from page
    // run_base; first_rank is the rank of the first row pushed (non-zero where a kRest push resumes after kPrefix).
    void begin_run(uint32_t requester_, uint32_t run_base_, uint32_t first_rank) {
        requester = requester_;
        run_base = run_base_;
        rank = first_rank;
        begin_unit();
    }

    // Adds input row `row` (a local row of the pushed cache slot), of the next rank, to the open unit.
    void add_row(uint32_t row) {
        const uint32_t offset = rank - unit_start;
        const uint32_t packet = offset % banks;
        if (mine) {
            noc_async_read(
                input.get_noc_addr(slot_base + row),
                data_l1 + packet * kPacketBytes + offset / banks * page_bytes,
                page_bytes);
        }
        if (++rank % kUnit == 0) {
            end_unit();
            begin_unit();
        }
    }

    // Ends the run's last, partly filled unit.
    void end_run() {
        if (rank != unit_start) {
            end_unit();
        }
    }

    // Hands the remaining packets to the writer, then an empty one (0 rows), which ends it.
    void finish() {
        cb_reserve_back(kData, 1);
        cb_reserve_back(kMeta, 1);
        reinterpret_cast<volatile tt_l1_ptr PacketMeta*>(get_write_ptr(kMeta))->rows = 0;
        cb_push_back(kData, 1);
        cb_push_back(kMeta, 1);
    }

private:
    // Opens the unit holding `rank`; if it is this sender's, reserves its slots.
    void begin_unit() {
        unit_start = rank;
        mine = turn == sender_index;
        if (mine) {
            cb_reserve_back(kData, banks);
            cb_reserve_back(kMeta, banks);
            data_l1 = get_write_ptr(kData);
        }
    }

    // Ends the open unit: if it is this sender's, records where its packets go (kMeta) and hands it on once its reads
    // have landed. Then passes the turn on.
    void end_unit() {
        if (mine) {
            const uint32_t rows = rank - unit_start;
            bool fits = true;
            for (uint32_t packet = 0; packet < banks; ++packet) {
                auto* meta =
                    reinterpret_cast<volatile tt_l1_ptr PacketMeta*>(get_write_ptr(kMeta) + packet * kMetaEntryBytes);
                if (packet >= rows) {
                    meta->rows = kSkipRows;
                    continue;
                }
                const uint32_t packet_rows = (rows - packet + banks - 1) / banks;
                const uint32_t first_page = compact_page(run_base, unit_start + packet);
                // The packet's rows land on pages first_page + k * banks; an undersized output drops the unit.
                fits &= first_page + (packet_rows - 1) * banks < out_pages;
                meta->requester = requester;
                meta->first_page = first_page;
                meta->rows = packet_rows;
            }
            ASSERT(fits);
            noc_async_read_barrier();  // the reads land in the reserved slots even when the unit is dropped
            if (fits) {
                cb_push_back(kData, banks);
                cb_push_back(kMeta, banks);
            }
        }
        turn = turn + 1 == senders ? 0 : turn + 1;
    }

    const InputAccessor& input;
    const uint32_t slot_base;  // first input page of the pushed cache slot
    const uint32_t sender_index;
    const uint32_t senders;
    uint32_t turn = 0;  // sender of the open unit; carries over between requesters
    // The open unit.
    uint32_t requester = 0;
    uint32_t run_base = 0;
    uint32_t rank = 0;        // rank of the next row in the run
    uint32_t unit_start = 0;  // rank of the unit's first row (mid-unit where a kRest push resumes)
    bool mine = false;        // the unit is this sender's
    uint32_t data_l1 = 0;
};

}  // namespace sparse_push
