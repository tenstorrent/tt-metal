// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// routed_combine's row-segment reads, in batches under rotating transaction ids.
//
// A batch waits only for the batch that last held its id, so reads overlap across unit boundaries while
// at most batch_reads * batch_depth are in flight: with every core issuing a whole unit (66 reads) at
// once, the grid's reads congest and a unit's wait grows from 1 to 11 us at 4096 tokens.
//
// A transaction id's barrier polls the NIU's count of that id's requests in flight, which a request
// joins only once the command buffer has sent it, so it can pass before the most recent read is even
// issued. Every read waits for the command buffer before it issues, so a batch with a later read behind
// it is safe to wait for by id; the most recent read is waited for with the full barrier instead.

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

namespace routed_combine {

template <uint32_t batch_reads, uint32_t batch_depth>
struct ReadBatches {
    static_assert(batch_depth >= 2, "a batch is waited for by id only with a later batch behind it");

    // completes[t]: 1 + the unit whose reads end with the batch under id t + 1, or 0.
    uint32_t completes[batch_depth] = {};
    uint32_t issued = 0;
    uint32_t retired = 0;
    // Units, counted from the core's first, whose reads are all in.
    uint32_t done = 0;

    void mark_retired() {
        const uint32_t slot = retired % batch_depth;
        ++retired;
        if (completes[slot] != 0) {
            done = completes[slot];
            completes[slot] = 0;
        }
    }

    // The oldest batch in flight; only with a later batch behind it.
    void retire_next(const Noc& noc) {
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = retired % batch_depth + 1});
        mark_retired();
    }

    // Everything issued, after a full read barrier.
    void retire_all() {
        while (retired < issued) {
            mark_retired();
        }
    }

    // A unit's row segments into slot_address, segment i from row indices[i] at source_offset; on_last(trid)
    // adds reads to the unit's last batch.
    template <typename Accessor, typename OnLast>
    void issue(
        const Noc& noc,
        const Accessor& rows,
        uint32_t slot_address,
        volatile tt_l1_ptr uint32_t* indices,
        uint32_t source_offset,
        uint32_t segment_bytes,
        uint32_t end,
        uint32_t local_unit,
        OnLast&& on_last) {
        for (uint32_t batch_first = 0; batch_first < end; batch_first += batch_reads) {
            while (retired + batch_depth <= issued) {
                retire_next(noc);
            }
            const uint32_t slot = issued % batch_depth;
            const uint32_t batch_end = batch_first + batch_reads < end ? batch_first + batch_reads : end;
            for (uint32_t read = batch_first; read < batch_end; ++read) {
                noc.async_read<NocOptions::TXN_ID>(
                    rows,
                    CoreLocalMem<uint32_t>(slot_address + read * segment_bytes),
                    segment_bytes,
                    {.page_id = indices[read], .offset_bytes = source_offset},
                    {},
                    {.trid = slot + 1});
            }
            const bool last = batch_end == end;
            if (last) {
                on_last(slot + 1);
            }
            completes[slot] = last ? local_unit + 1 : 0;
            ++issued;
        }
    }
};

}  // namespace routed_combine
