// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

////////////////////////////////////////////////////////////////
// Moving chunks, in order.
//
// Nothing here knows which op is running. An op says where a chunk lands, and where a run has to
// stop. This file decides the order, and how much moves at once.
//
// Glossary:
//   chunk           -- the unit we move. min(input page, output page).
//   chunks_per_page -- how many chunks share one page. One number per side, and one of them is 1.
//   bank_step       -- page-id step to the next chunk sitting next to this one in memory.
//                      Interleaved DRAM: the bank count. Comes from the accessor.
//   lane            -- the chunks that share one bank, i.e. one residue class mod bank_step.
//   run             -- chunks next to each other in memory. One NOC command, one packet segment.
//   run_max         -- most chunks one run may take. The host picks it.
//   sweep           -- one pass across every lane: run_max * bank_step chunks. The last is ragged.
//
// Order: take run_max chunks from one lane, then move to the next lane. Runs stay long, and
// consecutive runs land in different banks. Long runs alone are not enough: stepping by bank_step
// and nothing else parks every transfer in one bank, which costs more than the runs win.
// run_max == 1 is plain ascending order, and is where a padded page or a one-chunk packet lands.
////////////////////////////////////////////////////////////////

// Which page holds a chunk, and how far into it the chunk sits.
template <uint32_t chunks_per_page>
FORCE_INLINE uint32_t page_of(uint32_t chunk) {
    return chunk / chunks_per_page;
}

template <uint32_t chunks_per_page, uint32_t chunk_size>
FORCE_INLINE uint32_t byte_off(uint32_t chunk) {
    return (chunk % chunks_per_page) * chunk_size;
}

// A run is one NOC command, so the hardware burst caps it however long the host asked for.
constexpr uint32_t burst_run_max(uint32_t chunk_size) {
    return chunk_size >= NOC_MAX_BURST_SIZE ? 1u : NOC_MAX_BURST_SIZE / chunk_size;
}

// Whether a chunk this big still fits one NOC command. If not, transfers go one chunk at a time.
constexpr bool chunk_fits_command(uint32_t chunk_size) { return chunk_size <= NOC_MAX_BURST_SIZE; }

// The run length actually in force: what the host asked for (0 means "the whole payload"), capped
// by the hardware burst.
constexpr uint32_t run_max_capped(uint32_t asked_chunks, uint32_t payload_chunks, uint32_t chunk_size) {
    const uint32_t want = asked_chunks != 0 ? asked_chunks : payload_chunks;
    const uint32_t burst = burst_run_max(chunk_size);
    return want < burst ? want : burst;
}

////////////////////////////////////////////////////////////////
// Can runs join?
//
// `packed` means the aligned page is exactly tiled by chunks. A padded page can never join: a run
// steps by the aligned page size while L1 is packed, so the pad would land in the payload.
////////////////////////////////////////////////////////////////

// Join chunks inside one page. Only when a page holds several of them and they are consecutive.
FORCE_INLINE bool join_in_page(bool packed, uint32_t chunks_per_page, uint32_t bank_step) {
    return packed && chunks_per_page > 1 && bank_step == 1;
}

// Carry a run on into the next page.
FORCE_INLINE bool join_pages(bool packed, uint32_t chunks_per_page, uint32_t page_stride, uint32_t bank_step) {
    if (!packed) {
        return false;
    }
    return chunks_per_page > 1 ? (join_in_page(packed, chunks_per_page, bank_step) && page_stride == 1)
                               : (page_stride == bank_step);
}

// The step to the next chunk sitting next to this one in memory. Chunks sharing a page are one
// apart; otherwise it is the accessor's own page step.
template <typename Accessor>
FORCE_INLINE uint32_t bank_step_of(const Accessor& acc, uint32_t chunks_per_page) {
    return chunks_per_page > 1 ? 1u : acc.contiguous_page_stride();
}

// Whether the aligned page is exactly tiled by chunks. Reader and writer both ask this, and both
// have to get the same answer, so there is one definition of it.
template <typename Accessor>
FORCE_INLINE bool packed_pages(const Accessor& acc, uint32_t chunks_per_page, uint32_t chunk_size) {
    return acc.get_aligned_page_size() == chunks_per_page * chunk_size;
}

// How many chunks the run starting at `chunk` may take. Capped by three things: memory contiguity,
// `end_chunk` (the op's boundary -- past it the chunks are not ours), and `limit` (what the caller
// still wants, and what is left in this lane).
template <uint32_t chunks_per_page, typename Accessor>
FORCE_INLINE uint32_t
run_length(const Accessor& acc, bool in_page, bool across_pages, uint32_t chunk, uint32_t end_chunk, uint32_t limit) {
    // At one chunk the contiguity query cannot help, and on a sharded tensor it loops over rank.
    if (limit <= 1) {
        return limit;
    }
    uint32_t run;
    if constexpr (chunks_per_page == 1) {
        run = across_pages ? acc.num_contiguous_pages(chunk, end_chunk) : 1u;
    } else {
        run = in_page ? chunks_per_page - chunk % chunks_per_page : 1u;
        if (across_pages) {
            const uint32_t end_page = (end_chunk + chunks_per_page - 1) / chunks_per_page;
            run += (acc.num_contiguous_pages(chunk / chunks_per_page, end_page) - 1) * chunks_per_page;
        }
        // end_page rounds up, so clip in chunks too. bank_step is 1 whenever a page holds several
        // chunks, so chunks left and steps left are the same number here.
        const uint32_t room = end_chunk - chunk;
        run = run < room ? run : room;
    }
    return run < limit ? run : limit;
}

////////////////////////////////////////////////////////////////
// The walk
//
// Holds a position. The only reason it holds more than that is the last sweep, which is ragged:
// its lanes are `tail_short_` or `tail_short_ + 1` chunks tall.
////////////////////////////////////////////////////////////////
class Walk {
public:
    // Walk [first, first + count) chunks, starting `skip` chunks into the order.
    FORCE_INLINE void init(uint32_t first, uint32_t count, uint32_t skip, uint32_t bank_step, uint32_t run_max) {
        bank_step_ = bank_step;
        run_max_ = run_max;
        sweep_ = run_max * bank_step;
        const uint32_t full = count / sweep_;
        const uint32_t rem = count - full * sweep_;
        tail_first_ = first + full * sweep_;
        tail_short_ = rem / bank_step;
        tail_tall_ = rem - tail_short_ * bank_step;
        lane_ = 0;
        k_ = 0;
        if (count == 0) {
            sweep_first_ = c_ = first;
            return;
        }
        if (skip < full * sweep_) {
            const uint32_t sweep = skip / sweep_;
            const uint32_t into = skip - sweep * sweep_;
            sweep_first_ = first + sweep * sweep_;
            lane_ = into / run_max;
            k_ = into - lane_ * run_max;
        } else {
            // In the tail the first tail_tall_ lanes are one chunk taller, so the chunks before
            // lane r are r * tail_short_ + min(r, tail_tall_).
            sweep_first_ = tail_first_;
            const uint32_t into = skip - full * sweep_;
            const uint32_t tall = tail_tall_ * (tail_short_ + 1);
            if (into < tall) {
                lane_ = into / (tail_short_ + 1);
                k_ = into - lane_ * (tail_short_ + 1);
            } else {
                // tail_short_ > 0 here: at tail_short_ == 0 the sweep holds only its taller lanes.
                const uint32_t rest = into - tall;
                lane_ = tail_tall_ + rest / tail_short_;
                k_ = rest - (lane_ - tail_tall_) * tail_short_;
            }
        }
        c_ = sweep_first_ + lane_ + k_ * bank_step_;
    }

    FORCE_INLINE uint32_t chunk() const { return c_; }

    // Chunks left in this lane, i.e. the longest run still allowed here.
    FORCE_INLINE uint32_t lane_room() const { return lane_chunks() - k_; }

    FORCE_INLINE void advance(uint32_t n) {
        ASSERT(n != 0 && n <= lane_room());
        k_ += n;
        if (k_ < lane_chunks()) {
            c_ += n * bank_step_;
            return;
        }
        k_ = 0;
        if (++lane_ == lanes()) {
            lane_ = 0;
            sweep_first_ += sweep_;
        }
        c_ = sweep_first_ + lane_;
    }

private:
    FORCE_INLINE bool in_tail() const { return sweep_first_ == tail_first_; }
    FORCE_INLINE uint32_t lane_chunks() const {
        return in_tail() ? tail_short_ + (lane_ < tail_tall_ ? 1u : 0u) : run_max_;
    }
    // Empty lanes can only be a tail of lanes, so a lane step never has to skip one.
    FORCE_INLINE uint32_t lanes() const { return (in_tail() && tail_short_ == 0) ? tail_tall_ : bank_step_; }

    uint32_t c_, bank_step_, run_max_, sweep_, lane_, k_, sweep_first_, tail_first_, tail_short_, tail_tall_;
};

// Debug only: a run has to be one linear stretch of memory, which is what one transfer assumes.
// `walk` is a copy, so probing it does not move the caller's.
template <typename AddrFn>
FORCE_INLINE bool run_is_linear(Walk walk, uint32_t run, uint32_t chunk_size, uint64_t first, AddrFn addr) {
    for (uint32_t i = 0; i < run; ++i) {
        if (addr(walk.chunk()) != first + i * chunk_size) {
            return false;
        }
        walk.advance(1);
    }
    return true;
}
