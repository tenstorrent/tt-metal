// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

////////////////////////////////////////////////////////////////
// Page walk, shared by both all_gather algorithms.
//
// Glossary:
//   chunk    -- the transfer unit, min(input page, output page).
//   chunk id -- a chunk's index in this device's contribution.
//   global   -- a chunk's index in the output tensor. Rows are strided: between them the output
//               holds the other devices' stripes.
//   seqno    -- a chunk's position in the emission order.
//   stride   -- chunk step between neighbours in memory, from TensorAccessor.
//   lane     -- residue class mod stride, i.e. one line of chunks contiguous in memory.
//   xfer     -- chunks per transfer: the most that fits a packet, one NOC command, and the
//               host's run cap.
//   tile     -- xfer * stride chunks, read column-major.
//   run      -- one tile column: chunks contiguous at the destination, sent as one transfer.
//   stripe   -- the chunks this device contributes per row of the output.
//
// A run is chunks contiguous in memory, sent as one transfer. Where they are comes from
// TensorAccessor, so no layout is special-cased here or on the host: `stride` is the accessor's own
// chunk step between neighbours, and `xfer` is how many chunks fit one transfer.
//
// Long runs alone are not enough. Stepping by `stride` parks a worker in one DRAM bank, which costs
// more bandwidth than the runs win, so the walk is a tiled transpose: `xfer` chunks from one lane,
// then the next lane. Every fallback then falls out of `xfer` -- a padded page or a page-sized
// packet gives xfer == 1, which is plain ascending order.
////////////////////////////////////////////////////////////////

// Chunks in one transfer: the most that fits a packet, a single NOC command, and the host's run
// cap (0 = no cap). A cap trades packet fill for DRAM bank spread. Hardware already bounds a
// transfer at min(fabric packet, NOC_MAX_BURST_SIZE) -- 7616 B on Wormhole, 15232 B on Blackhole --
// so a host cap only bites below that figure.
constexpr uint32_t chunks_per_transfer(uint32_t packet_size, uint32_t chunk_size, uint32_t run_cap_bytes = 0) {
    uint32_t cap = packet_size < NOC_MAX_BURST_SIZE ? packet_size : NOC_MAX_BURST_SIZE;
    if (run_cap_bytes != 0 && run_cap_bytes < cap) {
        cap = run_cap_bytes;
    }
    return cap < chunk_size ? 1u : cap / chunk_size;
}

// Whether a tensor's contiguity lines up with the walk, so a run can hold more than one chunk.
struct RunSource {
    bool sub_page;    // chunks inside one page are consecutive seqnos
    bool cross_page;  // a run may carry on into the next page
};

// `packed` has to be false for a padded page: a run steps by the aligned page size while the CB is
// packed, so the pad would land in the payload.
FORCE_INLINE RunSource run_source(bool packed, uint32_t chunks_per_page, uint32_t page_stride, uint32_t stride) {
    if (!packed) {
        return {false, false};
    }
    const bool sub_page = chunks_per_page > 1 && stride == 1;
    return {sub_page, chunks_per_page > 1 ? (sub_page && page_stride == 1) : (page_stride == stride)};
}

// Chunks contiguous in memory from `chunk` on, stepping the way the walk steps, capped at
// `end_chunk`. Can reach past the current run, so the caller has to clip it with run_limit().
template <uint32_t chunks_per_page, typename Accessor>
FORCE_INLINE uint32_t contiguous_chunks(const Accessor& acc, RunSource src, uint32_t chunk, uint32_t end_chunk) {
    if constexpr (chunks_per_page == 1) {
        return src.cross_page ? acc.num_contiguous_pages(chunk, end_chunk) : 1u;
    } else {
        uint32_t n = src.sub_page ? chunks_per_page - chunk % chunks_per_page : 1u;
        if (src.cross_page) {
            const uint32_t end_page = (end_chunk + chunks_per_page - 1) / chunks_per_page;
            n += (acc.num_contiguous_pages(chunk / chunks_per_page, end_page) - 1) * chunks_per_page;
        }
        // end_page rounds up, so clip in chunks too. stride is 1 whenever a page holds several
        // chunks, so chunks left and seqnos left are the same number here.
        const uint32_t room = end_chunk - chunk;
        return n < room ? n : room;
    }
}

// The walk order plus the output's runs. Reader and writer build this the same way, so their walks
// cannot diverge.
// Run grouping can still differ per side; that is safe only because the CB is packed -- chunk k
// sits at k * chunk_size.
struct WalkPlan {
    uint32_t stride;
    uint32_t xfer;
    RunSource out;
};

template <uint32_t chunks_per_page, uint32_t chunk_size, uint32_t xfer_max, typename Accessor>
FORCE_INLINE WalkPlan walk_plan(const Accessor& out) {
    const uint32_t page_stride = out.contiguous_page_stride();
    // Concat packs chunks inside one output page, so there neighbours are one chunk apart.
    const uint32_t stride = chunks_per_page > 1 ? 1u : page_stride;
    // `packed`: the output's aligned page is exactly tiled by chunks. Perf-only for multicast (no
    // run merging when false), but a correctness precondition for unicast, which relays by reading
    // its own output back into the packed CB. The host enforces it, so unicast always sees true.
    const bool packed = out.get_aligned_page_size() == chunks_per_page * chunk_size;
    return {stride, packed ? xfer_max : 1u, run_source(packed, chunks_per_page, page_stride, stride)};
}

// Walks a chunk id range as tiles of `xfer * stride`, reading each tile column-major: `xfer` chunks
// `stride` apart, then the next lane. One column is one run. `xfer == 1` is plain ascending order.
//
// The last tile is ragged -- its columns are `hb_` or `hb_ + 1` tall -- which is the only reason this
// holds state beyond the position.
class TiledWalk {
public:
    // Walk [first, first + count) from seqno `skip`. `stride` and `xfer` are at least 1.
    FORCE_INLINE void init(uint32_t first, uint32_t count, uint32_t skip, uint32_t stride, uint32_t xfer) {
        stride_ = stride;
        xfer_ = xfer;
        tile_ = xfer * stride;
        const uint32_t full = count / tile_;
        const uint32_t rem = count - full * tile_;
        tail_first_ = first + full * tile_;
        hb_ = rem / stride;
        he_ = rem - hb_ * stride;
        lane_ = 0;
        k_ = 0;
        if (count == 0) {
            tile_first_ = c_ = first;
            return;
        }
        if (skip < full * tile_) {
            const uint32_t tile = skip / tile_;
            const uint32_t in_tile = skip - tile * tile_;
            tile_first_ = first + tile * tile_;
            lane_ = in_tile / xfer;
            k_ = in_tile - lane_ * xfer;
        } else {
            // Lanes below he_ hold one extra chunk, so seqnos before lane r are r*hb_ + min(r, he_).
            tile_first_ = tail_first_;
            const uint32_t s = skip - full * tile_;
            const uint32_t tall = he_ * (hb_ + 1);
            if (s < tall) {
                lane_ = s / (hb_ + 1);
                k_ = s - lane_ * (hb_ + 1);
            } else {
                // hb_ > 0 here: at hb_ == 0 the tile holds only its taller lanes, so `s < tall`.
                const uint32_t rest = s - tall;
                lane_ = he_ + rest / hb_;
                k_ = rest - (lane_ - he_) * hb_;
            }
        }
        c_ = tile_first_ + lane_ + k_ * stride_;
    }

    FORCE_INLINE uint32_t chunk() const { return c_; }

    // Chunks left in this column, i.e. the longest run still allowed here.
    FORCE_INLINE uint32_t run_limit() const { return col() - k_; }

    FORCE_INLINE void advance(uint32_t n) {
        ASSERT(n != 0 && n <= run_limit());
        k_ += n;
        if (k_ < col()) {
            c_ += n * stride_;
            return;
        }
        k_ = 0;
        if (++lane_ == lanes()) {
            lane_ = 0;
            tile_first_ += tile_;
        }
        c_ = tile_first_ + lane_;
    }

private:
    FORCE_INLINE bool in_tail() const { return tile_first_ == tail_first_; }
    FORCE_INLINE uint32_t col() const { return in_tail() ? hb_ + (lane_ < he_ ? 1u : 0u) : xfer_; }
    // Empty columns can only be a tail of lanes, so a lane step never has to skip one.
    FORCE_INLINE uint32_t lanes() const { return (in_tail() && hb_ == 0) ? he_ : stride_; }

    uint32_t c_, stride_, xfer_, tile_, lane_, k_, tile_first_, tail_first_, hb_, he_;
};

// Longest run allowed at the walk's current position, given `left` chunks still wanted. At one chunk
// the query is skipped: it cannot help, and on a sharded tensor it loops over rank.
template <uint32_t chunks_per_page, typename Accessor>
FORCE_INLINE uint32_t
next_run(const TiledWalk& walk, const Accessor& acc, RunSource src, uint32_t chunk, uint32_t end_chunk, uint32_t left) {
    const uint32_t room = walk.run_limit();
    const uint32_t limit = room < left ? room : left;
    if (limit <= 1) {
        return limit;
    }
    const uint32_t run = contiguous_chunks<chunks_per_page>(acc, src, chunk, end_chunk);
    return run < limit ? run : limit;
}

// Debug only: a run has to be one linear stretch of memory, which is what one transfer assumes.
// `walk` is a copy, so probing it does not move the caller's.
template <typename AddrFn>
FORCE_INLINE bool run_is_linear(TiledWalk walk, uint32_t run, uint32_t chunk_size, uint64_t first, AddrFn addr) {
    for (uint32_t i = 0; i < run; ++i) {
        if (addr(walk.chunk()) != first + i * chunk_size) {
            return false;
        }
        walk.advance(1);
    }
    return true;
}

// A chunk's index in this device's contribution -> its index in the output tensor. Between rows the
// output holds the other devices' stripes, so a run stops at the row edge.
template <uint32_t chunks_per_stripe, uint32_t num_devices>
class StripeMap {
public:
    struct Pos {
        uint32_t global;   // chunk index in the output tensor
        uint32_t row_end;  // one past this row's last chunk of our stripe
    };

    FORCE_INLINE void init(uint32_t stripe) { offset_ = stripe * chunks_per_stripe; }

    FORCE_INLINE Pos at(uint32_t local) const {
        const uint32_t row = local / chunks_per_stripe;
        const uint32_t base = row * (num_devices * chunks_per_stripe) + offset_;
        return {base + local - row * chunks_per_stripe, base + chunks_per_stripe};
    }

private:
    uint32_t offset_;
};

// A chunk index split into the page holding it and the byte offset inside that page.
template <uint32_t chunks_per_page>
FORCE_INLINE uint32_t page_of(uint32_t chunk) {
    return chunk / chunks_per_page;
}

template <uint32_t chunks_per_page, uint32_t chunk_size>
FORCE_INLINE uint32_t byte_off_of(uint32_t chunk) {
    return (chunk % chunks_per_page) * chunk_size;
}
