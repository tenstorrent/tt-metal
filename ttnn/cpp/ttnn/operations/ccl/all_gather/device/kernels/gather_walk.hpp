// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "api/dataflow/dataflow_api.h"
#elif !defined(FORCE_INLINE) || !defined(ASSERT)
// Host build (unit tests): include tests/ttnn/unit_tests/gtests/accessor/host_kernel_shim.hpp first (with
// HOST_KERNEL_SHIM_KEEP_MACROS). No fallbacks here, so the shim's throwing ASSERT can never be shadowed by a
// no-op one that an earlier include defined.
#error "host build: define FORCE_INLINE and ASSERT (include host_kernel_shim.hpp) before this header"
#endif

#include <cstdint>

#include "gather_rules.hpp"

////////////////////////////////////////////////////////////////
// The page walk, shared by both all_gather algorithms. The integer rules (k, xfer, slot, stride, run sources,
// seek) live in gather_rules.hpp, which the host compiles too; everything here is per-chunk kernel code.
//
// Glossary:
//   walk page-- the k = 1 transfer unit, a whole number of which make up each accessor page:
//                 Matched / Concat: the input's *aligned* page (an output page holds out_cpp / k of them;
//                                   Matched: 1, Concat: the concat ratio);
//                 Split:            the output's *content* page (an input page holds in_cpp / k of them).
//   k        -- chunks per walk page, a power of 2 chosen by the host (choose_k; 1 = the walk page itself).
//   chunk    -- the transfer unit: a walk page / k. Chunk id c sits in walk page c / k at sub-chunk c % k, so
//               in_cpp = split ratio * k and out_cpp = concat ratio * k.
//   slot     -- a chunk's CB slot: chunk bytes, or (one chunk per output page, k = 1) the output's padded aligned
//               page, so L1 and the output step alike and output runs can cross pages.
//   chunk id -- a chunk's index in this device's contribution.
//   global   -- a chunk's index in the output tensor. Rows are strided: between them the output
//               holds the other devices' stripes. Rows are whole walk pages.
//   seqno    -- a chunk's position in the emission order.
//   stride   -- walk-page step between neighbours in memory (walk_stride_for: the output accessor's
//               contiguous_page_stride, or 1 where several walk pages share an output page). CT in kernels.
//   lane     -- the walk pages first_page + l + m * stride, m = 0, 1, ... At stride == an interleaved accessor's
//               bank count a lane stays in one bank of *that accessor's* page space: the input's, or the output's
//               only while a stripe row is a multiple of the stride (StripeMap remaps rows). A lane's chunk stream
//               at position q is walk page first_page + l + (q / k) * stride, sub-chunk q % k: page-major,
//               sub-chunk-minor.
//   xfer     -- chunks per column: what fits a packet, one NOC command and the host's run cap (max_xfer), or 1
//               where the output is not packed. Host-chosen, CT in kernels.
//   tile     -- positions [t * xfer, (t + 1) * xfer) of every lane, read column-major (lane after lane).
//   column   -- one lane's part of a tile: xfer chunks, the most one transfer may carry.
//   run      -- consecutive chunks of one column that are also contiguous in memory: one NOC command
//               / one scatter segment. next_run() is the only place that forms runs.
//   stripe   -- the chunks this device contributes per row of the output.
//   entry    -- one CB page: chunks_per_entry slots (cb_entry_bytes), the reader/writer handshake unit.
//
// The walk is the full tile grid clipped at the slice end: a column stops at `end`, an empty lane ends
// the tile. Lane lengths only fall with the lane index, so empty lanes are always a suffix and no tail
// state is needed. At k == 1 the order is the plain column-major tile order (the host gtest's
// LegacyTiledWalk oracle checks it).
//
// The kernel never seeks: the host computes a WalkPos with walk_seek() for any resume point.
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
// Compile-time and launch checks shared by the four kernels
////////////////////////////////////////////////////////////////

// The walk's CT invariants, once for all four kernels. `in_cpp` is the input's chunks per page (a writer, which
// has no input, passes k); `packet_size` 0 means the kernel sends no packets (the unicast reader).
template <
    uint32_t k,
    uint32_t in_cpp,
    uint32_t out_cpp,
    uint32_t cps,
    uint32_t chunk_size,
    uint32_t slot_bytes,
    uint32_t cb_entry_bytes,
    uint32_t xfer,
    uint32_t packet_size>
FORCE_INLINE constexpr void check_walk_ct() {
    static_assert(k != 0 && (k & (k - 1)) == 0, "k is a power of 2");
    static_assert(in_cpp % k == 0 && out_cpp % k == 0 && cps % k == 0, "pages and stripes are whole walk pages");
    static_assert(chunk_size <= slot_bytes && cb_entry_bytes % slot_bytes == 0, "a CB entry is whole slots");
    static_assert(slot_is_valid(chunk_size, slot_bytes, out_cpp, k), "a padded slot is one chunk per page, k = 1");
    static_assert(
        chunk_size % gather_noc_write_alignment == 0 && slot_bytes % gather_noc_write_alignment == 0,
        "chunks and slots must be multiples of the NoC write alignment");
    static_assert(xfer == 1 || xfer * slot_bytes <= NOC_MAX_BURST_SIZE, "a column is one NoC command");
    static_assert(packet_size == 0 || xfer == 1 || xfer * slot_bytes <= packet_size, "a column fits one packet");
}

// Debug: the host's walk stride (walk_stride_for over its mirror of contiguous_page_stride) is this output
// accessor's.
template <typename Accessor>
FORCE_INLINE void check_walk_stride(
    [[maybe_unused]] const Accessor& out,
    [[maybe_unused]] uint32_t k,
    [[maybe_unused]] uint32_t out_cpp,
    [[maybe_unused]] uint32_t stride) {
    ASSERT(stride == walk_stride_for(out_cpp, k, out.contiguous_page_stride()));
}

////////////////////////////////////////////////////////////////
// Run formation
////////////////////////////////////////////////////////////////

// The host's RunSource for an accessor with `cpp` chunks per page. The host derives cross_page from its
// mirror of contiguous_page_stride() (all_gather_geometry.cpp contiguous_page_stride, which TT_FATALs where
// that mirror cannot be trusted), so the two agree by construction; debug builds check it here, once per
// kernel launch. There is no release-mode fallback: a disagreement is a host bug, not a slow path.
template <typename Accessor>
FORCE_INLINE RunSource checked_run_source(
    RunSource host,
    [[maybe_unused]] const Accessor& acc,
    [[maybe_unused]] uint32_t k,
    [[maybe_unused]] uint32_t cpp,
    [[maybe_unused]] uint32_t stride) {
    // cross_page was granted for page_stride == stride (cpp == k) or page_stride == 1 (cpp > k, stride 1).
    ASSERT(!host.cross_page || acc.contiguous_page_stride() == (cpp == k ? stride : 1u));
    return host;
}

// Chunks contiguous in memory from `chunk` on, stepping the way the walk steps, capped at `bound`
// (accessor chunk space; a whole number of walk pages: every bound is a slice end or a row end). Can
// reach past the current column, so next_run() clips it with run_limit().
//
// The `room` clip: with stride 1 the walk's chunks are consecutive ids, so bound - chunk is exact; it
// trims the rounded-up end_page when bound is mid accessor page (concat: a row ends inside an output
// page). With stride > 1 it is a loose bound (lane chunks below bound are a subset of [chunk, bound)),
// and exactness comes from bound being page-aligned in the accessor: either cpp == k, or no cross-page
// run at all.
template <uint32_t k, uint32_t cpp, typename Accessor>
FORCE_INLINE uint32_t contiguous_chunks(const Accessor& acc, RunSource src, uint32_t chunk, uint32_t bound) {
    static_assert(k != 0 && (k & (k - 1)) == 0, "k is a power of 2");
    static_assert(cpp % k == 0, "an accessor page is a whole number of walk pages");
    ASSERT(bound % k == 0);
    ASSERT(chunk < bound);
    if constexpr (cpp == 1) {
        return src.cross_page ? acc.num_contiguous_pages(chunk, bound) : 1u;
    } else {
        uint32_t n = src.whole_page ? cpp - chunk % cpp : k - chunk % k;
        if (src.cross_page) {
            const uint32_t end_page = (bound + cpp - 1) / cpp;
            n += (acc.num_contiguous_pages(chunk / cpp, end_page) - 1) * cpp;
        }
        const uint32_t room = bound - chunk;
        return n < room ? n : room;
    }
}

////////////////////////////////////////////////////////////////
// The walk
////////////////////////////////////////////////////////////////

// Walks a slice of whole walk pages lane by lane, xfer chunks per column, tiles column-major. One
// column bounds one transfer; next_run() splits it where memory is not contiguous.
//
// STRIDE / XFER are the kernels' CT args. 0 makes them runtime, which only the host gtests use (to sweep them
// without instantiating every pair). K is always CT: page_of / byte_off_of divide by it and advance() shifts by it.
//
// State: c_, rem_, base_, lane_, end_ (+ stride_/xfer_ when runtime). run_limit() is the column's room only --
// the end clip belongs to next_run()'s bound, which every caller already has (input: the slice end; output:
// StripeMap's bound). advance() has no division: K is a power of 2.
template <uint32_t K, uint32_t STRIDE = 0, uint32_t XFER = 0>
class TiledWalk {
    static_assert(K != 0 && (K & (K - 1)) == 0, "k is a power of 2");

public:
    static constexpr uint32_t k = K;

    // Walk the slice ending at `end` (a whole walk page) from `pos`.
    FORCE_INLINE void init(uint32_t end, WalkPos pos) {
        static_assert(STRIDE != 0 && XFER != 0, "runtime stride/xfer: use the 4-argument init");
        start(end, pos);
    }
    FORCE_INLINE void init(uint32_t end, uint32_t stride, uint32_t xfer, WalkPos pos) {
        ASSERT(STRIDE == 0 || stride == STRIDE);
        ASSERT(XFER == 0 || xfer == XFER);
        stride_ = stride;
        xfer_ = xfer;
        start(end, pos);
    }

    FORCE_INLINE uint32_t chunk() const { return c_; }

    // Same position and future (host tests compare a resumed walk against a stepped one). C++17: kernels.
    bool operator==(const TiledWalk& o) const {
        return c_ == o.c_ && rem_ == o.rem_ && base_ == o.base_ && lane_ == o.lane_ && end_ == o.end_ &&
               stride() == o.stride() && xfer() == o.xfer();
    }

    // Chunks left in this column: the longest run allowed here. Not clipped at end -- see the class note.
    FORCE_INLINE uint32_t run_limit() const { return rem_; }

    FORCE_INLINE void advance(uint32_t n) {
        ASSERT(n != 0 && n <= rem_);
        // A run past the slice end means the caller's bound was not the slice end.
        ASSERT(lane_adv(c_, n - 1) < end_);
        rem_ -= n;
        c_ = lane_adv(c_, n);
        if (rem_ != 0 && c_ < end_) {
            return;
        }
        // Next lane; the tile ends after its last lane, or at an empty lane (always a suffix).
        lane_ += K;
        if (lane_ != stride() * K && base_ + lane_ < end_) {
            c_ = base_ + lane_;
            rem_ = xfer();
            return;
        }
        base_ = lane_adv(base_, xfer());
        lane_ = 0;
        c_ = base_;
        rem_ = xfer();
    }

private:
    FORCE_INLINE uint32_t stride() const {
        if constexpr (STRIDE != 0) {
            return STRIDE;
        } else {
            return stride_;
        }
    }
    FORCE_INLINE uint32_t xfer() const {
        if constexpr (XFER != 0) {
            return XFER;
        } else {
            return xfer_;
        }
    }

    // `n` positions further along c's lane: n chunks, plus (stride - 1) pages per walk page crossed.
    FORCE_INLINE uint32_t lane_adv(uint32_t c, uint32_t n) const {
        return c + n + (((c & (K - 1)) + n) / K) * ((stride() - 1) * K);
    }

    FORCE_INLINE void start(uint32_t end, WalkPos pos) {
        ASSERT(end % K == 0);  // slices are whole walk pages (pos.base is mid-page when xfer % K != 0)
        ASSERT(pos.lane < stride() && pos.i < xfer());
        end_ = end;
        base_ = pos.base;
        lane_ = pos.lane * K;
        c_ = lane_adv(base_ + lane_, pos.i);
        rem_ = xfer() - pos.i;
    }

    uint32_t c_;                              // current chunk id
    uint32_t rem_;                            // chunks left in this column
    uint32_t base_;                           // the tile's anchor: lane 0's chunk at the tile's first position
    uint32_t lane_;                           // current lane * K (lanes are K chunk ids apart at equal position)
    uint32_t end_;                            // slice end (chunk id)
    uint32_t stride_ = STRIDE, xfer_ = XFER;  // read only when the template's are 0
};

// Longest run allowed at the walk's current position, given `left` chunks still wanted. `bound` must
// be at most the slice end in the accessor's chunk space (the input end chunk, or StripeMap's bound).
// At one chunk the query is skipped: it cannot help, and on a sharded tensor it loops over rank.
template <uint32_t cpp, typename Walk, typename Accessor>
FORCE_INLINE uint32_t
next_run(const Walk& walk, const Accessor& acc, RunSource src, uint32_t chunk, uint32_t bound, uint32_t left) {
    const uint32_t room = walk.run_limit();
    const uint32_t limit = room < left ? room : left;
    if (limit <= 1) {
        return limit;
    }
    const uint32_t run = contiguous_chunks<Walk::k, cpp>(acc, src, chunk, bound);
    return run < limit ? run : limit;
}

// Debug only: a run has to be one linear stretch of memory (`step` bytes per chunk: the slot), which is
// what one transfer assumes. `walk` is a copy, so probing it does not move the caller's.
template <typename Walk, typename AddrFn>
FORCE_INLINE bool run_is_linear(Walk walk, uint32_t run, uint32_t step, uint64_t first, AddrFn addr) {
    for (uint32_t i = 0; i < run; ++i) {
        if (addr(walk.chunk()) != first + i * step) {
            return false;
        }
        walk.advance(1);
    }
    return true;
}

// A chunk's index in this device's contribution -> its index in the output tensor, plus the bound a
// run from it must stop at: the end of our stripe in this row, or the slice end, whichever is first.
// chunks_per_stripe is in chunks (the stripe's walk pages * k), so rows are whole walk pages.
template <uint32_t chunks_per_stripe, uint32_t num_devices>
class StripeMap {
public:
    struct Pos {
        uint32_t global;  // chunk index in the output tensor
        uint32_t bound;   // one past the last chunk a run from here may take
    };

    // `end` is the slice end in local chunk ids.
    FORCE_INLINE void init(uint32_t stripe, uint32_t end) {
        offset_ = stripe * chunks_per_stripe;
        end_ = end;
    }

    FORCE_INLINE Pos at(uint32_t local) const {
        ASSERT(local < end_);
        const uint32_t row = local / chunks_per_stripe;
        const uint32_t in_row = local - row * chunks_per_stripe;
        const uint32_t global = row * (num_devices * chunks_per_stripe) + offset_ + in_row;
        const uint32_t to_row_end = chunks_per_stripe - in_row;
        const uint32_t to_end = end_ - local;
        return {global, global + (to_row_end < to_end ? to_row_end : to_end)};
    }

private:
    uint32_t offset_;
    uint32_t end_;
};

// A chunk index split into the page holding it and the byte offset inside that page. chunks_per_page
// is the accessor's (in_cpp or out_cpp), k included; chunk_size is the content chunk (in-page chunks
// sit at chunk stride; only one-chunk pages can have a slot that differs, and their offset is 0).
template <uint32_t chunks_per_page>
FORCE_INLINE uint32_t page_of(uint32_t chunk) {
    return chunk / chunks_per_page;
}

template <uint32_t chunks_per_page, uint32_t chunk_size>
FORCE_INLINE uint32_t byte_off_of(uint32_t chunk) {
    return (chunk % chunks_per_page) * chunk_size;
}

////////////////////////////////////////////////////////////////
// Run loops, shared by all four kernels. Each consumes `left` chunks of `walk` as runs, hands every
// run to `f`, and returns the L1 address past the last run. CB slots are packed: chunk j of an entry sits
// at j * slot_bytes, whatever the run grouping (so reader and writer may group differently).
//
// `addr` is only used by the debug ASSERT that a run is one linear stretch of memory; with asserts
// compiled out it is never called.
////////////////////////////////////////////////////////////////

// Output side: runs of this device's stripe in the output tensor, clipped at the row edge and slice end.
// f(l1, page, off, run) -- `page`/`off` locate the run's first chunk in the output; the run covers
// run * slot_bytes bytes on both sides (a padded slot's pad included).
// addr(page, off) -> uint64_t NoC address of that chunk (debug only).
// Used by every writer-side loop (unicast/multicast writer, multicast reader's send) and by the
// unicast reader's relay, which reads back what upstream delivered into our output.
template <
    uint32_t chunks_per_page,
    uint32_t chunk_size,
    uint32_t slot_bytes,
    typename Walk,
    uint32_t chunks_per_stripe,
    uint32_t num_devices,
    typename Accessor,
    typename AddrFn,
    typename F>
FORCE_INLINE uint32_t emit_runs(
    Walk& walk,
    const StripeMap<chunks_per_stripe, num_devices>& map,
    const Accessor& acc,
    RunSource src,
    uint32_t l1_addr,
    uint32_t left,
    AddrFn addr,
    F f) {
    static_assert(slot_bytes == chunk_size || chunks_per_page == 1, "a padded slot needs one chunk per page");
    [[maybe_unused]] auto run_addr = [&](uint32_t chunk) {
        const uint32_t global = map.at(chunk).global;
        return addr(page_of<chunks_per_page>(global), byte_off_of<chunks_per_page, chunk_size>(global));
    };
    while (left > 0) {
        const auto pos = map.at(walk.chunk());
        const uint32_t page = page_of<chunks_per_page>(pos.global);
        const uint32_t off = byte_off_of<chunks_per_page, chunk_size>(pos.global);
        const uint32_t run = next_run<chunks_per_page>(walk, acc, src, pos.global, pos.bound, left);
        ASSERT(run_is_linear(walk, run, slot_bytes, addr(page, off), run_addr));
        f(l1_addr, page, off, run);
        l1_addr += run * slot_bytes;
        left -= run;
        walk.advance(run);
    }
    return l1_addr;
}

// Input side: runs of this device's own input, chunk ids [.., end_chunk) (the slice end).
// addr(chunk) -> uint64_t NoC address of an input chunk; f(src, l1, run) reads `run` chunks from `src` into
// `run` slots. A multi-chunk run exists only where the input steps by the slot (the host's in_runs), so
// it is (run - 1) * slot_bytes + chunk_size contiguous bytes.
template <
    uint32_t chunks_per_page,
    uint32_t chunk_size,
    uint32_t slot_bytes,
    typename Walk,
    typename Accessor,
    typename AddrFn,
    typename F>
FORCE_INLINE uint32_t read_input_runs(
    Walk& walk,
    const Accessor& acc,
    RunSource src,
    uint32_t end_chunk,
    uint32_t l1_addr,
    uint32_t left,
    AddrFn addr,
    F f) {
    while (left > 0) {
        const uint32_t chunk = walk.chunk();
        const uint32_t run = next_run<chunks_per_page>(walk, acc, src, chunk, end_chunk, left);
        const uint64_t src_addr = addr(chunk);
        ASSERT(run_is_linear(walk, run, slot_bytes, src_addr, addr));
        f(src_addr, l1_addr, run);
        l1_addr += run * slot_bytes;
        left -= run;
        walk.advance(run);
    }
    return l1_addr;
}
