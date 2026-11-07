// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm — the port item walk shared by the port reader and the port sender (dataflow only).
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace agrms_port {

// A send / receive entry: slice q, landing-tile columns [c0, c1) of that slice (this link's part), and a row filter
// rf: 0 = every tile-row, 1 / 2 = the even / odd tile-rows only (the far slice of a balanced ring split between the two
// directions by rows, so every packet carries the link's whole column run: Refinement 3, FAR_ROW_SPLIT).
struct Entry {
    uint32_t q;
    uint32_t c0;
    uint32_t c1;
    uint32_t rf;
    FORCE_INLINE bool empty() const { return c1 <= c0; }
    FORCE_INLINE bool has_row(uint32_t r) const { return rf == 0 || (r & 1) == rf - 1; }
    // rows of [r0, r0 + rows) this entry carries
    FORCE_INLINE uint32_t rows_in(uint32_t r0, uint32_t rows) const {
        return rf == 0 ? rows : (rows + ((r0 & 1) == rf - 1 ? 1 : 0)) / 2;
    }
};

FORCE_INLINE Entry unpack_entry(uint32_t v) { return Entry{(v >> 24) & 0x3F, (v >> 12) & 0xFFF, v & 0xFFF, v >> 30}; }

// Geometry shared by the port reader and sender (compile-time args + a few runtime values).
struct Geom {
    uint32_t page_bytes;    // T_in
    uint32_t run_pages;     // tiles per packet
    uint32_t R_t;           // tile-rows per chip
    uint32_t n_groups;      // active row groups (row r -> group r % n_groups, local row r / n_groups)
    uint32_t k;             // members per group
    uint32_t w_pitch;       // landing pages per local row
    uint32_t xfer_rows;     // rows per transport item
    uint32_t Wt_lp;         // tiles per slice
    uint32_t landing_base;  // cb_x_landing base address (identical on every core of every chip)
    uint32_t ws_arg;        // runtime-arg index of member_ws[0..k] (landing tile boundaries of the members)
    uint32_t rows_per_group;  // 0: round-robin rows (row r -> group r % n_groups, local row r / n_groups);
                              // > 0 (native HEIGHT shards): group r / rows_per_group, local row r % rows_per_group
    uint32_t x_base;          // native HEIGHT: input shard address (own slice read in place, row pitch Wt_lp)

    // group and local row of tile-row r
    FORCE_INLINE void place(uint32_t r, uint32_t& grp, uint32_t& li) const {
        if (rows_per_group == 0) {
            grp = r % n_groups;
            li = r / n_groups;
        } else {
            grp = r / rows_per_group;
            li = r - grp * rows_per_group;
        }
    }

    FORCE_INLINE uint32_t ws(uint32_t m) const { return get_arg_val<uint32_t>(ws_arg + m); }
    FORCE_INLINE uint32_t n_xb() const { return (R_t + xfer_rows - 1) / xfer_rows; }

    // chunks one row of an entry splits into (identical for every row: the member layout is per group)
    FORCE_INLINE uint32_t chunks_per_row(const Entry& e) const {
        if (e.empty()) {
            return 0;
        }
        const uint32_t t0 = e.q * Wt_lp + e.c0, t1 = e.q * Wt_lp + e.c1;
        uint32_t n = 0;
        for (uint32_t m = 0; m < k; ++m) {
            const uint32_t a = t0 > ws(m) ? t0 : ws(m);
            const uint32_t b = t1 < ws(m + 1) ? t1 : ws(m + 1);
            if (a < b) {
                n += (b - a + run_pages - 1) / run_pages;
            }
        }
        return n;
    }

    FORCE_INLINE uint32_t item_rows(uint32_t xb) const {
        const uint32_t r0 = xb * xfer_rows;
        return (R_t - r0) < xfer_rows ? (R_t - r0) : xfer_rows;
    }

    // items xb < c of entry e that carry at least one row (a row-filtered entry can have empty items: they send
    // nothing, so they carry no arrival increment either). Closed form: with xfer_rows >= 2 only a 1-row last item
    // can miss the filter; with xfer_rows == 1 item xb is row xb.
    FORCE_INLINE uint32_t items_before(const Entry& e, uint32_t c) const {
        if (e.empty() || e.rf == 0 || c == 0) {
            return e.empty() ? 0 : c;
        }
        if (xfer_rows == 1) {
            return e.rf == 1 ? (c + 1) / 2 : c / 2;
        }
        const uint32_t last = n_xb() - 1;
        return c - ((c > last && e.rows_in(last * xfer_rows, item_rows(last)) == 0) ? 1 : 0);
    }
};

// Walk the chunks of item (entry e, row-block xb) in order. f(worker_index, l1_addr, bytes, is_last_chunk).
// A chunk is a run of <= run_pages landing tiles of one row inside one member (contiguous in that member's L1);
// a member's segment of a row is split into balanced chunks. from_input: l1_addr is the chunk's source in the
// member's resident input shard (native HEIGHT own slice) instead of its landing slot (same chunk boundaries).
template <typename F>
FORCE_INLINE void for_each_chunk_of_item(
    const Geom& g, const Entry& e, uint32_t xb, uint32_t cpr, F&& f, bool from_input = false) {
    const uint32_t t0 = e.q * g.Wt_lp + e.c0, t1 = e.q * g.Wt_lp + e.c1;
    const uint32_t r0 = xb * g.xfer_rows;
    const uint32_t rows = g.item_rows(xb);
    uint32_t left = e.rows_in(r0, rows) * cpr;
    for (uint32_t r = r0; r < r0 + rows; ++r) {
        if (!e.has_row(r)) {
            continue;
        }
        uint32_t grp, li;
        g.place(r, grp, li);
        for (uint32_t m = 0; m < g.k; ++m) {
            const uint32_t ws = g.ws(m), we = g.ws(m + 1);
            const uint32_t a = t0 > ws ? t0 : ws;
            const uint32_t b = t1 < we ? t1 : we;
            if (a >= b) {
                continue;
            }
            const uint32_t len = b - a;
            const uint32_t nch = (len + g.run_pages - 1) / g.run_pages;
            const uint32_t base_n = len / nch, rem = len % nch;
            uint32_t off = a;
            for (uint32_t c = 0; c < nch; ++c) {
                const uint32_t n = base_n + (c < rem ? 1 : 0);
                const uint32_t addr = from_input ? g.x_base + (li * g.Wt_lp + (off - e.q * g.Wt_lp)) * g.page_bytes
                                                 : g.landing_base + (li * g.w_pitch + (off - ws)) * g.page_bytes;
                --left;
                f(grp * g.k + m, addr, n * g.page_bytes, left == 0);
                off += n;
            }
        }
    }
}

}  // namespace agrms_port
