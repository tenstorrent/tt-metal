// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host tests (test_all_gather_kernel_lib.cpp) compile this file with their own FORCE_INLINE and ASSERT.
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "api/dataflow/dataflow_api.h"
#endif

#include <algorithm>
#include <cstdint>
#include <type_traits>

// Page walk, shared by both all_gather algorithms.
//   unit:    A bytes, the cut quantum. Local unit u is global unit g = (u / S) * N * S + stripe * S + u % S,
//            which lives in output page g / Uo at byte (g % Uo) * A.
//   element: what the walk steps over: E units, one output page (E == Uo) or one unit (E == 1).
//   stripe:  the S units this device contributes to each output row.
//   lane:    local elements lambda apart whose output pages are L = contiguous_page_stride() apart, i.e. adjacent
//            in memory.
//   run:     elements contiguous at the destination, sent as one segment.
// The walk is a tiled transpose (a packet's worth from one lane, then the next), so a core never parks on one
// DRAM bank.

// Walks ids as tiles of xfer * stride, each read column-major: xfer ids stride apart, then the next lane. A
// column is the longest run allowed; xfer == 1 is ascending order. The last tile is ragged (columns hb_ or
// hb_ + 1 tall), the only reason this holds state beyond the position.
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
            tile_first_ = e_ = first;
            return;
        }
        if (skip < full * tile_) {
            const uint32_t tile = skip / tile_;
            const uint32_t in_tile = skip - tile * tile_;
            tile_first_ = first + tile * tile_;
            lane_ = in_tile / xfer;
            k_ = in_tile - lane_ * xfer;
        } else {
            // Lanes below he_ hold one extra element, so seqnos before lane r are r*hb_ + min(r, he_).
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
        e_ = tile_first_ + lane_ + k_ * stride_;
    }

    FORCE_INLINE uint32_t element() const { return e_; }

    // Elements left in this column, i.e. the longest run still allowed here.
    FORCE_INLINE uint32_t run_limit() const { return col() - k_; }

    FORCE_INLINE void advance(uint32_t n) {
        ASSERT(n != 0 && n <= run_limit());
        k_ += n;
        if (k_ < col()) {
            e_ += n * stride_;
            return;
        }
        k_ = 0;
        if (++lane_ == lanes()) {
            lane_ = 0;
            tile_first_ += tile_;
        }
        e_ = tile_first_ + lane_;
    }

private:
    FORCE_INLINE bool in_tail() const { return tile_first_ == tail_first_; }
    FORCE_INLINE uint32_t col() const { return in_tail() ? hb_ + (lane_ < he_ ? 1u : 0u) : xfer_; }
    // Empty columns can only be a tail of lanes, so a lane step never has to skip one.
    FORCE_INLINE uint32_t lanes() const { return (in_tail() && hb_ == 0) ? he_ : stride_; }

    uint32_t e_, stride_, xfer_, tile_, lane_, k_, tile_first_, tail_first_, hb_, he_;
};

// An interleaved TensorAccessor (device): its runs of pages L apart do not depend on where they start.
template <class T, class = void>
struct IsInterleavedAcc : std::false_type {};
template <class T>
struct IsInterleavedAcc<T, std::void_t<decltype(T::DSpec::is_interleaved)>>
    : std::bool_constant<T::DSpec::is_interleaved> {};

// P is cut to a multiple of A here. dst_cross: runs may cross output pages. cross_row_enabled: a tuning knob.
struct WalkGeo {
    uint32_t A, Uo, S, N, E, P;
    bool dst_cross;
    bool cross_row_enabled;
};

// n elements at local elements l0 + k * lambda, linear at the destination from addr.
struct DstRun {
    uint32_t l0, n, lambda;
    uint64_t addr;
};

// The destination side of the walk: runs of elements contiguous in the output, one accessor query per run.
// If N * (S / Uo) divides L, lambda = L / N steps whole rows onto the next page of the same bank, so a run is
// never clipped at the row end (cross_row). Otherwise lambda = L (or 1 for unit elements) and a run stops at the
// row end, where other devices' stripes start.
template <class OutAcc>
struct Walk {
    Walk(const OutAcc& out, const WalkGeo& g) :
        out_(out), A_(g.A), Uo_(g.Uo), S_(g.S), N_(g.N), E_(g.E), dst_cross_(g.dst_cross) {
        L_ = out.contiguous_page_stride();
        const bool pages = g.E == g.Uo;
        cross_row_ = g.cross_row_enabled && pages && L_ % (g.N * (g.S / g.Uo)) == 0;
        lambda_ = cross_row_ ? L_ / g.N : (pages ? L_ : 1);
        // A column is at most a packet, and (when runs stop at the row end) at most one row of its lane, so
        // that most columns are a single run.
        uint32_t per_lane = (g.P - g.P % g.A) / (g.E * g.A);
        if (!cross_row_ && pages) {
            const uint32_t row_lane = (g.S / g.E + lambda_ - 1) / lambda_;
            per_lane = per_lane < row_lane ? per_lane : row_lane;
        }
        h_ = per_lane > 1 ? per_lane : 1;
        if constexpr (IsInterleavedAcc<OutAcc>::value) {
            // n pages L apart are all contiguous, or (page size not the bank pitch) none are: probe once.
            whole_runs_ = out.num_contiguous_pages(0, L_ + 1) == 2;
        }
    }

    // Walk positions [skip, skip + take) of elements [e0, e0 + ne), for `stripe`. The order depends
    // only on (e0, ne), never on the stripe, so a relay's positions match its upstream's.
    FORCE_INLINE void init(uint32_t stripe, uint32_t e0, uint32_t ne, uint32_t skip, uint32_t take) {
        stripe_units_ = stripe * S_;
        left_ = take;
        tw_.init(e0, ne, skip, lambda_, h_);
    }

    FORCE_INLINE bool next(DstRun& r) {
        if (left_ == 0) [[unlikely]] {
            return false;
        }
        const uint32_t l = tw_.element();
        const uint32_t u = l * E_;
        const uint32_t row = u / S_;
        const uint32_t g = row * N_ * S_ + stripe_units_ + (u - row * S_);
        const uint32_t page = g / Uo_;
        const uint32_t off = g - page * Uo_;
        uint32_t n = std::min(tw_.run_limit(), left_);
        if (n > 1 && !cross_row_) {
            // Elements of this row from l on, lambda apart.
            const uint32_t row_end = (row + 1) * (S_ / E_);
            n = std::min(n, (row_end - l - 1) / lambda_ + 1);
        }
        if (n > 1) {
            n = std::min(n, dst_contig(page, off, n));
        }
        r = {l, n, lambda_, out_.get_noc_addr(page, off * A_)};
        tw_.advance(n);
        left_ -= n;
        return true;
    }

private:
    // Elements contiguous in memory from (page, off), at most n, all owned (the caller clipped n).
    FORCE_INLINE uint32_t dst_contig(uint32_t page, uint32_t off, uint32_t n) const {
        if (E_ == Uo_) {
            // Element k is page + k * L.
            if constexpr (IsInterleavedAcc<OutAcc>::value) {
                return dst_cross_ && whole_runs_ ? n : 1;
            }
            return dst_cross_ ? out_.num_contiguous_pages(page, page + (n - 1) * L_ + 1) : 1;
        }
        // Elements are units, lambda == 1: the rest of this page, then whole pages if they follow it.
        uint32_t m = Uo_ - off;
        if (m < n && dst_cross_ && L_ == 1) {
            const uint32_t end_page = page + (off + n + Uo_ - 1) / Uo_;
            m += (out_.num_contiguous_pages(page, end_page) - 1) * Uo_;
        }
        return m;
    }

    const OutAcc& out_;
    uint32_t A_, Uo_, S_, N_, E_, L_, lambda_, h_;
    bool dst_cross_, cross_row_, whole_runs_ = false;
    uint32_t stripe_units_ = 0, left_ = 0;
    TiledWalk tw_;
};

// A stretch linear in the source, `bytes` a multiple of A. A whole-page read stops at the source page's allocation
// (readable_bytes < bytes); the stale slot tail lands in the destination page's padding.
struct Piece {
    uint64_t src;
    uint32_t bytes, readable_bytes;
};

// Splits a dst run into stretches linear in the input. Splits add reads, never segments.
template <class InAcc>
struct InputPieces {
    InputPieces(const InAcc& in, uint32_t A, uint32_t Ui, uint32_t E, bool src_cross) :
        in_(in),
        A_(A),
        Ui_(Ui),
        E_(E),
        L_(in.contiguous_page_stride()),
        ai_(in.get_aligned_page_size()),
        src_cross_(src_cross) {
        if constexpr (IsInterleavedAcc<InAcc>::value) {
            whole_runs_ = in.num_contiguous_pages(0, L_ + 1) == 2;
        }
    }

    template <class F>
    FORCE_INLINE void for_each(const DstRun& r, F&& f) {
        if (r.lambda == 1) {
            span(r.l0 * E_, r.n * E_, f);
            return;
        }
        // Elements lambda apart continue in the input only as whole input pages at its own stride.
        const bool pages = src_cross_ && E_ == Ui_ && r.lambda == L_;
        for (uint32_t k = 0; k < r.n;) {
            const uint32_t l = r.l0 + k * r.lambda;
            if (pages) {
                uint32_t m;
                if constexpr (IsInterleavedAcc<InAcc>::value) {
                    m = whole_runs_ ? r.n - k : 1;
                } else {
                    m = in_.num_contiguous_pages(l, l + (r.n - k - 1) * r.lambda + 1);
                }
                f(Piece{in_.get_noc_addr(l), m * E_ * A_, m * E_ * A_});
                k += m;
            } else {
                span(l * E_, E_, f);
                ++k;
            }
        }
    }

private:
    // Local units [u, u + n). Their input pages are consecutive ids, which continue in memory only
    // when the input's stride is 1.
    template <class F>
    FORCE_INLINE void span(uint32_t u, uint32_t n, F& f) {
        while (n) {
            const uint32_t page = u / Ui_;
            const uint32_t off = u - page * Ui_;
            uint32_t m = Ui_ - off;
            if (m < n && src_cross_ && L_ == 1) {
                m += (in_.num_contiguous_pages(page, (u + n + Ui_ - 1) / Ui_) - 1) * Ui_;
            }
            m = std::min(m, n);
            // Without src_cross a piece is inside one page, so ai_ bounds it.
            const uint32_t bytes = m * A_;
            f(Piece{in_.get_noc_addr(page, off * A_), bytes, src_cross_ ? bytes : std::min(bytes, ai_ - off * A_)});
            u += m;
            n -= m;
        }
    }

    const InAcc& in_;
    uint32_t A_, Ui_, E_, L_, ai_;
    bool src_cross_, whole_runs_ = false;
};

// Relay: the run is re-read from our own output, where upstream wrote it: one piece at the run's address.
struct RelayPieces {
    uint32_t A, E;

    template <class F>
    FORCE_INLINE void for_each(const DstRun& r, F&& f) {
        f(Piece{r.addr, r.n * E * A, r.n * E * A});
    }
};
