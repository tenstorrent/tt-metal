// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The all_gather page walk and slot layout. Plain arithmetic over accessor queries, with no device includes: the
// host plan (walk_shape, slot_offsets) and the host sim (test_all_gather_kernel_lib.cpp) compile it as is.
//
//   unit:    A bytes, the cut quantum. Local unit u is global unit g = (u / S) * N * S + stripe * S + u % S, which
//            lives in output page g / Uo at byte (g % Uo) * A.
//   element: what the walk steps over: E units, one output page (E == Uo) or one unit (E == 1).
//   stripe:  the S units this device contributes to each output row.
//   lane:    local elements lambda apart whose output pages are L = contiguous_page_stride() apart, i.e. adjacent
//            in memory.
//   run:     elements contiguous at the destination, sent as one segment.
// The walk is a tiled transpose (up to h elements down one lane, then the next lane), so one packet's segments are
// long runs in one bank instead of one page from each bank.

#include <algorithm>
#include <cstdint>

// The walk and the packet builder are always inlined into the reader, so the compile-time geometry (named CT args)
// folds into their arithmetic: divisions by S, Uo, lambda become shifts and multiplies. Out of line each run paid
// 6-10 hardware divides (BH, 26 MB: reader cost per packet 1142 -> 935 cycles at k = 2).
#define AG_INLINE inline __attribute__((always_inline))

namespace ag {

// Output geometry; counts are in units of A bytes.
struct WalkGeo {
    uint32_t A;      // the cut quantum: a multiple of 16 and of every source's read alignment
    uint32_t Uo;     // output page footprint, ceil(page content / A)
    uint32_t S;      // units this device contributes to one output row
    uint32_t N;      // devices
    uint32_t E;      // units per walk element: Uo when S is whole output pages, else 1
    uint32_t P_A;    // packet payload cap, a multiple of A
    bool dst_cross;  // an output page's footprint is its bank pitch, so a run may continue into the next page
};

struct WalkShape {
    uint32_t lambda;  // element step inside a run
    uint32_t h;       // column cap: elements per lane before the walk moves to the next lane
    bool cross_row;   // runs continue across output rows
};

// L: the output's contiguous_page_stride(). Shared by the kernel walk and the host cost model.
//
// If N * (S / Uo) divides L, lambda = L / N steps whole rows onto the next page of the same bank, so a run is not
// clipped at the row end (cross_row: 6-74% faster on BH where it applies). Otherwise lambda = L (or 1 for unit
// elements) and a run stops at the row end, where the other devices' stripes start; the column is then capped at
// one row of its lane so that most columns are a single run. A column is at most a packet (but at least one element).
constexpr WalkShape walk_shape(const WalkGeo& g, uint32_t L) {
    const bool pages = g.E == g.Uo;
    const bool cross_row = pages && L % (g.N * (g.S / g.Uo)) == 0;
    const uint32_t lambda = cross_row ? L / g.N : (pages ? L : 1);
    uint32_t h = g.P_A / (g.E * g.A);
    if (pages && !cross_row) {
        const uint32_t row_lane = (g.S / g.E + lambda - 1) / lambda;
        h = std::min(h, row_lane);
    }
    return {lambda, std::max<uint32_t>(h, 1), cross_row};
}

// ---- slot layout (one CB page) ----
// Here rather than in ag_packet.hpp because the host plan sizes slots and cannot include the fabric packet header.

// Written by the reader after the slot's packets. units: A-units of one iteration. iter_end: this is the iteration's
// last slot, so the writer signals after it (SignalCadence in ag_sync.hpp) without re-deriving the schedule.
struct SlotInfo {
    uint16_t packets;
    uint16_t iter_end;
    uint32_t units;
};

// sizeof(ag::PacketRecord) (ag_packet.hpp asserts it). Spelled out so host code needs no fabric header.
inline constexpr uint32_t kPacketRecordBytes = 48;

// [SlotInfo][k records | pad to A][k headers, each padded to A][k payloads of P_A]. Every offset is a multiple of A
// and so is slot_bytes, so payload L1 addresses are congruent with their A-aligned sources (CB bases are
// DRAM-aligned, a multiple of A). The header region belongs to the writer; the reader never writes it.
struct SlotLayout {
    uint32_t rec_off, hdr_off, hdr_stride, pay_off, pay_stride, slot_bytes;

    // L1 addresses of packet j's record, header and payload in the slot at `slot`.
    constexpr uint32_t record(uint32_t slot, uint32_t j) const { return slot + rec_off + j * kPacketRecordBytes; }
    constexpr uint32_t header(uint32_t slot, uint32_t j) const { return slot + hdr_off + j * hdr_stride; }
    constexpr uint32_t payload(uint32_t slot, uint32_t j) const { return slot + pay_off + j * pay_stride; }
};

constexpr uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

constexpr SlotLayout slot_offsets(uint32_t k, uint32_t A, uint32_t header_bytes, uint32_t P_A) {
    // Records hold uint64_t addresses; SlotInfo is 8 bytes, so they start 8-aligned right after it.
    const uint32_t rec_off = sizeof(SlotInfo);
    const uint32_t hdr_off = align_up(rec_off + k * kPacketRecordBytes, A);
    const uint32_t hdr_stride = align_up(header_bytes, A);
    const uint32_t pay_off = hdr_off + k * hdr_stride;
    return {rec_off, hdr_off, hdr_stride, pay_off, P_A, pay_off + k * P_A};
}

// The most units one slot of k packets can carry. A relay reader that has consumed c units of an iteration waits for
// min(c + max_slot_units, iteration units) before reserving the next slot, so it never waits holding one.
constexpr uint32_t max_slot_units(uint32_t k, uint32_t A, uint32_t P_A) { return k * (P_A / A); }

// ---- the walk ----

// Walks ids as tiles of xfer * stride, each read column-major: xfer ids stride apart, then the next lane. A column is
// the longest run allowed; xfer == 1 is ascending order. The last tile is ragged (columns hb_ or hb_ + 1 tall),
// the only reason this holds state beyond the position.
class TiledWalk {
public:
    // Walk [first, first + count) from seqno `skip`. `stride` and `xfer` are at least 1.
    AG_INLINE void init(uint32_t first, uint32_t count, uint32_t skip, uint32_t stride, uint32_t xfer) {
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
        if (skip >= count) {
            // Nothing left to walk (an empty slice, or a resume at its end).
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
            // Lanes below he_ hold one extra element, so seqnos before lane r are r * hb_ + min(r, he_).
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

    AG_INLINE uint32_t element() const { return e_; }

    // Elements left in this column, i.e. the longest run still allowed here.
    AG_INLINE uint32_t run_limit() const { return col() - k_; }

    // n in [1, run_limit()].
    AG_INLINE void advance(uint32_t n) {
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
    AG_INLINE bool in_tail() const { return tile_first_ == tail_first_; }
    AG_INLINE uint32_t col() const { return in_tail() ? hb_ + (lane_ < he_ ? 1u : 0u) : xfer_; }
    // Empty columns can only be a tail of lanes, so a lane step never has to skip one.
    AG_INLINE uint32_t lanes() const { return (in_tail() && hb_ == 0) ? he_ : stride_; }

    uint32_t e_ = 0, stride_ = 1, xfer_ = 1, tile_ = 1, lane_ = 0, k_ = 0, tile_first_ = 0, tail_first_ = 0, hb_ = 0,
             he_ = 0;
};

// n elements at local elements l0 + k * lambda, linear at the destination from addr. addr is the DstAddr frame (for
// a fabric header); local is the same place on the caller's own NOC, for its own reads and writes. They differ
// wherever the fabric frame is not the caller's (BH DRAM: the fabric frame is NOC1's bank endpoint).
struct DstRun {
    uint32_t l0, n, lambda;
    uint64_t addr, local;
};

// The default destination address: the accessor's own, on the caller's NOC. Kernels pass ag::FabricFrameAddr
// (ag_sync.hpp) so the address can go into a fabric header.
struct AccessorAddr {
    template <class Acc>
    uint64_t operator()(const Acc& acc, uint32_t page, uint32_t offset) const {
        return acc.get_noc_addr(page, offset);
    }
};

// The destination side of the walk: runs of elements contiguous in the output, one accessor address query per run.
template <class OutAcc, class DstAddr = AccessorAddr>
class Walk {
public:
    Walk(const OutAcc& out, const WalkGeo& g, DstAddr addr = {}) :
        out_(out),
        addr_(addr),
        A_(g.A),
        Uo_(g.Uo),
        S_(g.S),
        N_(g.N),
        E_(g.E),
        L_(out.contiguous_page_stride()),
        dst_cross_(g.dst_cross) {
        const WalkShape s = walk_shape(g, L_);
        lambda_ = s.lambda;
        h_ = s.h;
        cross_row_ = s.cross_row;
    }

    // Walk positions [skip, skip + take) of elements [e0, e0 + ne), for `stripe`. The order depends only on
    // (e0, ne), never on the stripe, so a relay's positions match its upstream's.
    // When runs stop at the row end, a slice that starts mid-row walks that partial row first (head), then tiles
    // anchored at row starts: tiles anchored mid-row would straddle every row end and split every column in two.
    AG_INLINE void init(uint32_t stripe, uint32_t e0, uint32_t ne, uint32_t skip, uint32_t take) {
        stripe_units_ = stripe * S_;
        left_ = take;
        const uint32_t row_elems = S_ / E_;
        const uint32_t into_row = e0 % row_elems;
        const uint32_t head = (cross_row_ || into_row == 0) ? 0 : std::min(ne, row_elems - into_row);
        body_first_ = e0 + head;
        body_ne_ = ne - head;
        if (skip < head) {
            head_left_ = head - skip;
            tw_.init(e0, head, skip, lambda_, h_);
        } else {
            head_left_ = 0;
            tw_.init(body_first_, body_ne_, skip - head, lambda_, h_);
        }
    }

    AG_INLINE bool next(DstRun& r) {
        if (left_ == 0) {
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
        r = {l, n, lambda_, addr_(out_, page, off * A_), out_.get_noc_addr(page, off * A_)};
        tw_.advance(n);
        left_ -= n;
        if (head_left_ != 0) {
            head_left_ -= n;
            if (head_left_ == 0) {
                tw_.init(body_first_, body_ne_, 0, lambda_, h_);
            }
        }
        return true;
    }

private:
    // How many of the n elements from (page, off) are contiguous in memory; the caller has clipped n to owned elements.
    AG_INLINE uint32_t dst_contig(uint32_t page, uint32_t off, uint32_t n) const {
        if (E_ == Uo_) {
            // Element k is page + k * L.
            return dst_cross_ ? out_.num_contiguous_pages(page, page + (n - 1) * L_ + 1) : 1;
        }
        // Unit elements, lambda == 1: the rest of this page, then whole pages if they follow it.
        uint32_t m = Uo_ - off;
        if (m < n && dst_cross_ && L_ == 1) {
            const uint32_t end_page = page + (off + n + Uo_ - 1) / Uo_;
            m += (out_.num_contiguous_pages(page, end_page) - 1) * Uo_;
        }
        return m;
    }

    const OutAcc& out_;
    DstAddr addr_;
    uint32_t A_, Uo_, S_, N_, E_, L_, lambda_ = 1, h_ = 1;
    bool dst_cross_, cross_row_ = false;
    uint32_t stripe_units_ = 0, left_ = 0, head_left_ = 0, body_first_ = 0, body_ne_ = 0;
    TiledWalk tw_;
};

// A stretch linear in the source; bytes is a multiple of A. A read stops at the source page's allocation
// (readable_bytes < bytes): the slot's stale tail then lands in the destination page's padding.
struct Piece {
    uint64_t src;
    uint32_t bytes, readable_bytes;
};

// Splits a dst run into stretches linear in the input, in destination order. Splits add reads, never segments.
// Usage: start(run), then next() until it returns false.
template <class InAcc>
class InputPieces {
public:
    InputPieces(const InAcc& in, uint32_t A, uint32_t Ui, uint32_t E, bool src_cross) :
        in_(in),
        A_(A),
        Ui_(Ui),
        E_(E),
        L_(in.contiguous_page_stride()),
        ai_(in.get_aligned_page_size()),
        src_cross_(src_cross) {}

    AG_INLINE void start(const DstRun& r) {
        r_ = r;
        if (r.lambda == 1) {
            // Consecutive elements are consecutive units.
            span_u_ = r.l0 * E_;
            span_n_ = r.n * E_;
            k_ = r.n;
        } else {
            span_n_ = 0;
            k_ = 0;
        }
        // Elements lambda apart continue in the input only as whole input pages at the input's own stride.
        pages_ = src_cross_ && E_ == Ui_ && r.lambda == L_;
    }

    AG_INLINE bool next(Piece& p) {
        if (span_n_ == 0) {
            if (k_ == r_.n) {
                return false;
            }
            const uint32_t l = r_.l0 + k_ * r_.lambda;
            if (pages_) {
                const uint32_t m = in_.num_contiguous_pages(l, l + (r_.n - k_ - 1) * r_.lambda + 1);
                const uint32_t bytes = m * E_ * A_;
                p = {in_.get_noc_addr(l), bytes, bytes};
                k_ += m;
                return true;
            }
            span_u_ = l * E_;
            span_n_ = E_;
            ++k_;
        }
        // Units [span_u_, span_u_ + span_n_): their input pages are consecutive ids, which continue in memory only
        // when the input's stride is 1.
        const uint32_t page = span_u_ / Ui_;
        const uint32_t off = span_u_ - page * Ui_;
        uint32_t m = Ui_ - off;
        if (m < span_n_ && src_cross_ && L_ == 1) {
            m += (in_.num_contiguous_pages(page, (span_u_ + span_n_ + Ui_ - 1) / Ui_) - 1) * Ui_;
        }
        m = std::min(m, span_n_);
        const uint32_t bytes = m * A_;
        // Without src_cross a piece is inside one page, so the allocation bounds it.
        p = {in_.get_noc_addr(page, off * A_), bytes, src_cross_ ? bytes : std::min(bytes, ai_ - off * A_)};
        span_u_ += m;
        span_n_ -= m;
        return true;
    }

private:
    const InAcc& in_;
    uint32_t A_, Ui_, E_, L_, ai_;
    bool src_cross_, pages_ = false;
    DstRun r_{};
    uint32_t k_ = 0, span_u_ = 0, span_n_ = 0;
};

// Relay: the run is re-read from our own output, where upstream wrote it: one piece at the run's own-NOC address.
class RelayPieces {
public:
    RelayPieces(uint32_t A, uint32_t E) : unit_bytes_(A * E) {}

    AG_INLINE void start(const DstRun& r) {
        const uint32_t bytes = r.n * unit_bytes_;
        p_ = {r.local, bytes, bytes};
        pending_ = true;
    }

    AG_INLINE bool next(Piece& p) {
        p = p_;
        const bool had = pending_;
        pending_ = false;
        return had;
    }

private:
    uint32_t unit_bytes_;
    Piece p_{};
    bool pending_ = false;
};

}  // namespace ag
