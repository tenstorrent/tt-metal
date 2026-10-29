// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The all_gather page walk, packet builder, slot filler and sync helpers both algorithms' kernels share. The geometry
// and the slot layout are shared with the host (all_gather_walk.hpp, which also defines the symbols used here; Ui is
// in_page_units).
//
// The walk is a tiled transpose (up to h elements down one lane, then the next lane), so one packet's segments are
// long runs in one bank instead of one page from each bank. A run is elements contiguous at the destination, sent as
// one segment.
//
// Address rule. Two frames, never mixed:
//   - Fabric frame: every address placed in a fabric command header. The header setters re-translate it for the NOC
//     the receiving router writes with.
//       - Record (data) addresses come from the walk (addrgen_detail::get_noc_address), the frame the fabric write
//         setters expect.
//       - Semaphore and mirror addresses come from fabric_frame_addr(x, y, l1). Never get_noc_addr(x, y, l1) on a NOC1
//         writer: that is a NOC1-frame address.
//   - Own frame: every read or write a RISC issues itself (accessor addresses on its own NOC, DstRun::local). The
//     fabric frame is not usable there: on BH a DRAM record address names the bank's NOC1 endpoint, which a NOC0
//     read or write does not reach. So the reader re-reads relays from, and writes local copies to, the own-NOC
//     segment addresses PacketBuilder::next reports, never record addresses.
//
// Sync protocol (counted units, subtract-consume, never reset: a reset could drop a fast neighbour's credits for the
// next launch):
//   - init: every writer first sends its init (unicast: one to the paired worker of its neighbor; multicast: one to
//     every other device), then waits for the init credits addressed to it and subtracts them, before its first data
//     send. Always on, persistent output included: a device may still be reading its previous output when a fast
//     neighbor starts writing the next one. The reader does not wait for init: it fills slots and does the local copy
//     during the round trip.
//   - data_valid (unicast): the writer signals cumulative units after the data they cover, on the same sender (FIFO),
//     at a cadence. The reader waits only while holding no slot.
//   - end: the reader waits for every unit (unicast) or done credit (multicast) it will receive, subtracts them, and
//     barriers its atomics.
//
// Deadlock freedom. The writer's only semaphore wait is init, once, before its first send. Init is sent
// unconditionally at every writer's start, before that writer waits on anything, so every init wait is met once every
// writer has started. Beyond it, a writer waits only for its own reader's slots and for fabric / mux credits, which the
// fabric drains. The reader waits for data_valid only holding no slot, for units its upstream sent in the previous
// iteration; iteration 0 reads only the input. So by induction on the iteration every wait is met.
//
// The walk and the packet builder are always inlined (FORCE_INLINE) into the reader, so the compile-time geometry
// folds into their arithmetic: divisions by S, Uo, lambda become shifts and multiplies instead of 6-10 hardware
// divides per run (BH, 26 MB, 2 packets per slot: 18% fewer reader cycles per packet).

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "ttnn/operations/ccl/all_gather/shared_with_host/all_gather_walk.hpp"

// TEST ONLY, DO NOT UPSTREAM: TT_AG_ZONES=1 defines AG_TEST_ZONES (device-profiler zones around the init wait, the
// main loop and the tail); TT_AG_NO_INIT=1 defines AG_TEST_NO_INIT (skip the init handshake; races across back-to-back
// launches, diagnosis only). See all_gather_test_knobs.hpp.
#ifdef AG_TEST_ZONES
#define AG_TEST_ZONE(name) DeviceZoneScopedN(name)
#else
#define AG_TEST_ZONE(name)
#endif

namespace ttnn::operations::ccl::all_gather_walk {

// ---- the walk ----

// Walks ids as tiles of xfer * stride, each read column-major: xfer ids stride apart, then the next lane. A column is
// the longest run allowed; xfer == 1 is ascending order. The last tile is ragged (columns hb_ or hb_ + 1 tall),
// the only reason this holds state beyond the position.
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

    FORCE_INLINE uint32_t element() const { return e_; }

    // Elements left in this column, i.e. the longest run still allowed here.
    FORCE_INLINE uint32_t run_limit() const { return col() - k_; }

    // n in [1, run_limit()].
    FORCE_INLINE void advance(uint32_t n) {
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

    uint32_t e_ = 0, stride_ = 1, xfer_ = 1, tile_ = 1, lane_ = 0, k_ = 0, tile_first_ = 0, tail_first_ = 0, hb_ = 0,
             he_ = 0;
};

// n elements at local elements l0 + k * lambda, linear at the destination from addr. addr is in the fabric frame (for
// a fabric header); local is the same place on the caller's own NOC, for its own reads and writes.
struct DstRun {
    uint32_t l0, n, lambda;
    uint64_t addr, local;
};

// The destination side of the walk: runs of elements contiguous in the output, one accessor address query per run.
template <class OutAcc>
class Walk {
public:
    Walk(const OutAcc& out, const WalkGeo& g) :
        out_(out),
        A_(g.unit_bytes),
        Uo_(g.out_page_units),
        S_(g.stripe_units),
        N_(g.num_devices),
        E_(g.element_units),
        L_(out.contiguous_page_stride()),
        dst_cross_(g.dst_cross) {
        const WalkShape s = walk_shape(g, L_);
        lambda_ = s.lambda;
        h_ = s.column_height;
        cross_row_ = s.cross_row;
    }

    // Walk positions [skip, skip + take) of elements [e0, e0 + ne), for `stripe`. The order depends only on
    // (e0, ne), never on the stripe, so a relay's positions match its upstream's.
    // When runs stop at the row end, a slice that starts mid-row walks that partial row first (head), then tiles
    // anchored at row starts: tiles anchored mid-row would straddle every row end and split every column in two
    // (BH: bfp8 9 MB 20-25% slower without the head, bf16 4-26 MB up to 5%).
    FORCE_INLINE void init(uint32_t stripe, uint32_t e0, uint32_t ne, uint32_t skip, uint32_t take) {
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

    FORCE_INLINE bool next(DstRun& r) {
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
        r = {
            l,
            n,
            lambda_,
            tt::tt_fabric::linear::addrgen_detail::get_noc_address(out_, page, off * A_),
            out_.get_noc_addr(page, off * A_)};
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
    FORCE_INLINE uint32_t dst_contig(uint32_t page, uint32_t off, uint32_t n) const {
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
    InputPieces(const InAcc& in, const WalkGeo& g) :
        in_(in),
        A_(g.unit_bytes),
        Ui_(g.in_page_units),
        E_(g.element_units),
        L_(in.contiguous_page_stride()),
        ai_(in.get_aligned_page_size()),
        src_cross_(g.src_cross) {}

    FORCE_INLINE void start(const DstRun& r) {
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
        // Elements lambda apart continue in the input only as whole input pages at the input's own stride; then one
        // read covers the run (BH, interleaved tiles: 3-6% faster at 8-26 MB bf16, 13-18% on a bfp8 ring).
        pages_ = src_cross_ && E_ == Ui_ && r.lambda == L_;
    }

    FORCE_INLINE bool next(Piece& p) {
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
    explicit RelayPieces(const WalkGeo& g) : element_bytes_(g.unit_bytes * g.element_units) {}

    FORCE_INLINE void start(const DstRun& r) {
        const uint32_t bytes = r.n * element_bytes_;
        p_ = {r.local, bytes, bytes};
        pending_ = true;
    }

    FORCE_INLINE bool next(Piece& p) {
        p = p_;
        const bool had = pending_;
        pending_ = false;
        return had;
    }

private:
    uint32_t element_bytes_;
    Piece p_{};
    bool pending_ = false;
};

// ---- packets ----

// Packs the runs a walk yields, in order, into packets of at most `cap` bytes and kSegmentsPerPacket segments, cut on
// the unit. W: Walk. Src: InputPieces or RelayPieces. A piece that starts where the last segment ends extends it, so a
// run split into several input pieces, or runs that happen to abut, cost one segment. A packet never spans walk
// iterations: next() returns 0 once the walk is exhausted, and the caller re-inits the walk.
template <class W, class Src>
class PacketBuilder {
public:
    PacketBuilder(W& walk, Src& src, uint32_t cap) : walk_(walk), src_(src), cap_(cap) {}

    // Fills one packet: read(src_noc_addr, payload_off, bytes) for each read into this packet's payload
    // (payload_off a multiple of A), then writes *rec. Returns the payload bytes, 0 when the walk is exhausted (rec
    // untouched). local, if given, receives each segment's own-NOC address (DstRun::local), for a local copy.
    template <class Read>
    FORCE_INLINE uint32_t next(PacketRecord* rec, Read&& read, uint64_t* local = nullptr) {
        uint64_t addrs[kSegmentsPerPacket];
        uint16_t sizes[kSegmentsPerPacket];
        uint32_t nseg = 0, used = 0;
        while (used < cap_ && (have_ || pull())) {
            const bool merge = nseg != 0 && dst_ == addrs[nseg - 1] + sizes[nseg - 1];
            if (!merge && nseg == kSegmentsPerPacket) {
                break;
            }
            // Both are multiples of A, so every cut is on A.
            const uint32_t n = std::min(cur_.bytes, cap_ - used);
            const uint32_t rd = std::min(n, cur_.readable_bytes);
            if (rd != 0) {
                read(cur_.src, used, rd);
            }
            if (merge) {
                // Contiguous in the fabric frame is the same bank and page run, so contiguous locally too.
                sizes[nseg - 1] += n;
            } else {
                addrs[nseg] = dst_;
                sizes[nseg] = n;
                if (local != nullptr) {
                    local[nseg] = local_;
                }
                ++nseg;
            }
            used += n;
            dst_ += n;
            local_ += n;
            cur_.src += n;
            cur_.bytes -= n;
            cur_.readable_bytes -= rd;
            have_ = cur_.bytes != 0;
        }
        if (used != 0) {
            for (uint32_t i = 0; i < nseg; ++i) {
                rec->dst[i] = addrs[i];
                rec->bytes[i] = sizes[i];
            }
            rec->segments = static_cast<uint16_t>(nseg);
            rec->payload_bytes = static_cast<uint16_t>(used);
        }
        return used;
    }

private:
    // Loads the next piece (and the next run when this one is done); false when the walk is exhausted.
    FORCE_INLINE bool pull() {
        while (!src_.next(cur_)) {
            DstRun r;
            if (!walk_.next(r)) {
                return false;
            }
            src_.start(r);
            run_dst_ = r.addr;
            run_local_ = r.local;
        }
        dst_ = run_dst_;
        local_ = run_local_;
        run_dst_ += cur_.bytes;
        run_local_ += cur_.bytes;
        have_ = true;
        return true;
    }

    W& walk_;
    Src& src_;
    const uint32_t cap_;
    Piece cur_{};                         // the unsent rest of the current piece, valid while have_
    uint64_t dst_ = 0;                    // where cur_ lands
    uint64_t run_dst_ = 0;                // where the run's next piece lands
    uint64_t local_ = 0, run_local_ = 0;  // the same two places on the caller's own NOC
    bool have_ = false;
};

// ---- sync ----

// A core's L1 address for a fabric header (semaphores, mirror cores).
FORCE_INLINE uint64_t fabric_frame_addr(uint32_t x, uint32_t y, uint32_t l1) { return safe_get_noc_addr(x, y, l1, 0); }

// A slot's SlotInfo and records are written by the reader's CPU and read by the writer's CPU. CB push_back and
// wait_front are volatile accesses, which do not order ordinary loads and stores, and BH RISCs cache L1 reads. So the
// reader calls slot_publish() between building a slot and push_back, and the writer calls slot_acquire() between
// wait_front and reading the slot.
FORCE_INLINE void slot_publish() {
    asm volatile("" ::: "memory");  // compiler barrier: keep the slot's record and SlotInfo stores before push_back
}

FORCE_INLINE void slot_acquire() {
    invalidate_l1_cache();          // drop cached L1 lines, so this RISC sees the records the reader RISC just wrote
    asm volatile("" ::: "memory");  // compiler barrier: keep the slot reads after wait_front and the invalidate
}

// Waits until the local semaphore reaches `count`, then takes `count` off it. Call Noc::async_atomic_barrier()
// before the kernel ends.
FORCE_INLINE void wait_and_consume(uint32_t sem_l1, uint32_t count) {
    if (count == 0) {
        return;
    }
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1), count);
    noc_semaphore_inc(get_noc_addr(sem_l1), uint32_t{0} - count);
}

// The reader's slot loop, one walk pass of take_units units: per slot, wait(consumed) -> reserve -> up to k packets
// (records + payload reads) -> one read barrier -> [local copy, flushed] -> SlotInfo -> push. local_copy also writes
// every packet into our own output, on a second VC so the writes do not queue behind the fabric traffic on this NOC;
// the caller's final write barrier acks them.
template <class Builder, class Wait>
FORCE_INLINE void fill_slots(
    Noc& noc,
    CircularBuffer& cb,
    const SlotLayout& lay,
    uint32_t unit_bytes,
    Builder& builder,
    uint32_t take_units,
    bool local_copy,
    Wait&& wait) {
    auto local = [&](uint32_t slot, uint32_t j) { return reinterpret_cast<uint64_t*>(lay.local(slot, j)); };
    for (uint32_t consumed = 0; consumed < take_units;) {
        wait(consumed);
        cb.reserve_back(1);
        const uint32_t slot = cb.get_write_ptr();
        uint32_t packets = 0, units = 0;
        for (; packets < lay.k; ++packets) {
            auto* rec = reinterpret_cast<PacketRecord*>(lay.record(slot, packets));
            const uint32_t payload = lay.payload(slot, packets);
            auto into_payload = [&](uint64_t src, uint32_t off, uint32_t n) {
                noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    PrecomposedUnicastEndpoint{}, CoreLocalMem<uint32_t>(payload + off), n, {.noc_addr = src}, {});
            };
            const uint32_t bytes = builder.next(rec, into_payload, local_copy ? local(slot, packets) : nullptr);
            if (bytes == 0) {
                break;
            }
            units += bytes / unit_bytes;
        }
        // The walk ending before the schedule's units (a host / walk mismatch) would push empty slots forever. A
        // kernel has no release-mode way to fail the op, so this is a watcher assert.
        ASSERT(units != 0);
        noc.async_read_barrier();
        if (local_copy) {
            for (uint32_t j = 0; j < packets; ++j) {
                const auto& rec = *reinterpret_cast<const PacketRecord*>(lay.record(slot, j));
                const uint64_t* dst = local(slot, j);
                for (uint32_t i = 0, off = 0; i < rec.segments; off += rec.bytes[i], ++i) {
                    noc.async_write<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                        CoreLocalMem<uint32_t>(lay.payload(slot, j) + off),
                        PrecomposedUnicastEndpoint{},
                        rec.bytes[i],
                        {},
                        {.noc_addr = dst[i]},
                        {.vc = NOC_UNICAST_WRITE_VC + 1});
                }
            }
            noc.async_writes_flushed();
        }
        consumed += units;
        *reinterpret_cast<SlotInfo*>(slot) =
            SlotInfo{static_cast<uint16_t>(packets), static_cast<uint16_t>(consumed == take_units ? 1 : 0), units};
        slot_publish();
        cb.push_back(1);
    }
}

}  // namespace ttnn::operations::ccl::all_gather_walk

namespace ag = ttnn::operations::ccl::all_gather_walk;
