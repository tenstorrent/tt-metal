// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The all_gather page walk, packet builder, slot filler and sync helpers both algorithms' kernels share. The geometry
// and the slot layout are shared with the host (all_gather_walk.hpp, which also defines the terms used here).
//
// The walk goes block by block (up to `height` elements down one lane, then the next lane), so one packet's chunks are
// long runs in one bank instead of one page from each bank. A run is elements contiguous at the destination, sent as
// one chunk.
//
// Address rule. Two kinds of address, never mixed:
//   - Fabric address: every address placed in a fabric command header. The header setters re-translate it for the NOC
//     the receiving router writes with.
//       - Record (data) addresses come from the walk (addrgen_detail::get_noc_address), the form the fabric write
//         setters expect.
//       - Semaphore and mirror addresses come from noc0_addr(x, y, l1). Never get_noc_addr(x, y, l1) on a NOC1
//         writer: that is a NOC1 address.
//   - Own-NOC address: every read or write a RISC issues itself (accessor addresses on its own NOC,
//     DstRun::own_addr). A fabric address is not usable there: on BH a DRAM record address names the bank's NOC1
//     endpoint, which a NOC0 read or write does not reach. So the reader re-reads relays from, and writes local copies
//     to, the own-NOC chunk addresses PacketBuilder::next reports, never record addresses.
//
// Sync protocol (counted units, subtract-consume, never reset: a reset could drop a fast neighbour's credits for the
// next launch):
//   - init: every writer first sends its init (unicast: one to the paired worker of its neighbor; multicast: one to
//     every other device), then waits for the init credits addressed to it and subtracts them, before its first data
//     send. Always on, persistent output included: a device may still be reading its previous output when a fast
//     neighbor starts writing the next one. The reader does not wait for init: it fills slots and does the local copy
//     during the round trip.
//   - data_valid (unicast): the writer signals cumulative units after the data they cover, on the same sender (FIFO),
//     every slots_per_sync slots. The reader waits only while holding no slot.
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
// folds into their arithmetic: divisions by stripe_units, out_page_units and step become shifts and multiplies instead
// of hardware divides on every run.

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

namespace ttnn::operations::ccl::all_gather_walk {

static_assert(kMaxChunksPerPacket <= NOC_SCATTER_WRITE_MAX_CHUNKS, "a packet has more chunks than a scatter write");

// ---- compile-time args ----

// The WalkGeo the host appended at compile-time arg `base` (kWalkCtArgCount args with the packet cap).
template <uint32_t base>
constexpr WalkGeo walk_geo_from_ct_args() {
    return {
        .unit_bytes = get_compile_time_arg_val(base + 0),
        .in_page_units = get_compile_time_arg_val(base + 1),
        .out_page_units = get_compile_time_arg_val(base + 2),
        .stripe_units = get_compile_time_arg_val(base + 3),
        .num_devices = get_compile_time_arg_val(base + 4),
        .element_units = get_compile_time_arg_val(base + 5),
        .max_packet_bytes = get_compile_time_arg_val(base + 6),
        .dst_cross = get_compile_time_arg_val(base + 7) != 0,
        .src_cross = get_compile_time_arg_val(base + 8) != 0};
}

// The packet cap that follows the WalkGeo at compile-time arg `base`.
template <uint32_t base>
constexpr uint32_t packet_cap_from_ct_args() {
    constexpr WalkGeo geo = walk_geo_from_ct_args<base>();
    constexpr uint32_t cap = get_compile_time_arg_val(base + kWalkCtArgCount - 1);
    static_assert(
        cap % geo.unit_bytes == 0 && cap >= geo.unit_bytes && cap <= geo.max_packet_bytes,
        "the packet cap is a whole number of units within the payload");
    return cap;
}

// The slot layout, checked against the CB page size (slot_bytes) the host sized it to.
template <uint32_t packets, uint32_t unit_bytes, uint32_t header_bytes, uint32_t packet_cap, uint32_t slot_bytes>
constexpr SlotLayout checked_slot_layout() {
    constexpr SlotLayout lay = slot_offsets(packets, unit_bytes, header_bytes, packet_cap);
    static_assert(lay.slot_bytes == slot_bytes, "host and kernel disagree on the slot layout");
    return lay;
}

// ---- the walk ----

// Walks elements as blocks of height * step, each read column-major: `height` elements `step` apart, then the next
// lane. A column is the longest run allowed; height == 1 is ascending order. The last block is ragged (columns
// tail_height_ or tail_height_ + 1 tall), the only reason this holds state beyond the position.
class BlockWalk {
public:
    // Walk [first, first + count) from position `skip`. `step` and `height` are at least 1.
    FORCE_INLINE void init(uint32_t first, uint32_t count, uint32_t skip, uint32_t step, uint32_t height) {
        step_ = step;
        height_ = height;
        block_size_ = height * step;
        const uint32_t full_blocks = count / block_size_;
        const uint32_t tail_count = count - full_blocks * block_size_;
        tail_first_ = first + full_blocks * block_size_;
        tail_height_ = tail_count / step;
        tail_tall_lanes_ = tail_count - tail_height_ * step;
        lane_ = 0;
        row_ = 0;
        if (skip >= count) {
            // Nothing left to walk (an empty slice, or a resume at its end).
            block_first_ = element_ = first;
            return;
        }
        if (skip < full_blocks * block_size_) {
            const uint32_t block = skip / block_size_;
            const uint32_t in_block = skip - block * block_size_;
            block_first_ = first + block * block_size_;
            lane_ = in_block / height;
            row_ = in_block - lane_ * height;
        } else {
            // Lanes below tail_tall_lanes_ hold one extra element, so the positions before lane r are
            // r * tail_height_ + min(r, tail_tall_lanes_).
            block_first_ = tail_first_;
            const uint32_t tail_pos = skip - full_blocks * block_size_;
            const uint32_t tall_elements = tail_tall_lanes_ * (tail_height_ + 1);
            if (tail_pos < tall_elements) {
                lane_ = tail_pos / (tail_height_ + 1);
                row_ = tail_pos - lane_ * (tail_height_ + 1);
            } else {
                // tail_height_ > 0 here: at tail_height_ == 0 the block holds only its taller lanes, so
                // `tail_pos < tall_elements`.
                const uint32_t rest = tail_pos - tall_elements;
                lane_ = tail_tall_lanes_ + rest / tail_height_;
                row_ = rest - (lane_ - tail_tall_lanes_) * tail_height_;
            }
        }
        element_ = block_first_ + lane_ + row_ * step_;
    }

    FORCE_INLINE uint32_t element() const { return element_; }

    // Elements left in this column, i.e. the longest run still allowed here.
    FORCE_INLINE uint32_t run_limit() const { return col() - row_; }

    // n in [1, run_limit()].
    FORCE_INLINE void advance(uint32_t n) {
        row_ += n;
        if (row_ < col()) {
            element_ += n * step_;
            return;
        }
        row_ = 0;
        if (++lane_ == lanes()) {
            lane_ = 0;
            block_first_ += block_size_;
        }
        element_ = block_first_ + lane_;
    }

private:
    FORCE_INLINE bool in_tail() const { return block_first_ == tail_first_; }
    FORCE_INLINE uint32_t col() const {
        return in_tail() ? tail_height_ + (lane_ < tail_tall_lanes_ ? 1u : 0u) : height_;
    }
    // Empty columns can only be a tail of lanes, so a lane step never has to skip one.
    FORCE_INLINE uint32_t lanes() const { return (in_tail() && tail_height_ == 0) ? tail_tall_lanes_ : step_; }

    uint32_t element_ = 0, step_ = 1, height_ = 1, block_size_ = 1, lane_ = 0, row_ = 0, block_first_ = 0,
             tail_first_ = 0, tail_height_ = 0, tail_tall_lanes_ = 0;
};

// `count` elements at local elements first + k * step, linear at the destination from addr. addr is a fabric address
// (for a fabric header); own_addr is the same place on the caller's own NOC, for its own reads and writes.
struct DstRun {
    uint32_t first, count, step;
    uint64_t addr, own_addr;
};

// The destination side of the walk: runs of elements contiguous in the output, one accessor address query per run.
template <class OutAcc>
class Walk {
public:
    Walk(const OutAcc& out, const WalkGeo& g) :
        out_(out),
        unit_bytes_(g.unit_bytes),
        out_page_units_(g.out_page_units),
        stripe_units_(g.stripe_units),
        num_devices_(g.num_devices),
        element_units_(g.element_units),
        out_page_stride_(out.contiguous_page_stride()),
        dst_cross_(g.dst_cross) {
        const WalkShape s = walk_shape(g, out_page_stride_);
        step_ = s.step;
        height_ = s.height;
        cross_row_ = s.cross_row;
    }

    // Walk positions [skip, skip + take) of elements [e0, e0 + ne), for `stripe`. The order depends only on
    // (e0, ne), never on the stripe, so a relay's positions match its upstream's.
    // When runs stop at the row end, a slice that starts mid-row walks that partial row first (head), then blocks
    // anchored at row starts: blocks anchored mid-row would straddle every row end and split every column in two.
    FORCE_INLINE void init(uint32_t stripe, uint32_t e0, uint32_t ne, uint32_t skip, uint32_t take) {
        stripe_offset_units_ = stripe * stripe_units_;
        left_ = take;
        const uint32_t row_elems = stripe_units_ / element_units_;
        const uint32_t into_row = e0 % row_elems;
        const uint32_t head = (cross_row_ || into_row == 0) ? 0 : std::min(ne, row_elems - into_row);
        body_first_ = e0 + head;
        body_ne_ = ne - head;
        if (skip < head) {
            head_left_ = head - skip;
            bw_.init(e0, head, skip, step_, height_);
        } else {
            head_left_ = 0;
            bw_.init(body_first_, body_ne_, skip - head, step_, height_);
        }
    }

    // True when next() has no run left.
    FORCE_INLINE bool done() const { return left_ == 0; }

    FORCE_INLINE bool next(DstRun& r) {
        if (left_ == 0) {
            return false;
        }
        const uint32_t element = bw_.element();
        const uint32_t unit = element * element_units_;
        const uint32_t row = unit / stripe_units_;
        const uint32_t out_unit =
            row * num_devices_ * stripe_units_ + stripe_offset_units_ + (unit - row * stripe_units_);
        const uint32_t page = out_unit / out_page_units_;
        const uint32_t off = out_unit - page * out_page_units_;
        uint32_t n = std::min(bw_.run_limit(), left_);
        if (n > 1 && !cross_row_) {
            // Elements of this row from `element` on, step apart.
            const uint32_t row_end = (row + 1) * (stripe_units_ / element_units_);
            n = std::min(n, (row_end - element - 1) / step_ + 1);
        }
        if (n > 1) {
            n = std::min(n, dst_contig(page, off, n));
        }
        r = {
            element,
            n,
            step_,
            tt::tt_fabric::linear::addrgen_detail::get_noc_address(out_, page, off * unit_bytes_),
            out_.get_noc_addr(page, off * unit_bytes_)};
        bw_.advance(n);
        left_ -= n;
        if (head_left_ != 0) {
            head_left_ -= n;
            if (head_left_ == 0) {
                bw_.init(body_first_, body_ne_, 0, step_, height_);
            }
        }
        return true;
    }

private:
    // How many of the n elements from (page, off) are contiguous in memory; the caller has clipped n to owned elements.
    FORCE_INLINE uint32_t dst_contig(uint32_t page, uint32_t off, uint32_t n) const {
        if (element_units_ == out_page_units_) {
            // Element k is page + k * out_page_stride_.
            return dst_cross_ ? out_.num_contiguous_pages(page, page + (n - 1) * out_page_stride_ + 1) : 1;
        }
        // Unit elements, step == 1: the rest of this page, then whole pages if they follow it.
        uint32_t m = out_page_units_ - off;
        if (m < n && dst_cross_ && out_page_stride_ == 1) {
            const uint32_t end_page = page + (off + n + out_page_units_ - 1) / out_page_units_;
            m += (out_.num_contiguous_pages(page, end_page) - 1) * out_page_units_;
        }
        return m;
    }

    const OutAcc& out_;
    uint32_t unit_bytes_, out_page_units_, stripe_units_, num_devices_, element_units_, out_page_stride_;
    uint32_t step_ = 1, height_ = 1;
    bool dst_cross_, cross_row_ = false;
    uint32_t stripe_offset_units_ = 0, left_ = 0, head_left_ = 0, body_first_ = 0, body_ne_ = 0;
    BlockWalk bw_;
};

// A piece linear in the source; bytes is a multiple of unit_bytes. A read stops at the source page's allocation
// (readable_bytes < bytes): the slot's stale tail then lands in the destination page's padding.
struct Piece {
    uint64_t src;
    uint32_t bytes, readable_bytes;
};

// Splits a dst run into pieces linear in the input, in destination order. Splits add reads, never chunks.
// Usage: start(run), then next() until it returns false.
template <class InAcc>
class InputPieces {
public:
    InputPieces(const InAcc& in, const WalkGeo& g) :
        in_(in),
        unit_bytes_(g.unit_bytes),
        in_page_units_(g.in_page_units),
        element_units_(g.element_units),
        in_page_stride_(in.contiguous_page_stride()),
        in_aligned_page_bytes_(in.get_aligned_page_size()),
        src_cross_(g.src_cross) {}

    FORCE_INLINE void start(const DstRun& r) {
        run_ = r;
        if (r.step == 1) {
            // Consecutive elements are consecutive units.
            unit_ = r.first * element_units_;
            units_left_ = r.count * element_units_;
            next_ = r.count;
        } else {
            units_left_ = 0;
            next_ = 0;
        }
        // Elements `step` apart continue in the input only as whole input pages at the input's own stride; then one
        // read covers the run.
        whole_pages_ = src_cross_ && element_units_ == in_page_units_ && r.step == in_page_stride_;
    }

    // True when next() has no piece of the run left.
    FORCE_INLINE bool done() const { return units_left_ == 0 && next_ == run_.count; }

    FORCE_INLINE bool next(Piece& p) {
        if (units_left_ == 0) {
            if (next_ == run_.count) {
                return false;
            }
            const uint32_t element = run_.first + next_ * run_.step;
            if (whole_pages_) {
                const uint32_t m =
                    in_.num_contiguous_pages(element, element + (run_.count - next_ - 1) * run_.step + 1);
                const uint32_t bytes = m * element_units_ * unit_bytes_;
                p = {in_.get_noc_addr(element), bytes, bytes};
                next_ += m;
                return true;
            }
            unit_ = element * element_units_;
            units_left_ = element_units_;
            ++next_;
        }
        // Units [unit_, unit_ + units_left_): their input pages are consecutive ids, which continue in memory only
        // when the input's stride is 1.
        const uint32_t page = unit_ / in_page_units_;
        const uint32_t off = unit_ - page * in_page_units_;
        uint32_t m = in_page_units_ - off;
        if (m < units_left_ && src_cross_ && in_page_stride_ == 1) {
            m += (in_.num_contiguous_pages(page, (unit_ + units_left_ + in_page_units_ - 1) / in_page_units_) - 1) *
                 in_page_units_;
        }
        m = std::min(m, units_left_);
        const uint32_t bytes = m * unit_bytes_;
        // Without src_cross a piece is inside one page, so the allocation bounds it.
        p = {
            in_.get_noc_addr(page, off * unit_bytes_),
            bytes,
            src_cross_ ? bytes : std::min(bytes, in_aligned_page_bytes_ - off * unit_bytes_)};
        unit_ += m;
        units_left_ -= m;
        return true;
    }

private:
    const InAcc& in_;
    uint32_t unit_bytes_, in_page_units_, element_units_, in_page_stride_, in_aligned_page_bytes_;
    bool src_cross_, whole_pages_ = false;
    DstRun run_{};
    uint32_t next_ = 0, unit_ = 0, units_left_ = 0;  // next_: the run's elements already split
};

// Relay: the run is re-read from our own output, where upstream wrote it: one piece at the run's own-NOC address.
class RelayPieces {
public:
    explicit RelayPieces(const WalkGeo& g) : element_bytes_(g.unit_bytes * g.element_units) {}

    FORCE_INLINE void start(const DstRun& r) {
        const uint32_t bytes = r.count * element_bytes_;
        p_ = {r.own_addr, bytes, bytes};
        pending_ = true;
    }

    // True when next() has no piece of the run left.
    FORCE_INLINE bool done() const { return !pending_; }

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

// Packs the runs a walk yields, in order, into packets of at most `cap` bytes and kMaxChunksPerPacket chunks, cut on
// the unit. W: Walk. Src: InputPieces or RelayPieces. A piece that starts where the last chunk ends extends it, so a
// run split into several input pieces, or runs that happen to abut, cost one chunk. A packet never spans walk
// iterations: next() returns 0 once the walk is exhausted, and the caller re-inits the walk.
template <class W, class Src>
class PacketBuilder {
public:
    PacketBuilder(W& walk, Src& src, uint32_t cap) : walk_(walk), src_(src), cap_(cap) {}

    // Fills one packet: read(src_noc_addr, payload_off, bytes) for each read into this packet's payload
    // (payload_off a multiple of unit_bytes), then writes *rec. Returns the payload bytes, 0 when the walk is
    // exhausted (rec untouched). own_addrs, if given, receives each chunk's own-NOC address (DstRun::own_addr), for a
    // local copy.
    template <class Read>
    FORCE_INLINE uint32_t next(PacketRecord* rec, Read&& read, uint64_t* own_addrs = nullptr) {
        uint64_t addrs[kMaxChunksPerPacket];
        uint16_t sizes[kMaxChunksPerPacket];
        uint32_t num_chunks = 0, used = 0;
        while (used < cap_ && (have_ || pull())) {
            const bool merge = num_chunks != 0 && dst_ == addrs[num_chunks - 1] + sizes[num_chunks - 1];
            if (!merge && num_chunks == kMaxChunksPerPacket) {
                break;
            }
            // Both are multiples of unit_bytes, so every cut is on the unit.
            const uint32_t n = std::min(cur_.bytes, cap_ - used);
            const uint32_t rd = std::min(n, cur_.readable_bytes);
            if (rd != 0) {
                read(cur_.src, used, rd);
            }
            if (merge) {
                // Contiguous fabric addresses are the same bank and page run, so contiguous on our own NOC too.
                sizes[num_chunks - 1] += n;
            } else {
                addrs[num_chunks] = dst_;
                sizes[num_chunks] = n;
                if (own_addrs != nullptr) {
                    own_addrs[num_chunks] = own_;
                }
                ++num_chunks;
            }
            used += n;
            dst_ += n;
            own_ += n;
            cur_.src += n;
            cur_.bytes -= n;
            cur_.readable_bytes -= rd;
            have_ = cur_.bytes != 0;
        }
        if (used != 0) {
            for (uint32_t i = 0; i < num_chunks; ++i) {
                rec->dst[i] = addrs[i];
                rec->chunk_bytes[i] = sizes[i];
            }
            rec->num_chunks = static_cast<uint16_t>(num_chunks);
            rec->payload_bytes = static_cast<uint16_t>(used);
        }
        return used;
    }

    // True when the walk has nothing left to send.
    FORCE_INLINE bool exhausted() const { return walk_.done() && !have_ && src_.done(); }

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
            run_own_ = r.own_addr;
        }
        dst_ = run_dst_;
        own_ = run_own_;
        run_dst_ += cur_.bytes;
        run_own_ += cur_.bytes;
        have_ = true;
        return true;
    }

    W& walk_;
    Src& src_;
    const uint32_t cap_;
    Piece cur_{};                     // the unsent rest of the current piece, valid while have_
    uint64_t dst_ = 0;                // where cur_ lands
    uint64_t run_dst_ = 0;            // where the run's next piece lands
    uint64_t own_ = 0, run_own_ = 0;  // the same two places on the caller's own NOC
    bool have_ = false;
};

// ---- sync ----

// A core's L1 address for a fabric header (semaphores, mirror cores).
FORCE_INLINE uint64_t noc0_addr(uint32_t x, uint32_t y, uint32_t l1) { return safe_get_noc_addr(x, y, l1, 0); }

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

// The reader's slot loop, one walk pass, until the builder is exhausted: per slot, wait(units already pushed) ->
// reserve -> up to lay.packets packets (records + payload reads) -> one read barrier -> [local copy, flushed] ->
// SlotInfo -> push. The slot that exhausts the builder ends the pass (iter_end), and an empty pass pushes none.
// local_copy also writes every packet into our own output, on a second VC so the writes do not queue behind the fabric
// traffic on this NOC; the caller's final write barrier acks them.
template <class Builder, class Wait>
FORCE_INLINE void fill_slots(
    Noc& noc,
    CircularBuffer& cb,
    const SlotLayout& lay,
    uint32_t unit_bytes,
    Builder& builder,
    bool local_copy,
    Wait&& wait) {
    auto own_addrs = [&](uint32_t slot, uint32_t j) { return reinterpret_cast<uint64_t*>(lay.own_addrs(slot, j)); };
    if (builder.exhausted()) {
        return;
    }
    uint32_t consumed = 0;
    for (bool iter_end = false; !iter_end;) {
        wait(consumed);
        cb.reserve_back(1);
        const uint32_t slot = cb.get_write_ptr();
        uint32_t packets = 0, units = 0;
        for (; packets < lay.packets; ++packets) {
            auto* rec = reinterpret_cast<PacketRecord*>(lay.record(slot, packets));
            const uint32_t payload = lay.payload(slot, packets);
            auto into_payload = [&](uint64_t src, uint32_t off, uint32_t n) {
                noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    PrecomposedUnicastEndpoint{}, CoreLocalMem<uint32_t>(payload + off), n, {.noc_addr = src}, {});
            };
            const uint32_t bytes = builder.next(rec, into_payload, local_copy ? own_addrs(slot, packets) : nullptr);
            if (bytes == 0) {
                break;
            }
            units += bytes / unit_bytes;
        }
        noc.async_read_barrier();
        if (local_copy) {
            for (uint32_t j = 0; j < packets; ++j) {
                const auto& rec = *reinterpret_cast<const PacketRecord*>(lay.record(slot, j));
                const uint64_t* dst = own_addrs(slot, j);
                for (uint32_t i = 0, off = 0; i < rec.num_chunks; off += rec.chunk_bytes[i], ++i) {
                    noc.async_write<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                        CoreLocalMem<uint32_t>(lay.payload(slot, j) + off),
                        PrecomposedUnicastEndpoint{},
                        rec.chunk_bytes[i],
                        {},
                        {.noc_addr = dst[i]},
                        {.vc = NOC_UNICAST_WRITE_VC + 1});
                }
            }
            noc.async_writes_flushed();
        }
        consumed += units;
        iter_end = builder.exhausted();
        *reinterpret_cast<SlotInfo*>(slot) =
            SlotInfo{static_cast<uint16_t>(packets), static_cast<uint16_t>(iter_end ? 1 : 0), units};
        slot_publish();
        cb.push_back(1);
    }
}

}  // namespace ttnn::operations::ccl::all_gather_walk
