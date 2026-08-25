// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Wire format and ring geometry for the store-and-forward combine path, shared verbatim by the
// host program factory and both sender kernels.  A single definition is the point: the producer
// derives a destination page from these formulas and the consumer on the next chip derives the
// same page from the same formulas, with nothing negotiated at run time, so any divergence
// between host and device shows up as data landing at a plausible-but-wrong address.

namespace ttnn::operations::experimental::deepseek_prefill::combine::sf {

// Tail words, appended after the token payload in both the L1 ring slot and the DRAM staging
// page.  The tail is INVARIANT across hops: it names the final destination, never this hop's, so
// a relay re-sends the page without rewriting any of it.  That is what lets re-forwarding be one
// contiguous read into a ring slot and one contiguous write back out.
constexpr uint32_t TAIL_OUTPUT_PAGE_IDX = 0;
constexpr uint32_t TAIL_FINAL_DST_CHIP = 1;
constexpr uint32_t TAIL_MAGIC = 2;
constexpr uint32_t TAIL_RESERVED = 3;
constexpr uint32_t TAIL_WORDS = 4;

constexpr uint32_t tail_bytes() { return TAIL_WORDS * sizeof(uint32_t); }

// Guards against consuming a staging page that was never written.  The arrival counter is what
// actually bounds reads; this catches an off-by-one in that accounting, which would otherwise
// surface as one wrong token rather than as a failure.
constexpr uint32_t MAGIC = 0x5AF2C0DEu;

// A relay does not trust the tail to tell it how far the page still has to travel -- it derives
// that from which FIFO the page arrived in, and cross-checks it against the routing invariant
// manhattan_distance(this_chip, TAIL_FINAL_DST_CHIP) == that FIFO's level.  Checking the real
// invariant beats checking a copy of it that the sender would have to keep re-stamping.

// Reader -> writer queue header, one per queued item.  Words 0..3 keep the meaning and offsets
// the non-store-and-forward path already uses, so the writer's packet-header construction is
// shared between both paths.  The header occupies a full DRAM alignment so the payload that
// follows it is a legal destination for a relay's DRAM read.
constexpr uint32_t HDR_ROUTE = 0;     // 1D EDM direction index
constexpr uint32_t HDR_DISTANCE = 1;  // hops for the 1D packet header; always 1 here
constexpr uint32_t HDR_PAGE_IDX = 2;  // output page for a final write, staging page for a relay
constexpr uint32_t HDR_DST_CHIP = 3;  // packet destination: the NEIGHBOUR, never the final chip
constexpr uint32_t HDR_CMD = 4;
constexpr uint32_t HDR_SEM_ADDR = 5;  // target of an atomic-inc command
constexpr uint32_t HDR_INC_VALUE = 6;
constexpr uint32_t HDR_INC_DIR = 7;  // direction the inc travels, which for a credit is the
                                     // opposite of the data direction it accounts for

// Reader -> writer command.  The writer holds no routing state of its own; it switches on this
// and sends exactly one hop.
enum Cmd : uint32_t {
    CMD_FINAL_WRITE = 0,  // payload -> neighbour's output page, tail not sent
    CMD_STAGE = 1,        // payload + tail -> neighbour's staging page
    CMD_ARRIVED_INC = 2,  // header-only atomic inc of the downstream arrival counter
    CMD_CREDIT_INC = 3,   // header-only atomic inc of the upstream credit counter
    CMD_DONE = 4,         // terminates the writer loop
};

// End-of-stream rides the arrival counter rather than a staging page: a page would need a ring
// slot, hence a credit, which would make closing a stream block at exactly the moment it is
// trying to finish.  Folding it into the same word the arrival count lives in also makes it
// impossible to observe the close ahead of the last batched increment.
constexpr uint32_t EOS_BIAS = 1u << 30;

// How many staged pages accumulate before their arrival increment is sent. Batching matters a lot:
// one increment per token costs roughly a quarter of the link, every fourth about 5%, every eighth
// about 2%, and every thirty-second is free. A stream boundary always forces one out regardless, or
// the downstream reader strands on a partial batch. This is the first knob to sweep once there are
// per-link numbers to sweep against.
constexpr uint32_t BUMP_EVERY = 8;

// How many staged pages one relay pass reads before it waits for any of them.  A page-at-a-time read
// followed immediately by a barrier leaves the reader idle for the whole of each page's DRAM latency;
// issuing the batch first overlaps those latencies.  Measured worth on an 8x4 line at extent 8, one
// link: 2 pages beat 1 by 10%, which is real but only a tenth of the gap to the non-relaying path --
// the reader is on the critical path, but its read latency is not the whole of why.
// This only pays if the reader->writer queue can hold a batch: at queue depth 4 a batch of 2 was
// SLOWER than no batching, because two free pages were rarely available and the pass paid the
// gather cost for nothing.  The kernel caps each batch at the distance to the queue's circular wrap,
// so no alignment between the two is required.
constexpr uint32_t BATCH = 4;

// Longest shortest-path hop count on the combine axis.
constexpr uint32_t max_distance(uint32_t extent, bool is_ring) { return is_ring ? (extent / 2) : (extent - 1); }

// Number of staging levels.  A page staged at a chip has remaining distance in [2, max_distance],
// and occupies the level one below that distance, so levels run [1, max_distance - 1].  Zero
// means no token on this mesh can ever need a relay and the path is inert.
constexpr uint32_t num_levels(uint32_t extent, bool is_ring) {
    const uint32_t m = max_distance(extent, is_ring);
    return m >= 2 ? (m - 1) : 0;
}

// Hops still reachable from a position in one direction.  Symmetric on a ring; on a line the end
// chips reach fewer, which is why the live level set is per-chip and not a single global count.
constexpr uint32_t max_distance_in_dir(uint32_t pos, uint32_t extent, bool is_ring, bool positive) {
    if (is_ring) {
        return extent / 2;
    }
    return positive ? (extent - 1 - pos) : pos;
}

// Whether a chip can ever hold a page bound for a downstream ring at this level, i.e. whether it
// can hold one of remaining distance level + 1.
constexpr bool out_live(uint32_t pos, uint32_t extent, bool is_ring, bool positive, uint32_t level) {
    return (level + 1) <= max_distance_in_dir(pos, extent, is_ring, positive);
}

// Sentinel for "no chip feeds us in this direction", which on a line is true at one end.
constexpr uint32_t NO_POSITION = 0xFFFFFFFFu;

// Position of the chip that sends INTO us travelling in this direction -- the neighbour on the far
// side from the one we send to.
constexpr uint32_t upstream_pos(uint32_t pos, uint32_t extent, bool is_ring, bool positive) {
    if (is_ring) {
        return positive ? (pos + extent - 1) % extent : (pos + 1) % extent;
    }
    if (positive) {
        return pos == 0 ? NO_POSITION : pos - 1;
    }
    return pos + 1 >= extent ? NO_POSITION : pos + 1;
}

// Whether anything can ever arrive in our FIFO for this direction and level.  A level that is not
// in-live needs no drain and no end-of-stream, which is what gives the termination argument its
// base case without any cross-chip dependency.
constexpr bool in_live(uint32_t pos, uint32_t extent, bool is_ring, bool positive, uint32_t level) {
    const uint32_t up = upstream_pos(pos, extent, is_ring, positive);
    return up != NO_POSITION && out_live(up, extent, is_ring, positive, level);
}

// Words in the reader -> writer queue header.  The header is padded to the DRAM alignment, so this
// must fit: 8 words is 32 bytes, which fits Wormhole's 32-byte alignment exactly and Blackhole's
// 64-byte alignment with room to spare.
constexpr uint32_t HDR_WORDS = 8;

// Stride of one staging page, and equally of one L1 ring slot.  Keeping them identical is what
// lets a relay re-read a page as a single contiguous transfer straight into the slot base, with
// payload and tail landing where the writer already expects them.  Padded to the DRAM alignment
// because the relay's read targets L1 and an unaligned destination silently corrupts.
constexpr uint32_t page_bytes(uint32_t aligned_output_page_bytes, uint32_t dram_alignment) {
    const uint32_t raw = aligned_output_page_bytes + tail_bytes();
    return ((raw + dram_alignment - 1) / dram_alignment) * dram_alignment;
}

// First page of the ring owned by one (direction, level, sender core) stream.  Stream identity
// carries across chips -- core s travelling in direction d hands off to core s travelling in
// direction d on the next chip -- so this is the whole of the addressing agreement.
constexpr uint32_t base_page(
    uint32_t dir, uint32_t level, uint32_t core, uint32_t levels, uint32_t num_cores, uint32_t slots) {
    return ((dir * levels + (level - 1)) * num_cores + core) * slots;
}

constexpr uint32_t num_pages(uint32_t levels, uint32_t num_cores, uint32_t slots) {
    return 2u * levels * num_cores * slots;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine::sf
