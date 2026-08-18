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
// page.  All four cross the fabric: the first two are the routing state the next hop needs, the
// last two let the receiver reject a page it should not be reading.
constexpr uint32_t TAIL_OUTPUT_PAGE_IDX = 0;
constexpr uint32_t TAIL_FINAL_DST_CHIP = 1;
constexpr uint32_t TAIL_LEVEL = 2;
constexpr uint32_t TAIL_SEQ = 3;
constexpr uint32_t TAIL_WORDS = 4;

constexpr uint32_t tail_bytes() { return TAIL_WORDS * sizeof(uint32_t); }

// A live page always carries level >= 1, so a zeroed or never-written staging page fails the
// level check.  That removes the need for a separate magic word in the 16 wire bytes.
constexpr uint32_t LEVEL_INVALID = 0;

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
