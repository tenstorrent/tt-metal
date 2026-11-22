// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The sparse 2D dispatch's packed words and argument layouts that the host writes and the kernels read: one definition
// for the host (plan, factory, device operation), the port reader and the port sender. Plain constexpr only (no
// includes but <cstdint>): the kernels include it as is.
//
// The route word's layout (route_word) is also written by the MoE head's router (moe_head/device/kernels/
// router_common.hpp, route_words::put), and the ring tie bit (kRingTieBit) is the one its tie_break reads. That header
// is not included here (it is the head's); keep the two in step by hand.
#pragma once

#include <cstdint>

namespace sparse_dispatch_2d_layout {

// A packed word's field: `bits` wide from bit `shift` (pack: no masking - the host checks the ranges first)
struct Field {
    uint32_t shift, bits;
    constexpr uint32_t mask() const { return bits >= 32 ? ~0u : (1u << bits) - 1; }
    constexpr uint32_t limit() const { return 1u << bits; }  // (the values it holds: below this; bits < 32)
    constexpr uint32_t end() const { return shift + bits; }  // (the next field's shift)
    constexpr uint32_t pack(uint32_t value) const { return value << shift; }
    constexpr uint32_t get(uint32_t word) const { return (word >> shift) & mask(); }
};

// ---- A manifest entry: 16 bytes (the metadata page a reader hands its sender has the same shape) ----
inline constexpr uint32_t kEntryWords = 4;
inline constexpr uint32_t kEntryBytes = kEntryWords * 4;
// Word 0 of a row's entry: its slot in xbuf, the hops it still has to go on the port's axis (the receiving port
// forwards it with one less), the END generation count of the hops taken (a row's entry travels one generation a hop)
// and the dimension order of its tree (0: X first). Word 1 is the destination mask.
inline constexpr Field kSlot{0, 16}, kHops{kSlot.end(), 4}, kGeneration{kHops.end(), 4}, kOrder{kGeneration.end(), 1};
// An END entry (no row) has word 0 = END | generation << kGeneration.shift; a metadata page of SKIP is filler.
inline constexpr uint32_t kEndBit = 1u << 31, kSkipBit = 1u << 30;

constexpr uint32_t entry_word0(uint32_t slot, uint32_t hops, uint32_t generation, uint32_t order) {
    return kSlot.pack(slot) | kHops.pack(hops) | kGeneration.pack(generation) | kOrder.pack(order);
}
constexpr uint32_t end_word0(uint32_t generation) { return kEndBit | kGeneration.pack(generation); }

// What the entry fields can hold: the host checks a mesh against these.
inline constexpr uint32_t kMaxSlots = kSlot.limit();            // N * S slots
inline constexpr uint32_t kMaxGeneration = kGeneration.mask();  // GMAX
inline constexpr uint32_t kMaxAxisHops = kHops.mask();          // hops along one line (an axis of at most 16 chips)

// ---- ring_offs: ring 1's first entry in a port core's manifest | ring 2's << 16 (ring 0 starts at 0) ----
inline constexpr uint32_t kRingOffsetBits = 16;
constexpr uint32_t pack_ring_offsets(uint32_t ring1, uint32_t ring2) {
    return Field{0, kRingOffsetBits}.pack(ring1) | Field{kRingOffsetBits, kRingOffsetBits}.pack(ring2);
}
constexpr uint32_t ring_offset(uint32_t ring_offs, uint32_t ring) {
    return ring == 0 ? 0 : Field{kRingOffsetBits * (ring - 1), kRingOffsetBits}.get(ring_offs);
}

// ---- A span byte: the hops forward and backward on an axis from a position (mesh_routing.hpp span_entry) ----
inline constexpr Field kSpanForward{0, 4}, kSpanBackward{kSpanForward.end(), 4};
constexpr uint32_t span_byte(uint32_t forward, uint32_t backward) {
    return kSpanForward.pack(forward) | kSpanBackward.pack(backward);
}

// ---- A row's route word(s) ----
// The flags (its dimension order, and which of the fabric links it takes) and the destination mask. With one
// word per row (up to 16 chips): mask | flags << 16. With two: word 0 the mask, word 1 the flags.
inline constexpr Field kFlagOrder{0, 1}, kFlagLink{kFlagOrder.end(), 1};
inline constexpr Field kMaskInWord{0, 16}, kFlagsInWord{kMaskInWord.end(), 16};

// ---- The tie-break hash ----
// A row's free choices (which way round a ring, X or Y first at equal links, which link) are made per row from row
// index x Knuth's multiplicative constant (2^32 / golden ratio): a fixed choice that spreads the rows evenly over both
// options. A product's bit b depends on the row's bits 0 .. b, so the high bits mix all of them, and one bit a choice
// keeps the choices independent. The ring tie is bit 30 here; the head takes bit 31 for the order and bit 29 for the
// link.
inline constexpr uint32_t kHashMultiplier = 2654435761u;
inline constexpr uint32_t kRingTieBit = 30;

// ---- Runtime args of the port cores (the kernels read them by these indices from the core's args block) ----
// A port core of another chip only: the reader's kDir or the sender's kReadyX is this, and the kernel returns.
inline constexpr uint32_t kIdlePort = 0xFFFFFFFFu;

namespace reader_arg {
// out, plan, words (the port's semaphore words: addresses), then the values the factory appends in this order
inline constexpr uint32_t kOut = 0, kPlan = 1, kWords = 2, kRingOffsets = 3, kRingMask = 4, kMyRow = 5, kMyCol = 6,
                          kDir = 7, kLink = 8, kCount = 9;
}  // namespace reader_arg

namespace sender_arg {
// out, words (addresses), then the tail the factory builds in this order, then the fabric connection's values
inline constexpr uint32_t kOut = 0, kWords = 1, kReadyX = 2, kReadyY = 3, kDstMeshId = 4, kDstChipId = 5,
                          kRingOffsets = 6;
// per output stream (0: continue, 1 / 2: turn + / -): on, x, y, ring
inline constexpr uint32_t kStreams = 7, kStreamCount = 3, kStreamWords = 4;
inline constexpr uint32_t kStreamOn = 0, kStreamX = 1, kStreamY = 2, kStreamRing = 3;
inline constexpr uint32_t kNeighbourRow = kStreams + kStreamCount * kStreamWords, kNeighbourCol = kNeighbourRow + 1,
                          kDir = kNeighbourCol + 1;
inline constexpr uint32_t kFabric = kDir + 1;  // the fabric connection's args start here (the same on every port core)
}  // namespace sender_arg
// the fabric connection's args (TENSIX VC0: eth channel, teardown and buffer-index semaphores); the factory keeps them
// as the sender's runtime args from index 0 as well (args_table.hpp, kept args), where build_from_args reads them
inline constexpr uint32_t kFabricArgs = 3;

}  // namespace sparse_dispatch_2d_layout
