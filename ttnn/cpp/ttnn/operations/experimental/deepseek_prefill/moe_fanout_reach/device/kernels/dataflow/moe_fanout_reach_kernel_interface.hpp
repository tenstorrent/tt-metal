// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// The contract between moe_fanout_reach's program factory and its kernel: argument positions, the L1
// carve, and the cross-core wiring. Both sides include this header rather than each keeping its own
// copy, because every value here is one a core also uses to address ANOTHER core's L1 -- a host-side
// sum that drifts from the kernel's carve does not fail a shape check, it reads the wrong words.
namespace mfr {

constexpr uint32_t align64(uint32_t x) { return (x + 63u) & ~63u; }

// A DRAM read needs a 64-byte-aligned L1 destination on Blackhole, so a token's index record gets a
// whole number of 64-byte lines to itself rather than being packed against its neighbour.
constexpr uint32_t indices_stride(uint32_t topk) { return align64(topk * 2u); }

// One scan round's vector, padded so every round starts on a 64-byte line: a peer core reads these
// straight out of L1 and the rounds are addressed by multiplying this stride.
constexpr uint32_t scan_row_bytes(uint32_t num_routed_experts) { return align64(num_routed_experts * 4u); }

// Everything the L1 carve is sized from. `tokens_per_core` is the largest token range any core walks,
// not this core's, because one CB configuration covers the whole grid.
struct Geometry {
    uint32_t num_routed_experts = 0;
    uint32_t topk = 0;
    uint32_t tokens_per_core = 0;
    uint32_t rounds = 0;
    // The output tensor's aligned DRAM page, which is also the stride between the two direction rows
    // in L1 so a row can be written out without being repacked.
    uint32_t out_page_bytes = 0;
};

enum Block : uint32_t {
    // This core's token index records, one 64-byte-padded record each.
    kIndices = 0,
    kTable,
    kOffsets,
    // expert -> (destination row << 16) | compacted expert index, or kNotRouted.
    kInfo,
    // The offsets row, re-indexed by compacted expert index.
    kCompactOffsets,
    // rounds + 2 immutable vectors: the local histogram, one per Hillis-Steele round, and the landing
    // area for the predecessor's total. Immutable matters -- a peer reads round r's vector while this
    // core is already writing round r + 1, and a ping-pong pair would hand it half of each.
    kScan,
    // The per-expert allocator this core's token walk advances.
    kAlloc,
    // reach[direction][hop], the two rows this core contributes.
    kOut,
    // A landing slot per child of the tree reduction, so their reads can be issued together. A core
    // never has more children than the scan has rounds -- both count the levels of the same binary
    // tree over the same cores.
    kGather,
    kBlockCount,
};

constexpr uint32_t block_bytes(Block b, const Geometry& g) {
    const uint32_t expert_vector = align64(g.num_routed_experts * 4u);
    switch (b) {
        case kIndices: return g.tokens_per_core * indices_stride(g.topk);
        case kTable: return expert_vector;
        case kOffsets: return expert_vector;
        case kInfo: return expert_vector;
        case kCompactOffsets: return expert_vector;
        case kScan: return (g.rounds + 2u) * scan_row_bytes(g.num_routed_experts);
        case kAlloc: return expert_vector;
        case kOut: return 2u * g.out_page_bytes;
        case kGather: return (g.rounds > 0u ? g.rounds : 1u) * 2u * g.out_page_bytes;
        default: return 0u;
    }
}

constexpr uint32_t block_offset(Block b, const Geometry& g) {
    uint32_t at = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(b); i++) {
        at += align64(block_bytes(static_cast<Block>(i), g));
    }
    return at;
}

// The carve plus one line of slack: the kernel rounds the circular buffer's base up to 64 bytes, and
// every block offset is measured from that rounded base.
constexpr uint32_t carve_bytes(const Geometry& g) { return block_offset(kBlockCount, g) + 64u; }

// An expert that is not in this dispatch group, or whose table entry names a row off the axis. The
// value cannot collide with a real packed word: that would need a destination row of 0xFFFF, which
// the host refuses.
constexpr uint32_t kNotRouted = 0xFFFFFFFFu;
constexpr uint32_t kCompactMask = 0xFFFFu;
constexpr uint32_t kRowShift = 16u;

// No core holds this position, e.g. the Hillis-Steele partner of a core near the start of the chain.
constexpr uint32_t kNoCore = 0xFFFFFFFFu;

// A 64-core grid is the most this op splits a sequence over, so the scan never needs more rounds than
// this and the runtime-argument block has a fixed length.
constexpr uint32_t kMaxRounds = 6;

// One circular buffer holds the whole carve; sub-blocks are addressed by block_offset().
constexpr uint32_t kCbCarve = 0;

enum CtArg : uint32_t {
    kNumRoutedExperts = 0,
    kTopk,
    kExtent,
    // This chip's position on the ring, which is what makes every hop in the table relative to HERE.
    kMyRow,
    kCapacity,
    // extent / 2 + 2 -- hops 1..m, the unused 0, and the terminating zero at m + 1.
    kHops,
    kTokensPerCore,
    kRounds,
    kOutPageBytes,
    // The scan's semaphores are ids 0..rounds, so the tree reduction's sits above them.
    kGatherSemId,
    kCtCount,
};

// TensorAccessorArgs for indices, table, offsets and the output follow the scalars.
constexpr uint32_t kAccessorBase = CtArg::kCtCount;

enum RtArg : uint32_t {
    kIndicesAddr = 0,
    kTableAddr,
    kOffsetsAddr,
    kOutAddr,
    // This core's contiguous token range. Contiguous and in core order is what the scan reproduces.
    kTokStart,
    kTokCount,
    kParentNocX,
    kParentNocY,
    kNumChildren,
    kChildrenBase,
};

// The tree reduction's children, kMaxRounds (x, y) pairs with the unused tail ignored.
constexpr uint32_t kChildWords = 2u * kMaxRounds;
// Per round r in 0..rounds: the core to read round r's vector from, then the core to tell that round
// r's vector is ready. Four words each, always kMaxRounds + 1 rounds long so the block's position
// does not depend on how many children a core has.
constexpr uint32_t kScanBase = RtArg::kChildrenBase + kChildWords;
constexpr uint32_t kScanWordsPerRound = 4u;
constexpr uint32_t kRtCount = kScanBase + (kMaxRounds + 1u) * kScanWordsPerRound;

}  // namespace mfr
