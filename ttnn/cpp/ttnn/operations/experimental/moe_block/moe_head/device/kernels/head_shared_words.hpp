// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE head's words and marker bits that the host writes and the kernels read: one definition for both. Plain
// constexpr only (no includes but <cstdint>): the kernels include it as is.
#pragma once

#include <cstdint>

// A packed word's field: `bits` wide from bit `shift` (pack: no masking - the host checks the ranges first)
namespace moe_head_words {

struct Field {
    uint32_t shift, bits;
    constexpr uint32_t mask() const { return bits >= 32 ? ~0u : (1u << bits) - 1; }
    constexpr uint32_t limit() const { return 1u << bits; }  // (the values it holds: below this; bits < 32)
    constexpr uint32_t end() const { return shift + bits; }  // (the next field's shift)
    constexpr uint32_t pack(uint32_t value) const { return value << shift; }
    constexpr uint32_t get(uint32_t word) const { return (word >> shift) & mask(); }
};

}  // namespace moe_head_words

// Row blocks' barrier (kernels/row_block.hpp, head_row_blocks.cpp)
namespace moe_head_row_block_words {

// (the RISCs: bit r of a core's RISC mask, r its done word's index - BRISC 0, NCRISC 1, TRISC 0 .. 2 at 2 .. 4)
inline constexpr uint32_t kRiscs = 5;
inline constexpr uint32_t kBriscBit = 1u << 0;
inline constexpr uint32_t kNcriscBit = 1u << 1;
inline constexpr uint32_t kDataMovementBits = kBriscBit | kNcriscBit;  // 0x3
inline constexpr uint32_t kTriscBits = 0x7u << 2;                      // 0x1C: a compute kernel's three TRISCs

// (a core's block words: persistent, zero, 16 B apart - RISC r's done word at kStride x r, then the go word, then
// the coordinator's count)
inline constexpr uint32_t kStride = 16;
inline constexpr uint32_t kGoWord = kStride * kRiscs;          // 80
inline constexpr uint32_t kCountWord = kGoWord + kStride;      // 96
inline constexpr uint32_t kWordsBytes = kCountWord + kStride;  // 112: the host's allocation (kHeadRowBlockWordsBytes)

// (a lead's table [3]: this bit set - one role for every core it runs on; else the count of (core, role) entries)
inline constexpr uint32_t kEveryCoreAlikeShift = 31;
inline constexpr uint32_t kEveryCoreAlike = 1u << kEveryCoreAlikeShift;  // 0x80000000

// (the table's [1], the coordinator: logical x | y << 8 | NoC x << 16 | NoC y << 24; a role's core and the kernel's
// own core key: logical x | y << 8)
inline constexpr moe_head_words::Field kLogicalX{0, 8}, kLogicalY{kLogicalX.end(), 8};
inline constexpr moe_head_words::Field kCoordinatorNocX{kLogicalY.end(), 8},
    kCoordinatorNocY{kCoordinatorNocX.end(), 8};
// (a released core: NoC x | y << 16)
inline constexpr moe_head_words::Field kReleaseNocX{0, 16}, kReleaseNocY{kReleaseNocX.end(), 16};

static_assert(kGoWord == 80 && kCountWord == 96 && kWordsBytes == 112, "the block words' layout");
static_assert(kCoordinatorNocY.shift == 24 && kCoordinatorNocY.end() == 32, "the coordinator word: 4 bytes");
static_assert(kTriscBits == 0x1C, "TRISC 0 .. 2 at bits 2 .. 4");

}  // namespace moe_head_row_block_words

// The shared expert's down passes (shared_expert_compute.cpp, ReduceScatterPlan::pass_bases): a row's pack base word
namespace moe_head_down_pass_words {

// (bit 31: pack into reduce_scatter_hold_cb, else reduce_scatter_final_block_cb; the bits below: the tile offset)
inline constexpr uint32_t kHoldShift = 31;
inline constexpr uint32_t kHold = 1u << kHoldShift;  // 0x80000000
inline constexpr uint32_t kBaseMask = kHold - 1;     // 0x7FFFFFFF

}  // namespace moe_head_down_pass_words

// The fabric port's reduce-scatter words (kernels/reduce_scatter.hpp ReduceScatterUnit / ReduceScatterCore /
// ReduceScatterArgs, head_gather_program.cpp ChipProgram::reduce_scatter_block) and a matmul core's counts word
// (x_gather.hpp tiles_of_block, ReduceScatterPlan::counts_word)
namespace moe_head_reduce_scatter_words {

using moe_head_words::Field;

// (a port's matmul cores at most: its stager's per-core tables - core_read, the hand-offs seen - and a core word's
// index field; head_plan.cpp checks the cores a link)
inline constexpr uint32_t kMaxPortCores = 128;

// (a unit word: entry | block << 8 | half mask << 16 | zone prefix masks << 20 (4 bits a half) | pieces << 28)
inline constexpr Field kUnitEntry{0, 8}, kUnitBlock{kUnitEntry.end(), 8}, kUnitHalfMask{kUnitBlock.end(), 2};
inline constexpr uint32_t kUnitZonePrefixMaskShift = 20, kUnitZonePrefixMaskBits = 4;
constexpr Field unit_zone_prefix_mask(uint32_t half) {
    return Field{kUnitZonePrefixMaskShift + kUnitZonePrefixMaskBits * half, kUnitZonePrefixMaskBits};
}
inline constexpr Field kUnitPieces{unit_zone_prefix_mask(2).shift, 1};
// (a unit's slots word: the zone slot of half 0 | of half 1 << 16)
inline constexpr Field kUnitSlot0{0, 16}, kUnitSlot1{kUnitSlot0.end(), 16};

// (a core word: NoC x | y << 8 | half << 16 | matmul core index << 20 | its pass columns << 27)
inline constexpr Field kCoreNocX{0, 8}, kCoreNocY{kCoreNocX.end(), 8}, kCoreHalf{kCoreNocY.end(), 1};
inline constexpr Field kCoreIndex{20, 7}, kCorePassColumns{kCoreIndex.end(), 5};

// (the router-done word: the matmul cores counting on it | it gates chunk 0's zone credit << 16 | it gates the
// latent's "started" << 17)
inline constexpr Field kRouterDoneCount{0, 16}, kRouterDoneGatesZone{kRouterDoneCount.end(), 1},
    kRouterDoneGatesLatent{kRouterDoneGatesZone.end(), 1};

// (a core's counts word: its tiles of each block, a byte a block - up to kCountsBlocks blocks; past them two (block,
// count) pairs, kCountsNoBlock in an unused pair's block)
inline constexpr uint32_t kCountBits = 8;
inline constexpr uint32_t kCountsBlocks = 32 / kCountBits;  // 4
constexpr Field counts_block(uint32_t block) { return Field{kCountBits * block, kCountBits}; }
inline constexpr Field kPairBlock0{0, kCountBits}, kPairCount0{kPairBlock0.end(), kCountBits},
    kPairBlock1{kPairCount0.end(), kCountBits}, kPairCount1{kPairBlock1.end(), kCountBits};
inline constexpr uint32_t kCountsNoBlock = kPairBlock0.mask();  // 0xFF

// (a matmul core's arrival slot word, per direction - head_engine_program.cpp reduce_scatter_column_dataflow_args,
// shared_expert_column_dataflow.cpp: its threshold | its zone pages << 16)
inline constexpr Field kArrivalThreshold{0, 16}, kArrivalZonePages{kArrivalThreshold.end(), 16};

static_assert(kUnitHalfMask.end() <= kUnitZonePrefixMaskShift && kUnitPieces.shift == 28, "the unit word's layout");
static_assert(kCoreHalf.end() <= kCoreIndex.shift && kCorePassColumns.end() == 32, "the core word's layout");
static_assert(kCoreIndex.limit() == kMaxPortCores, "a core word's index: every port core");
static_assert(kCountsBlocks == 4 && kPairCount1.end() == 32 && kCountsNoBlock == 0xFF, "the counts word's layout");

}  // namespace moe_head_reduce_scatter_words

// The fabric ports' and the router cores' packed words (head_gather_program.cpp latent_block / port_block /
// logits_block, head_engine_program.cpp, head_router_program.cpp; kernels/latent_all_to_all.hpp, x_gather.hpp
// FabricStartedTargets, router_fabric_sends.hpp, router_output_writer.cpp, fabric_port_sender.cpp,
// shared_expert_column_dataflow.cpp)
namespace moe_head_port_words {

using moe_head_words::Field;

// (a core's NoC word, x | y << 8: a logits port's router cores, a "started" target's port, a latent core)
inline constexpr Field kNocX{0, 8}, kNocY{kNocX.end(), 8};
// (a core's NoC xy word, x | y << 16: router core 0, the ports, the matmul cores' logits destinations and upstream
// ports)
inline constexpr Field kNocXyX{0, 16}, kNocXyY{kNocXyX.end(), 16};
// (a chip a port tells "started": its fabric chip id | its mesh id << 16)
inline constexpr Field kFabricChipId{0, 16}, kFabricMeshId{kFabricChipId.end(), 16};
// (a logits unit, an owner chip of my direction: its fabric chip id | my slot in its arrival zone << 16)
inline constexpr Field kLogitsUnitChipId{0, 16}, kLogitsUnitSlot{kLogitsUnitChipId.end(), 16};
// (a latent unit, a destination chip of my direction: its block | my slot in its LATZ << 8 | its fabric chip id << 16)
inline constexpr Field kLatentUnitBlock{0, 8}, kLatentUnitZoneSlot{kLatentUnitBlock.end(), 8},
    kLatentUnitChipId{kLatentUnitZoneSlot.end(), 16};
// (a latent matmul core: NoC x | y << 8 | matmul core index << 16; its rows word, 4 bits a block: first row | rows
// << 2)
inline constexpr Field kLatentCoreIndex{kNocY.end(), 16};
inline constexpr uint32_t kLatentRowsBits = 4;
constexpr Field latent_first_row(uint32_t block) { return Field{kLatentRowsBits * block, 2}; }
constexpr Field latent_num_rows(uint32_t block) { return Field{kLatentRowsBits * block + 2, 2}; }

static_assert(kNocXyY.end() == 32 && kFabricMeshId.end() == 32 && kLogitsUnitSlot.end() == 32, "16-bit halves");
static_assert(kLatentUnitChipId.end() == 32 && kLatentCoreIndex.end() == 32, "the latent unit / core words");

}  // namespace moe_head_port_words

// The routing planner's entries (kernels/routing_planner.hpp: token | expert << 11 | the weight's 15 low bits << 17)
// and its table rows; the host's limits (head_plan.cpp, head_sizing.cpp) follow the field widths
namespace moe_head_routing_words {

using moe_head_words::Field;

// (an entry: a token of the source chip, its expert of my chip, its bf16 gate weight less the sign bit)
inline constexpr uint32_t kTokenBits = 11, kExpertBits = 6, kWeightBits = 15;
inline constexpr Field kEntryToken{0, kTokenBits}, kEntryExpert{kEntryToken.end(), kExpertBits},
    kEntryWeight{kEntryExpert.end(), kWeightBits};
inline constexpr uint32_t kMaxTokens = kEntryToken.limit();           // 2048 tokens a chip
inline constexpr uint32_t kMaxExpertsPerChip = kEntryExpert.limit();  // 64
// (the planner's local-memory counters: per global expert first)
inline constexpr uint32_t kLocalCounterWords = 1024;

// (a table row: word 0 destination | the gate weight's bf16 bits << 16; word 1 the row's header word, source chip |
// token << 8 | expert << 24 - flat_routed_expert's y_row_major.hpp and the combine's combine_formats.hpp read them)
inline constexpr Field kRowDestination{0, 8}, kRowWeight{16, 16};
inline constexpr Field kRowSource{0, 8}, kRowToken{kRowSource.end(), 16}, kRowExpert{kRowToken.end(), 8};

static_assert(kEntryWeight.end() == 32, "an entry: token | expert | weight in one word");
static_assert(kMaxTokens == 2048 && kMaxExpertsPerChip == 64, "an entry's token (11 bits) and expert (6 bits)");
static_assert(kRowExpert.end() == 32 && kTokenBits <= kRowToken.bits, "a table row's header word");

}  // namespace moe_head_routing_words

// The 64 B the head's kernels align their DRAM reads / writes and the routing planner its regions to (align() of
// api/alignment.h)
namespace moe_head_alignment {
inline constexpr uint32_t kAlignBytes = 64;
}  // namespace moe_head_alignment

// A fabric port's packet headers (kernels/dataflow/fabric_port_sender.cpp, head_plan.cpp cores_check,
// head_gather_program.cpp ChipProgram::latent_block): a header a latent destination, at most this many (the sender's
// header array; source not recorded)
namespace moe_head_port_headers {
inline constexpr uint32_t kMaxLatentDestinations = 3;
}  // namespace moe_head_port_headers

// The router kernels' (router_common.hpp) shares: the joint routing rows' four shares of a row tile's 32 token
// columns, 8 columns each: share s builds columns [first_column(s), end_column(s)) (BRISC 0 .. 7, NCRISC 8 .. 15,
// compute MATH 16 .. 23, compute PACK 24 .. 31)
namespace router_shares {
constexpr uint32_t kColumns = 8;
constexpr uint32_t kBrisc = 0, kNcrisc = 1, kMath = 2, kPack = 3, kCount = 4;
constexpr uint32_t first_column(uint32_t share) { return share * kColumns; }
constexpr uint32_t end_column(uint32_t share) { return (share + 1) * kColumns; }
}  // namespace router_shares

// The router kernels' done words: byte offsets from the done-words base (the arena's "words" region, head_sizing.cpp
// router_layout: kCount words, 16 B apart), each a flag word one kernel sets and another waits on (and re-arms)
namespace router_done_words {
constexpr uint32_t kStride = 16;   // (bytes between the words)
constexpr uint32_t kTopKDone = 0;  // compute PACK -> BRISC: the top-K packed (cb_best), the BRISC's rows share can go
// share -> BRISC: that share's rows written (the NCRISC's, MATH's, PACK's: 16, 32, 48)
constexpr uint32_t rows_done(uint32_t share) { return kStride * share; }
constexpr uint32_t kNcriscRowsDone = rows_done(router_shares::kNcrisc), kMathRowsDone = rows_done(router_shares::kMath),
                   kPackRowsDone = rows_done(router_shares::kPack);
constexpr uint32_t kNcriscTopKDone = 64;   // compute PACK -> NCRISC: the top-K packed, the NCRISC's rows share can go
constexpr uint32_t kRowsWritten = 80;      // BRISC -> compute UNPACK: every share of the rows written (route words)
constexpr uint32_t kRouteWordsReady = 96;  // compute UNPACK -> BRISC: the route words in L1
constexpr uint32_t kPartialsIn = 112;      // NCRISC -> compute PACK: the engine's partial logits are in
// (128: no kernel uses it now - an earlier split top-K's partner word; still zeroed with the others)
constexpr uint32_t kLiveRowTiles = 144;  // BRISC -> NCRISC (padded_chunk): the live row tiles (kLiveRowTilesValid)
constexpr uint32_t kCount = 10;          // (the words: kLiveRowTiles the last)
static_assert(kLiveRowTiles == kStride * (kCount - 1), "the input feeder zeroes kCount words, kLiveRowTiles last");
// (kLiveRowTiles' valid bit, 0x80000000: set with the live row tiles; the NCRISC waits for it)
constexpr uint32_t kLiveRowTilesValidShift = 31;
constexpr uint32_t kLiveRowTilesValid = 1u << kLiveRowTilesValidShift;
}  // namespace router_done_words
