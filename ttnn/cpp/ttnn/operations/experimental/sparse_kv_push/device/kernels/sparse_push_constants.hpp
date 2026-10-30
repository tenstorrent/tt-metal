// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Constants and layouts shared by the host programs and the kernels of the sparse KV push and its plan:
//   sparse_push::plan_table   the plan table, built by the plan, read by the push;
//   sparse_push               the compact-buffer layout both agree on, and the push's own constants;
//   sparse_push_plan          the plan program's own constants.

namespace sparse_push {

// ---------------------------------------------------------------------------------------------------------------------
// Plan table: one per chip, in uint32 words, in kPageBytes pages. Every chip all-gathers every table, so it holds only
// what the owners read.
//
//   words [0, 32)          header (kWord*)
//   words [32, 64)         [owner]: rows the requester needs from the owner
//   words [64, 128)        [owner]: of those, rows before the split block (the prefix, PushPart::kPrefix)
//   words [128, ...)       the bitmap: one bit per cache row, kRowsPerWord rows per word, block by block
// ---------------------------------------------------------------------------------------------------------------------
namespace plan_table {
constexpr uint32_t kPageBytes = 4096;
constexpr uint32_t kPageWords = kPageBytes / sizeof(uint32_t);
constexpr uint32_t kMagic = 0x5350504Cu;
constexpr uint32_t kMaxDevices = 32;   // owners per table
constexpr uint32_t kRowsPerWord = 32;  // bitmap rows per word (bits in a uint32)

// Header words.
constexpr uint32_t kWordMagic = 0;
constexpr uint32_t kWordNumDevices = 1;
constexpr uint32_t kWordStripeChunk = 2;
constexpr uint32_t kWordActiveBlocks = 3;  // blocks the plan wrote; later bitmap words are stale
constexpr uint32_t kWordSplitBlock = 4;    // the block holding the chunk start; the prefix is the blocks before it
constexpr uint32_t kWordPagesPerPacket = 5;
constexpr uint32_t kWordBanks = 6;
constexpr uint32_t kWordHasSplit = 7;  // built with a chunk start; without one, every row is in the prefix

// Word offsets of the sections.
constexpr uint32_t kOwnerOff = 32;
constexpr uint32_t kOwnerPrefixOff = kOwnerOff + kMaxDevices;
constexpr uint32_t kBitmapOff = 128;
constexpr uint32_t kHeaderWords = kOwnerPrefixOff + kMaxDevices;  // what the push reads before the bitmap
static_assert(kHeaderWords <= kBitmapOff);
}  // namespace plan_table

// ---------------------------------------------------------------------------------------------------------------------
// Compact buffer: a requester's buffer of pushed rows, one run per owner in owner order. The plan rewrites the top-k
// indices with compact_page and the push writes rows with it, so they agree by construction.
// ---------------------------------------------------------------------------------------------------------------------

// Pages an owner's run of `rows` rows takes: whole units of pages_per_packet * banks, so every run starts on a unit,
// as compact_page needs.
template <uint32_t pages_per_packet, uint32_t banks>
constexpr uint32_t padded_run_pages(uint32_t rows) {
    constexpr uint32_t unit = pages_per_packet * banks;
    return (rows + unit - 1) / unit * unit;
}

// Page of the row with rank `rank` (0-based, in row order) in an owner's run starting at page run_base. The buffer is
// DRAM-interleaved: page i lives in bank i % banks. Ranks go in packets of pages_per_packet; packet j lands in bank
// j % banks with its rows banks pages apart, i.e. at consecutive addresses in that bank, so the writer sends each
// packet as one contiguous write.
template <uint32_t pages_per_packet, uint32_t banks>
constexpr uint32_t compact_page(uint32_t run_base, uint32_t rank) {
    const uint32_t packet = rank / pages_per_packet;
    const uint32_t row_in_packet = rank % pages_per_packet;
    return run_base + (packet % banks) + (packet / banks) * pages_per_packet * banks + row_in_packet * banks;
}

// ---------------------------------------------------------------------------------------------------------------------
// Push: the reader and writer kernels of each sender core.
// ---------------------------------------------------------------------------------------------------------------------

// Which of a requester's needed rows one push sends. The prefix is the rows of the blocks before the one holding the
// plan's chunk start: rows earlier chunks wrote, so it can be pushed before this chunk's KV is written.
enum class PushPart : uint32_t {
    kAll = 0,     // every needed row
    kPrefix = 1,  // the prefix
    kRest = 2,    // every needed row after the prefix
};

// Circular buffers.
enum Cb : uint32_t {
    kData = 0,     // packet rows, reader -> writer
    kMeta = 1,     // a PacketMeta per packet, reader -> writer
    kTable = 2,    // reader: the requester's table header, then its bitmap windows
    kUserId = 3,   // reader: the user id read for the trace-safe cache slot
    kHeaders = 4,  // writer: fabric packet headers
};

// Reader -> writer, one kMetaEntryBytes entry per packet in kMeta. rows == 0 ends the writer.
struct PacketMeta {
    uint32_t requester;   // chip whose compact buffer the packet goes to
    uint32_t first_page;  // compact-buffer page of its first row (see compact_page)
    uint32_t rows;
};
constexpr uint32_t kMetaEntryBytes = 16;  // a PacketMeta, padded to the 16 B L1 alignment
static_assert(sizeof(PacketMeta) <= kMetaEntryBytes);

constexpr uint32_t kBatch = 4;                 // packets per reader barrier, and in flight per writer flush
constexpr uint32_t kHeaderSlots = 1 + kBatch;  // one for semaphore increments, one per packet in flight

// Bitmap windows: the reader fetches each block's bitmap words (at most 64 B) as one kWindowBytes read from the
// kDramReadAlign boundary at or below them, so they can start anywhere in it. That bounds stripe_chunk.
constexpr uint32_t kDramReadAlign = 64;  // DRAM reads start on this alignment
constexpr uint32_t kWindowBytes = 2 * kDramReadAlign;
constexpr uint32_t kMaxStripeChunk = (kWindowBytes - kDramReadAlign) * 8;  // rows whose bitmap is at most 64 B
constexpr uint32_t kBlocksPerBatch = 16;  // blocks whose bitmaps are fetched behind one read barrier
constexpr uint32_t kTableCbBytes = plan_table::kBitmapOff * sizeof(uint32_t) + kBlocksPerBatch * kWindowBytes;

}  // namespace sparse_push

// ---------------------------------------------------------------------------------------------------------------------
// Plan: the sparse_push_plan program (overview in kernels/sparse_push_plan.cpp).
// ---------------------------------------------------------------------------------------------------------------------
namespace sparse_push_plan {

constexpr uint32_t kWorkers = 5;      // BRISC, NCRISC and the three TRISCs of a core
constexpr uint32_t kChunkWords = 16;  // bitmap words per dirty chunk, the unit of the reduce step
constexpr uint32_t kChunkRows = kChunkWords * sparse_push::plan_table::kRowsPerWord;
constexpr uint32_t kMaxChunksPerCore = 32;  // a core's reduce chunks are the bits of one map word
// Words between a rank range's per-owner counts: num_devices, padded so each range is whole 16 B L1 transfers.
constexpr uint32_t owner_stride(uint32_t num_devices) { return (num_devices + 3) / 4 * 4; }

// Scratch arrays are padded to whole kArrayPadWords (64 B), so each starts 64 B aligned.
constexpr uint32_t kArrayPadWords = 16;
constexpr uint32_t pad_words(uint32_t words) { return (words + kArrayPadWords - 1) / kArrayPadWords * kArrayPadWords; }

// Circular buffers (L1 buffers of every plan core).
enum Cb : uint32_t {
    kTableImage = 0,   // this core's copy of the whole plan table
    kIdx = 1,          // this core's indices, rewritten in place
    kDirtyChunks = 2,  // one flag byte per chunk this core touched
    kDirtyMaps = 3,    // the DirtyMap arrays below
    kScratch = 4,      // as a reducer: the chunk copies the other cores pushed
    kPartials = 5,     // [range][owner]: the owner's needed rows in the range
    kMisc = 6,         // single words shared by the workers (kMisc* below)
    kBuckets = 7,      // step 1's buckets, then step 2's copy lists
    kWordRanks = 8,    // [word]: the owner's needed rows before the word, within its range
};

// kDirtyMaps: three arrays of dirty_array_words(cores) words.
enum DirtyMap : uint32_t {
    kMyDirtyMap = 0,  // [reducer]: bit k set if this core dirtied the reducer's k-th chunk
    kColumn = 1,      // [core]: that core's map word for this core, added in by the core
    kRowEnds = 2,     // [core]: the core's largest needed row + 1, gathered on core 0
    kNumDirtyMaps = 3,
};
constexpr uint32_t dirty_array_words(uint32_t cores) { return pad_words(cores); }

// Words of kMisc. Each group starts on a 16-word (64 B) boundary, so it can be a NoC transfer source or target.
constexpr uint32_t kMiscActiveBlocks = 0;  // BRISC -> workers: blocks with any needed row
constexpr uint32_t kMiscChunks = 1;        // BRISC -> workers: how many reduce chunks this core owns
constexpr uint32_t kMiscRanges = 2;        // BRISC -> workers: rank ranges over the active blocks
constexpr uint32_t kMiscSplit = 16;        // the chunk start row, read from DRAM
constexpr uint32_t kMiscRowEnd = 32;       // [worker]: the largest row it scattered + 1
constexpr uint32_t kMiscGo = 48;           // core 0: the exchange's go value, multicast to every core
constexpr uint32_t kMiscTotals = 64;       // [owner]: rows this chip needs from the owner
constexpr uint32_t kMiscBase = 96;         // [owner]: the owner's compact-buffer run base
constexpr uint32_t kMiscBucketed = 128;    // [from * kWorkers + to]: rows worker `from` handed to worker `to`
constexpr uint32_t kMiscCopies = 160;      // [my_chunk]: how many cores pushed a copy of this core's chunk
static_assert(kMiscBase - kMiscTotals >= sparse_push::plan_table::kMaxDevices);
static_assert(kMiscBucketed - kMiscBase >= sparse_push::plan_table::kMaxDevices);
static_assert(kMiscCopies - kMiscBucketed >= kWorkers * kWorkers);
constexpr uint32_t misc_words(uint32_t max_chunks) { return kMiscCopies + pad_words(max_chunks); }

// Semaphores.
constexpr uint32_t kSemArrive = 0;                    // core 0: arrivals at the cross-core exchange
constexpr uint32_t kSemGo = 1;                        // every core: the exchange's go value from core 0
constexpr uint32_t kSemStep = 2;                      // BRISC -> workers: the step it has started (Step)
constexpr uint32_t kSemDone = 3;                      // [worker], kWorkers of them: the worker's progress (Progress)
constexpr uint32_t kSemZeroed = kSemDone + kWorkers;  // BRISC -> workers: the bitmap and the dirty chunks are zero
constexpr uint32_t kNumSems = kSemZeroed + 1;

}  // namespace sparse_push_plan
