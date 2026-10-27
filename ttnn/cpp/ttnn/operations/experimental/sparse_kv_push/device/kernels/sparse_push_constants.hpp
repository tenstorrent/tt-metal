// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the host programs and the kernels of the sparse KV push and its plan.

#pragma once

#include <cstdint>

namespace sparse_push {

// Plan table: uint32 words in kPageBytes pages. Every chip all-gathers every table, so it holds only what the
// owners read.
namespace plan_table {
constexpr uint32_t kPageBytes = 4096;
constexpr uint32_t kPageWords = kPageBytes / sizeof(uint32_t);
constexpr uint32_t kMagic = 0x5350504Cu;

// Header words.
constexpr uint32_t kWordMagic = 0;
constexpr uint32_t kWordNumDevices = 1;
constexpr uint32_t kWordStripeChunk = 2;
constexpr uint32_t kWordActiveBlocks = 3;  // blocks the plan wrote; later bitmap words are stale
constexpr uint32_t kWordSplitBlock = 4;    // first block of the chunk; the prefix part is the blocks before it
constexpr uint32_t kWordPagesPerPacket = 5;
constexpr uint32_t kWordBanks = 6;
constexpr uint32_t kWordHasSplit = 7;  // built with a chunk start; without one, every row is in the prefix part

constexpr uint32_t kOwnerOff = 32;        // [o]: rows needed from owner o
constexpr uint32_t kOwnerPrefixOff = 64;  // [o]: of those, rows before the split block
constexpr uint32_t kBitmapOff = 128;      // one bit per row
constexpr uint32_t kMaxDevices = kOwnerPrefixOff - kOwnerOff;
constexpr uint32_t kHeaderWords = kOwnerPrefixOff + kMaxDevices;  // what the push reads before the bitmap
}  // namespace plan_table

// Push circular buffers.
enum Cb : uint32_t { kData = 0, kMeta = 1, kTable = 2, kScalar = 3, kHeaders = 4 };

constexpr uint32_t kBatch = 4;                 // packets per reader barrier, and in flight per writer flush
constexpr uint32_t kHeaderSlots = 1 + kBatch;  // one for semaphore increments, one per packet in flight
constexpr uint32_t kMetaEntryBytes = 16;       // per packet: destination, first page, rows
// kTable holds the header, then a window of bitmap blocks. A block's bitmap words (at most 64 B) start anywhere in a
// 64 B aligned 128 B window, which bounds stripe_chunk to kMaxStripeChunk.
constexpr uint32_t kWindowBytes = 128;
constexpr uint32_t kBlocksPerFill = 16;  // blocks fetched per bitmap read barrier
constexpr uint32_t kTableCbBytes = plan_table::kBitmapOff * sizeof(uint32_t) + kBlocksPerFill * kWindowBytes;
constexpr uint32_t kMaxStripeChunk = 512;

}  // namespace sparse_push

namespace sparse_push_plan {

constexpr uint32_t kWorkers = 5;            // BRISC, NCRISC and the three TRISCs of a core
constexpr uint32_t kChunkWords = 16;        // bitmap words per dirty chunk
constexpr uint32_t kChunkShift = 9;         // rows per chunk = 16 * 32
constexpr uint32_t kMaxChunksPerCore = 32;  // a core's reduce chunks are the bits of one map word

enum Cb : uint32_t {
    kImage = 0,
    kIdx = 1,
    kDirtyBytes = 2,
    kAllDirty = 3,
    kScratch = 4,
    kRanges = 5,
    kMisc = 6,
    kBuckets = 7,
    kRank16 = 8
};

// Words of the kMisc buffer.
constexpr uint32_t kMiscActiveBlocks = 0;
constexpr uint32_t kMiscChunks = 1;      // this core's reduce chunks: core_idx + k * cores, k < this
constexpr uint32_t kMiscRanges = 2;      // block ranges of range_blocks blocks over the active blocks
constexpr uint32_t kMiscSplit = 16;      // chunk start row, read from DRAM (64 B aligned)
constexpr uint32_t kMiscRowEnd = 32;     // per worker: largest row it scattered + 1
constexpr uint32_t kMiscGo = 48;         // the barrier's go value, multicast by core 0
constexpr uint32_t kMiscTotals = 64;     // per owner: rows this requester needs from it
constexpr uint32_t kMiscBase = 96;       // per owner: compact-buffer run base
constexpr uint32_t kMiscBucketed = 128;  // [from * kWorkers + to]: indices worker `from` handed to worker `to`
constexpr uint32_t kMiscCopies = 160;    // per reduce chunk: dirty copies pushed into scratch
static_assert(kMiscBase - kMiscTotals >= sparse_push::plan_table::kMaxDevices);
constexpr uint32_t misc_words(uint32_t max_chunks) { return kMiscCopies + (max_chunks + 15) / 16 * 16; }

// kAllDirty arrays: [r] this core's map word for reducer r (bit k: chunk r + k * cores is dirty here); [c] core c's
// map word for this core; [c] core c's largest row + 1, gathered on core 0.
enum DirtyArray : uint32_t { kMine = 0, kColumn = 1, kRowEnds = 2, kNumDirtyArrays = 3 };
constexpr uint32_t dirty_array_words(uint32_t cores) { return (cores + 15) / 16 * 16; }

// Semaphores: arrive and go for the cross-core barrier, then step and one done flag per worker.
constexpr uint32_t kSemArrive = 0, kSemGo = 1, kSemStep = 2, kSemDone = 3;
constexpr uint32_t kSemZeroed = kSemDone + kWorkers;  // BRISC: the bitmap and the dirty bytes are zero
constexpr uint32_t kNumSems = kSemZeroed + 1;

}  // namespace sparse_push_plan
