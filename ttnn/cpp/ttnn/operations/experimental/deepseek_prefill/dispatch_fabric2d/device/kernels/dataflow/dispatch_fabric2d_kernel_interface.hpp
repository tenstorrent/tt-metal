// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Definitions the kernels and the host must agree on: the fwd_meta wire format, the sizes the compile-time
// arguments are built from, and the scratch layout. Host-only declarations sit behind KERNEL_BUILD.
//
// A stream is one link direction, served by a reader and a sender on one core. The reader passes tokens
// to the sender through the TokenQueue, an L1 queue whose entries each hold one token and its fwd_meta.

#include <cstddef>
#include <cstdint>

#ifndef KERNEL_BUILD
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt_stl/assert.hpp>

#include "../../dispatch_fabric2d_placement.hpp"
#include "../../dispatch_fabric2d_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

struct L1Layout {
    uint32_t pkt_hdr_signal;
    uint32_t drain_sink;
    uint32_t queue;          // queue_depth tokens, filled by the reader and drained by the sender
    uint32_t pkt_hdr_queue;  // one prebuilt packet header per entry
    // The reader's scratch: its copy of the control tensors and its routing index. No other chip
    // addresses it, so it goes last.
    uint32_t scratch;
};

// Per-chip values the host derives, plus the stream.
struct KernelPlan {
    StreamId stream = 0;
    uint32_t extent = 0;
    uint32_t fwd_pages_per_stream = 0;
    uint32_t queue_filled_addr = 0;
    uint32_t queue_freed_addr = 0;
    uint32_t fwd_arrived_addr = 0;
    // Tile rows the untilizer pool must deliver to this core before it may read a token, and the counter
    // it signals them on. Zero tile rows means the input is row-major and there is no pool.
    uint32_t untilize_sem_addr = 0;
    uint32_t untilize_tile_rows = 0;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d

namespace dspf2d {
namespace op = ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d;
}
#endif

namespace dspf2d {

// TokenQueue depth and the batch size entries move in, both in tokens. BATCH <= QUEUE_DEPTH avoids
// deadlock; BATCH <= QUEUE_DEPTH / 2 lets the sender drain one batch while the reader fills the next.
//
// BATCH is the DRAM channel count: interleaved pages sit on consecutive banks, so a batch this wide
// reads from every bank.
constexpr uint32_t DRAM_CHANNELS = 8;  // blackhole_140_arch.yaml, `dram:`
constexpr uint32_t BATCH = DRAM_CHANNELS;
constexpr uint32_t QUEUE_DEPTH = 2 * BATCH;

// Pad each prefetched routing-metadata record to 64 B. A DRAM read needs a 64-byte-aligned L1
// destination on Blackhole, which a packed record of num_experts_per_tok uint16 would not keep.
constexpr uint32_t META_PAD_STRIDE = 64;

// Words per assignment in the reader's assignment block: [dst_chip_id, dst_row, split_idx, split_count].
constexpr uint32_t ASSIGNMENT_WORDS = 4;

// Words per chunk descriptor in the reader's in/out chunk blocks: [origin_row, dst_row, split_idx, split_count].
constexpr uint32_t CHUNK_DESCRIPTOR_WORDS = 4;

// The RISCs of a stream core that build the routing index, each over a contiguous slice of the tokens:
// the reader and the three compute RISCs, which have no tile math in this op. The set is fixed: the
// compute kernel is built once per TRISC, and the reader waits for every RISC to finish.
enum IndexRisc : uint32_t { kRiscReader, kRiscUnpack, kRiscMath, kRiscPack, kRiscCount };
constexpr uint32_t INDEX_RISCS = kRiscCount;

// Program semaphores for the routing-index handoff. The runtime zeroes them on each launch, which
// op-private L1 does not guarantee. kSemTablesReady goes to 1 once the reader has built the tables the
// RISCs read; each RISC's own semaphore goes to 1 after its count pass and 2 after its fill pass.
constexpr uint32_t kSemTablesReady = 0;
constexpr uint32_t index_risc_sem(uint32_t risc) { return 1u + risc; }
constexpr uint32_t INDEX_SEMAPHORES = 1u + INDEX_RISCS;
constexpr uint32_t kRiscCounted = 1;
constexpr uint32_t kRiscFilled = 2;

// Size of the fwd_meta after each token in a queue entry. The token size is the output page's aligned
// size, a multiple of the DRAM alignment, so adding 64 keeps every entry DRAM-aligned and makes an entry
// the same size as a forwarding page.
constexpr uint32_t FORWARDING_METADATA_SIZE = 64;

// fwd_meta: the routing data a queue entry carries after its token, at entry_base + token_size_bytes.
// The reader fills it and the sender consumes it. The field order is wire format.
//
// A forward hop sends the whole struct (FWD_EXTRA_BYTES). The next hop's reader uses the first
// FWD_USED_BYTES and overwrites `cmd` and `this_addr` for its own send.
//
// Each token lands in two tensors at one page index, so both addresses travel with it. The addresses are
// uint64_t so the sender needs no sub-word loads.
struct FwdMetadata {
    // First, because the last hop sends the token and these words as one scatter packet straight out of
    // the entry: the second chunk starts at the byte after the token. The token size is a multiple of the
    // 16-byte NoC write alignment, so that chunk's source is aligned like the metadata page it lands on.
    uint32_t meta[3];  // (src chip, token index, topk index), the metadata this token carries
    // Aligns the addresses below to 8 bytes. It is also the fourth word of the METADATA_WIRE_BYTES the
    // last hop writes to the metadata page, so the reader zeroes it.
    uint32_t pad;
    uint64_t final_payload_addr;  // token page address on the destination chip
    uint64_t final_meta_addr;     // metadata page address on the destination chip
    uint64_t dst_chip;            // destination chip id
    uint64_t cmd;
    uint64_t this_addr;  // the address this hop writes to
};

// Bytes of fwd_meta a forward hop sends: all of it, so a forwarded packet fills exactly one forwarding
// page and its length stays a multiple of 64 B.
constexpr uint32_t FWD_EXTRA_BYTES = FORWARDING_METADATA_SIZE;

// Bytes of fwd_meta the next hop reads. Used only by the static_asserts below, which pin the layout.
constexpr uint32_t FWD_USED_BYTES = 4 * sizeof(uint32_t) + 3 * sizeof(uint64_t);

// kBlkExpertBucket holds one word per global expert id: that expert's bucket index, or BUCKET_NOT_HERE
// for an expert outside this dispatch group, so resolving a top-k choice is one indexed load. The routing
// pass tests `bucket >= num_buckets()`, which rejects the sentinel and any out-of-range index together.
constexpr uint32_t BUCKET_NOT_HERE = 0xFFFFFFFFu;

// Words per record in kBlkRecords: the token's index on this chip, its page on the destination, and its
// topk index.
constexpr uint32_t record_words() { return 3u; }

// Bytes the last hop writes to the metadata page: the three meta words plus pad, padded to the 16-byte
// NoC write alignment.
constexpr uint32_t METADATA_WIRE_BYTES = 16;

// [real_token_count, pad_side]. Read from DRAM, so it gets a whole 64-byte L1 block for two words.
constexpr uint32_t PADDING_CONFIG_BYTES = 64;

// Chips exchange fwd_meta, so every chip must run kernels built from this layout.
static_assert(sizeof(FwdMetadata) <= FORWARDING_METADATA_SIZE);
static_assert(offsetof(FwdMetadata, meta) == 0);
static_assert(offsetof(FwdMetadata, final_payload_addr) == 4 * sizeof(uint32_t));
static_assert(offsetof(FwdMetadata, cmd) == FWD_USED_BYTES);
static_assert(FWD_USED_BYTES == 40);
static_assert(FWD_EXTRA_BYTES % 64 == 0);
static_assert(METADATA_WIRE_BYTES <= FORWARDING_METADATA_SIZE);

constexpr uint64_t CMD_END = 0;          // end of stream; the entry carries no token
constexpr uint64_t CMD_FINAL_WRITE = 1;  // this hop is the last: write payload and metadata to their pages
constexpr uint64_t CMD_FORWARD = 2;      // push one page further along the stream
constexpr uint64_t CMD_FORWARD_END = 3;  // as CMD_FORWARD, and the last page of its chunk

// One (origin chip, destination chip) term of a stream's fwd_section, narrowed to the share the two chips
// agreed on. The chip that writes the section and the chip that reads it generate the same list, so the
// section is dense and carries no addresses.
//
// Each descriptor expands on device into experts_per_chip chunks, one per expert the destination hosts,
// in ascending global expert id. The experts come from expert_dispatch_table, a device tensor, so the
// host does not know them when it builds the program.
struct ChunkDescriptor {
    uint32_t origin_row = 0;  // where the tokens started, as a position on the dispatch axis
    uint32_t dst_row = 0;     // the chip hosting the experts
    uint32_t split_idx = 0;
    uint32_t split_count = 1;
};
static_assert(sizeof(ChunkDescriptor) == 4 * CHUNK_DESCRIPTOR_WORDS, "one uint32_t per descriptor word");

// --- The stream core's L1 scratch --------------------------------------------------------
//
// One ordered list of blocks, sized only here. The host reserves the sum and the kernel lays out the
// offsets from the same list. A mismatch overruns into the global semaphores, and the kernel's ASSERT
// against that compiles in only when the watcher is enabled.
enum ScratchBlock : uint32_t {
    kBlkIndices,
    kBlkOffsets,
    kBlkCounts,
    kBlkRegionOffsets,
    kBlkTable,
    kBlkExpertBucket,
    kBlkFirstPage,
    kBlkChipExperts,
    kBlkRowFill,
    kBlkBucketStart,
    kBlkRecords,
    kBlkPadding,
    kBlkInStart,
    kBlkOutStart,
    // Per routing-index RISC: routed choices per bucket over its token slice, plus its next_page and
    // next_record cursors. The later RISCs and the reader read its counts after the exchange; the cursors
    // are private to the RISC.
    kBlkRisc,
    kBlkCount
};

// Words one RISC owns in kBlkRisc: cnt, next_page and next_record, one per bucket each.
constexpr uint32_t index_risc_words(uint32_t num_buckets) { return 3u * num_buckets; }

struct ScratchGeometry {
    uint32_t seq_len = 0;
    uint32_t indices_pad_stride = 0;
    uint32_t extent = 0;
    uint32_t num_routed_experts = 0;
    uint32_t experts_per_chip = 0;
    uint32_t topk = 0;
    uint32_t num_forward = 0;  // forward_chunks_per_stream(extent), the same for every stream
};

// Chunk starts: one per (forward chunk, expert), which is also what the outgoing list expands to.
constexpr uint32_t chunk_start_count(const ScratchGeometry& g) { return g.num_forward * g.experts_per_chip; }

constexpr uint32_t scratch_block_raw_bytes(const ScratchGeometry& g, uint32_t block) {
    switch (block) {
        case kBlkIndices: return g.seq_len * g.indices_pad_stride;
        case kBlkOffsets: return 4u * g.extent * g.num_routed_experts;
        case kBlkCounts: return 4u * g.num_routed_experts;
        case kBlkRegionOffsets: return 4u * g.num_routed_experts;
        // A trailing sentinel column, so a padded token's unguarded lookup reads "not in this group".
        case kBlkTable: return 4u * (g.num_routed_experts + 1u);
        // Indexed like kBlkTable, sentinel column included.
        case kBlkExpertBucket: return 4u * (g.num_routed_experts + 1u);
        // Indexed by bucket; only this group's experts have one.
        case kBlkFirstPage: return 4u * g.extent * g.experts_per_chip;
        case kBlkChipExperts: return 4u * g.extent * g.experts_per_chip;
        // One counter per chip on the axis, used while the chip -> experts inverse is built. A separate
        // block because the blocks below are indexed by bucket.
        case kBlkRowFill: return 4u * g.extent;
        // Exclusive prefix sums plus a closing total: bucket b's records run from bucket_start[b] to
        // bucket_start[b + 1].
        case kBlkBucketStart: return 4u * (g.extent * g.experts_per_chip + 1u);
        case kBlkRecords:
            return 4u * g.seq_len * record_words() * g.topk;  // one per (token, top-k choice)
        // Reserved even with no padding config, so the host and kernel lists never differ.
        case kBlkPadding: return PADDING_CONFIG_BYTES;
        case kBlkInStart: return 4u * chunk_start_count(g);
        case kBlkOutStart: return 4u * chunk_start_count(g);
        case kBlkRisc: return 4u * INDEX_RISCS * index_risc_words(g.extent * g.experts_per_chip);
        default: return 0u;
    }
}

// Every block starts 64-byte aligned. Several are DRAM read destinations, which Blackhole requires to be
// 64-byte aligned; aligning all of them also covers blocks added later.
constexpr uint32_t scratch_block_bytes(const ScratchGeometry& g, uint32_t block) {
    return (scratch_block_raw_bytes(g, block) + 63u) & ~63u;
}

constexpr uint32_t scratch_bytes(const ScratchGeometry& g) {
    uint32_t total = 0;
    for (uint32_t b = 0; b < kBlkCount; b++) {
        total += scratch_block_bytes(g, b);
    }
    return total;
}

}  // namespace dspf2d
