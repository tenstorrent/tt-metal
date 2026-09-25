// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// What both kernel roles and the host agree on: the wire format of a queue entry's routing tail, the sizes
// the compile-time arguments are built from, and the host-side geometry the two argument structs are
// derived from. Each role's arguments live beside its kernel.
//
// Everything the host needs sits behind KERNEL_BUILD, so a kernel translation unit never sees it --
// neither the code nor its includes.

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
    uint32_t pkt_hdr_drain;
    uint32_t drain_sink;
    uint32_t queue;          // queue_depth tokens, filled by the reader and drained by the sender
    uint32_t pkt_hdr_queue;  // one prebuilt packet header per entry
    // The reader's copy of the control tensors and its routing index, read once at startup and then
    // indexed from L1. Nothing on another chip addresses this, so it sits last -- but it is still
    // computed identically everywhere.
    uint32_t control;
};

// The per-chip values that had to be worked out rather than read off the arguments, plus the stream.
struct KernelPlan {
    StreamId stream = 0;
    uint32_t extent = 0;
    uint32_t fwd_pages_per_stream = 0;
    uint32_t queue_filled_addr = 0;
    uint32_t queue_freed_addr = 0;
    uint32_t fwd_arrived_addr = 0;
    // Tile rows the untilizer pool owes this core before it may read a token, and the counter it
    // reports them on. Zero tile rows means the input was already row-major and there is no pool.
    uint32_t untilize_sem_addr = 0;
    uint32_t untilize_tile_rows = 0;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d

namespace dspf2d {
namespace op = ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d;
}
#endif

namespace dspf2d {

// Depth of the reader -> sender L1 queue, in tokens, and the half-queue batch entries move in. Both are
// packed as compile-time args, so this is their one definition. BATCH <= QUEUE_DEPTH/2 is what proves
// the reader cannot claim every entry before publishing any.
//
// BATCH is the DRAM channel count: pages of an interleaved buffer land on consecutive banks, so a batch
// of reads only reaches every bank once it is at least as wide as there are banks.
constexpr uint32_t DRAM_CHANNELS = 8;  // blackhole_140_arch.yaml, `dram:`
constexpr uint32_t BATCH = DRAM_CHANNELS;
constexpr uint32_t QUEUE_DEPTH = 2 * BATCH;

// Pad each prefetched routing-metadata record to 64 B. A DRAM read needs a 64-byte-aligned L1
// destination on Blackhole, which a packed record of num_experts_per_tok uint16 would not keep.
constexpr uint32_t META_PAD_STRIDE = 64;

// Words per assignment in the reader's assignment block: [dst_chip_id, dst_row, split_idx, split_count].
constexpr uint32_t ASSIGNMENT_WORDS = 4;

// The RISCs of one stream core that build the routing index together, one contiguous slice of the
// tokens each: the reader, and the three compute RISCs the op has no tile math for. Pinned by the
// core, not tunable: the compute kernel is built once per TRISC and maps each build to one RISC, and
// a RISC nobody runs would leave the reader waiting forever. The sender is not a RISC: it would need
// the reader's whole geometry in its own arguments for a fifth of the work.
enum IndexRisc : uint32_t { kRiscReader, kRiscUnpack, kRiscMath, kRiscPack, kRiscCount };
constexpr uint32_t INDEX_RISCS = kRiscCount;

// Program semaphores the RISCs hand off on, initialised to zero by the runtime on EVERY launch, which
// is what a word in op-private L1 can never promise. tables_ready goes to 1 when the reader has built
// the tables the RISCs read; each RISC's own semaphore goes to 1 after its count pass and 2 after its
// fill pass.
constexpr uint32_t kSemTablesReady = 0;
constexpr uint32_t index_risc_sem(uint32_t risc) { return 1u + risc; }
constexpr uint32_t INDEX_SEMAPHORES = 1u + INDEX_RISCS;
constexpr uint32_t kRiscCounted = 1;
constexpr uint32_t kRiscFilled = 2;

// Queue entry = token + routing tail. 64 keeps the entry stride DRAM-aligned (14336 + 64 = 64 * 225), which
// lets a fabric write target a (token_size + 64)-byte forwarding page directly.
constexpr uint32_t FORWARDING_METADATA_SIZE = 64;

// The routing tail a queue entry carries after its token, at (entry_base + token_size_bytes). The reader
// fills it, the sender consumes it.
//
// Field order is wire format, not convenience: the first four members are what the next hop needs, so a
// forwarded packet is the token plus a contiguous FWD_EXTRA_BYTES prefix of this struct. `cmd` and
// `this_addr` are recomputed by whichever reader picks the page up and so are not sent.
//
// Dispatch writes each token to TWO tensors at one page index, so both destination addresses travel with
// it; combine needs only one. All uint64_t so a sender needs no sub-word loads.
struct FwdMetadata {
    // First, because the last hop sends the token and these words as ONE scatter packet straight out of
    // the entry: the token is the first chunk, and the second chunk begins at the byte after it, which is
    // this field. Anything placed before it would go to the metadata page instead. The token size is a
    // multiple of the 16-byte NoC write alignment, which is what keeps that second chunk's source aligned
    // with the metadata page it lands on.
    uint32_t meta[3];  // (src chip, token index, top-k slot), the metadata this token carries
    // Makes the addresses below 8-byte aligned, and travels: it is the fourth word of the 16 the last
    // hop writes to the metadata page, so whoever fills an entry zeroes it once.
    uint32_t pad;
    uint64_t final_payload_addr;  // token page address on the FINAL destination chip
    uint64_t final_meta_addr;     // metadata page address on that same chip
    uint64_t dst_chip;            // final destination chip id
    uint64_t cmd;
    uint64_t this_addr;  // the address THIS hop writes to
};

// Wire format between chips rather than a private convenience. The WHOLE 64-byte tail goes on the
// wire, not just the prefix the next hop reads: a forwarded packet's length has to be a whole number
// of NoC beats or the trailing partial beat is dropped silently. Token pages are already 64-byte
// aligned, so sending the full tail keeps the transfer aligned as well.
constexpr uint32_t FWD_EXTRA_BYTES = FORWARDING_METADATA_SIZE;

// Bytes of the tail the next hop consumes. Read by nothing; it exists to pin `cmd`'s offset in the
// static_asserts below, which are what hold the layout the two hops agree on.
constexpr uint32_t FWD_USED_BYTES = 4 * sizeof(uint32_t) + 3 * sizeof(uint64_t);

// ES = expert bucket: one word per global expert id, so resolving a pick is a single indexed load
// rather than a dispatch-table lookup followed by a linear search over the destination chip's
// experts. The word is that expert's bucket index, or this sentinel for an expert of another
// dispatch group. The routing pass tests `bucket >= num_buckets()`, which rejects the sentinel and
// any index off the end at once -- so the only thing a live index must do is stay below it.
constexpr uint32_t BUCKET_NOT_HERE = 0xFFFFFFFFu;

// Words per bucket entry: the token's index on this chip, the page it lands on at the destination,
// and which top-k slot it came from. Sized here rather than in the kernel because the block list
// below reserves the space from it.
constexpr uint32_t entry_words() { return 3u; }

// Bytes the last hop writes to the metadata page: the three words rounded up to a NoC-friendly size.
constexpr uint32_t METADATA_WIRE_BYTES = 16;

// [real_token_count, pad_side], read straight out of DRAM, so a whole 64-byte L1 block rather than
// the two words it holds.
constexpr uint32_t PADDING_CONFIG_BYTES = 64;

// Asserted so that whoever changes this layout has to acknowledge they need some other means of ensuring
// every device runs kernels built from the same metadata format.
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

// One (origin chip, destination chip) term of a stream's forwarding section, narrowed to the share the two
// chips agreed on. Generated identically by the chip that writes the section and the chip that reads it,
// which is what lets the section be dense and carry no addresses.
//
// The expert is NOT here. A chunk is really (origin, destination, expert) for each of the experts the
// destination hosts, but which experts those are lives in expert_dispatch_table, a device tensor; reading
// it back to build the program would put a device sync in the middle of the build and defeat trace
// capture. So each descriptor expands on device into experts_per_chip chunks, in ascending global expert
// id, from the chip -> experts inverse both sides build from the same replicated table.
struct ChunkDescriptor {
    uint32_t origin_row = 0;  // where the tokens started, as a position on the dispatch axis
    uint32_t dst_row = 0;     // the chip hosting the experts
    uint32_t split_idx = 0;
    uint32_t split_count = 1;
};

// --- The stream core's L1 scratch --------------------------------------------------------
//
// One ordered list of blocks, sized here and nowhere else. The host reserves the sum and the kernel
// lays out the offsets; a mismatch between the two overruns into the global semaphores, guarded only by
// an ASSERT that is compiled out on this hardware. Adding a block here is the only way to add one to
// either side.
enum ControlBlock : uint32_t {
    kCbIndices,
    kCbOffsets,
    kCbCounts,
    kCbRegionOffsets,
    kCbTable,
    kCbExpertBucket,
    kCbFirstPage,
    kCbChipExperts,
    kCbRowFill,
    kCbBucketStart,
    kCbEntries,
    kCbPadding,
    kCbInStart,
    kCbOutStart,
    // Per routing index RISC: routed picks per bucket in its token slice, plus its page and entry cursors.
    // Written by that RISC; its counts are read by the later RISCs and the reader after the exchange,
    // its cursors are its own.
    kCbRisc,
    kCbCount
};

// Words one RISC owns in kCbRisc: cnt, next_page and next_entry, one per bucket each.
constexpr uint32_t index_risc_words(uint32_t num_buckets) { return 3u * num_buckets; }

struct ControlGeometry {
    uint32_t seq_len = 0;
    uint32_t indices_pad_stride = 0;
    uint32_t extent = 0;
    uint32_t num_routed_experts = 0;
    uint32_t experts_per_chip = 0;
    uint32_t topk = 0;
    uint32_t num_forward = 0;  // forward_chunks_per_stream(extent), the same for every stream
};

// Chunk-start slots: one per (forward chunk, expert), which is also what the outgoing list expands to.
constexpr uint32_t control_chunk_start_slots(const ControlGeometry& g) { return g.num_forward * g.experts_per_chip; }

constexpr uint32_t control_block_raw_bytes(const ControlGeometry& g, uint32_t block) {
    switch (block) {
        case kCbIndices: return g.seq_len * g.indices_pad_stride;
        case kCbOffsets: return 4u * g.extent * g.num_routed_experts;
        case kCbCounts: return 4u * g.num_routed_experts;
        case kCbRegionOffsets: return 4u * g.num_routed_experts;
        // The dispatch table carries a trailing sentinel column, so a padded token's unguarded lookup
        // lands on it and resolves to "not in this group".
        case kCbTable: return 4u * (g.num_routed_experts + 1u);
        // Indexed by the same expert id the table is, sentinel column included.
        case kCbExpertBucket: return 4u * (g.num_routed_experts + 1u);
        // Keyed by bucket, not by global expert id: the bucket is what the per-pick lookup
        // already yields, and only the experts of this dispatch group have a bucket at all.
        case kCbFirstPage: return 4u * g.extent * g.experts_per_chip;
        case kCbChipExperts: return 4u * g.extent * g.experts_per_chip;
        // One counter per chip on the axis while the inverse is being built. Its own block rather
        // than a corner of another one: the blocks below are indexed by bucket, and a buffer
        // carrying two index domains is how a later edit corrupts the scratch silently.
        case kCbRowFill: return 4u * g.extent;
        // Exclusive prefix sums with a closing total: bucket b's entries run from bucket_start[b] to
        // bucket_start[b + 1], so there is one more of these than there are buckets and the next
        // bucket's start is what bounds the fill.
        case kCbBucketStart: return 4u * (g.extent * g.experts_per_chip + 1u);
        case kCbEntries:
            return 4u * g.seq_len * entry_words() * g.topk;  // one per (token, pick)
        // Always laid out, whether or not a config was supplied: the block list is the one thing host and
        // kernel must agree on term for term, and making a block conditional is how that drifts.
        case kCbPadding: return PADDING_CONFIG_BYTES;
        case kCbInStart: return 4u * control_chunk_start_slots(g);
        case kCbOutStart: return 4u * control_chunk_start_slots(g);
        case kCbRisc: return 4u * INDEX_RISCS * index_risc_words(g.extent * g.experts_per_chip);
        default: return 0u;
    }
}

// Every block starts 64-byte aligned. Six of them are DRAM read destinations, which Blackhole
// requires to be 64-byte aligned. Aligning all of them costs under a kilobyte and makes the property
// hold for whatever block is added next, rather than for the ones someone remembered to check.
constexpr uint32_t control_block_bytes(const ControlGeometry& g, uint32_t block) {
    return (control_block_raw_bytes(g, block) + 63u) & ~63u;
}

constexpr uint32_t scratch_bytes(const ControlGeometry& g) {
    uint32_t total = 0;
    for (uint32_t b = 0; b < kCbCount; b++) {
        total += control_block_bytes(g, b);
    }
    return total;
}

}  // namespace dspf2d
