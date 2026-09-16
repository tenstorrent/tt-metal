// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// What both kernel roles and the host agree on: the wire format of a ring slot's routing tail, the sizes
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
    uint32_t ring;          // num_l1_slots tokens, filled by the reader and drained by the sender
    uint32_t pkt_hdr_ring;  // headers_per_slot(fanout) prebuilt headers per slot
    // fanout: per (slot, destination) delivery records and the metadata words they point at. Written
    // by the reader, read by the sender, so they live outside the reader's control carve, which the
    // sender knows nothing about.
    uint32_t mc_delivery;
    uint32_t mc_meta;
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
    uint32_t ring_filled_addr = 0;
    uint32_t ring_freed_addr = 0;
    uint32_t fwd_arrived_addr = 0;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d

namespace dspf2d {
namespace op = ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d;
}
#endif

namespace dspf2d {

// Depth of the reader -> sender L1 ring, in tokens, and the half-ring batch slots move in. Both are
// packed as compile-time args, so this is their one definition. BATCH <= NUM_L1_SLOTS/2 is what proves
// the reader cannot claim every slot before publishing any.
//
// The ring is sized from BATCH rather than the other way round, and BATCH is the DRAM channel count:
// pages of an interleaved buffer land on consecutive banks, so a batch of reads only reaches every
// bank once it is at least as wide as there are banks. At half that the relay leaves half the DRAM
// idle no matter how deep the queue is.
constexpr uint32_t DRAM_CHANNELS = 8;  // blackhole_140_arch.yaml, `dram:`
constexpr uint32_t BATCH = DRAM_CHANNELS;
constexpr uint32_t NUM_L1_SLOTS = 2 * BATCH;

// Tokens whose routing metadata the reader prefetches in one batch, and the pad each record gets. 64 B
// because a DRAM read needs a 64-byte-aligned L1 destination on Blackhole, which a packed record of
// num_experts_per_tok uint16 would not keep -- every token after the first would land at a wrong
// address and the whole bucket index would be built from garbage.
constexpr uint32_t META_PREFETCH = 64;
constexpr uint32_t META_PAD_STRIDE = 64;

// Words per assignment in the reader's assignment block: [dst_chip_id, dst_row, split_idx, split_count].
constexpr uint32_t ASSIGNMENT_WORDS = 4;

// Ring slot = token + routing tail. 64 keeps the slot stride DRAM-aligned (14336 + 64 = 64 * 225), which
// lets a fabric write target a (token_size + 64)-byte forwarding page directly.
constexpr uint32_t FORWARDING_METADATA_SIZE = 64;

// The routing tail a ring slot carries after its token, at (slot_base + token_size_bytes). The reader
// fills it, the sender consumes it.
//
// Field order is wire format, not convenience: the first four members are what the next hop needs, so a
// forwarded packet is the token plus a contiguous FWD_EXTRA_BYTES prefix of this struct. `cmd` and
// `this_addr` are recomputed by whichever reader picks the page up and so are not sent.
//
// Dispatch writes each token to TWO tensors at one page index, so both destination addresses travel with
// it; combine needs only one. All uint64_t so a sender needs no sub-word loads.
struct FwdMetadata {
    // First, because the last hop sends these words straight out of the slot to the metadata page and a
    // fabric write's L1 source has to be aligned. The tail starts at slot_base + token_size, and a token
    // page is 64-byte aligned, so offset 0 of the tail is the only place in it that is.
    uint32_t meta[3];             // (src chip, token index, top-k slot), the metadata this token carries
    uint32_t pad;                 // makes the addresses below 8-byte aligned
    uint64_t final_payload_addr;  // token page address on the FINAL destination chip
    uint64_t final_meta_addr;     // metadata page address on that same chip
    uint64_t dst_chip;            // final destination chip id
    uint64_t cmd;
    uint64_t this_addr;  // the address THIS hop writes to
};

// Wire format between chips rather than a private convenience.
// The whole 64-byte tail goes on the wire, not just the prefix the next hop reads. A forwarded
// packet's length has to be a whole number of NoC beats or the trailing partial beat is dropped
// silently: at 40 the last field, dst_chip, never arrives and reads back as zero. Token pages are
// already 64-byte aligned, so sending the full tail keeps the transfer aligned as well.
constexpr uint32_t FWD_EXTRA_BYTES = FORWARDING_METADATA_SIZE;

// What the next hop actually consumes out of that tail.
constexpr uint32_t FWD_USED_BYTES = 4 * sizeof(uint32_t) + 3 * sizeof(uint64_t);

// One multicast destination, packed into a word: the page it lands on, how many hops away that chip
// is, and which top-k slot it came from. Under fan-out a token reaches several pages on several chips
// and each needs its own slot, so the single meta[2] of the unicast path cannot carry it.
//
// A destination on the origin's own chip never enters this list -- the local phase owns those -- so a
// live entry always has hop >= 1, and an all-zero word is an empty slot. That is what lets the tail
// drop a separate count and still fit the layout FwdMetadata pins (see FanoutMetadata).
constexpr uint32_t FO_PAGE_BITS = 20;
constexpr uint32_t FO_HOP_BITS = 5;
constexpr uint32_t FO_SLOT_BITS = 4;
constexpr uint32_t FO_HOP_SHIFT = FO_PAGE_BITS;
constexpr uint32_t FO_SLOT_SHIFT = FO_PAGE_BITS + FO_HOP_BITS;
constexpr uint32_t FO_PAGE_MASK = (1u << FO_PAGE_BITS) - 1u;
constexpr uint32_t FO_HOP_MASK = (1u << FO_HOP_BITS) - 1u;
static_assert(FO_PAGE_BITS + FO_HOP_BITS + FO_SLOT_BITS <= 32);

// Everything the routing pass needs to know about one global expert id, in ONE word, so the per-pick
// inner loop is a single indexed load rather than a dispatch-table lookup followed by a linear search
// over the destination chip's experts.
//
// The hop field is stored ALREADY SHIFTED to where a packed destination wants it, so building that
// word is a mask and two ors. Bits 18 and 19 are the gap that keeps the slot field clear of it.
constexpr uint32_t ES_SLOT_BITS = 16;                              // bucket index, extent * experts_per_chip of them
constexpr uint32_t ES_SLOT_MASK = (1u << ES_SLOT_BITS) - 1u;
constexpr uint32_t ES_LOCAL_BIT = 1u << 16;                        // the expert lives on this chip
constexpr uint32_t ES_DIR_SHIFT = 17;                              // 0 clockwise, 1 counter-clockwise
constexpr uint32_t ES_HOP_FIELD = FO_HOP_MASK << FO_HOP_SHIFT;     // hops away, in FanoutMetadata position
// An expert of another dispatch group. Distinguishable from any live word because the fields above
// leave the top bits clear.
constexpr uint32_t ES_NOT_HERE = 0xFFFFFFFFu;
static_assert(ES_SLOT_BITS <= 16 && (1u << ES_DIR_SHIFT) < (1u << FO_HOP_SHIFT));
static_assert((ES_SLOT_MASK & ES_HOP_FIELD) == 0 && (ES_LOCAL_BIT & ES_HOP_FIELD) == 0);

// One multicast entry: the token, how many destinations it carries, the farthest of them, then the
// packed destinations. A token reaches at most one page per top-k pick, so topk bounds the list.
//
// `far` is stored rather than re-derived by scanning the destinations, because the pass that emits
// the entry already knows it and the own phase would otherwise recompute it per entry.
constexpr uint32_t fo_entry_words(uint32_t topk) { return 3u + topk; }

// Destinations one multicast page can carry. A token reaches at most one page per top-k pick, so this
// is the top-k bound rather than anything about the ring.
constexpr uint32_t FO_MAX_DESTS = 8;

// The fan-out tail, occupying the same 64 bytes as FwdMetadata. Hops are measured from the ORIGIN and
// never rewritten, so a page is immutable in flight: a chip `j` hops from the origin consumes the
// destinations with hop == j out of its own forwarding region -- a local write, not a fabric one --
// and forwards the page untouched if any hop > j + 1 remains. If the farthest is exactly j + 1 the
// page ends next door and that chip writes those destinations itself, straight into the neighbour's
// output pages, so a page enters a region only while something beyond that region's chip is left.
//
// `cmd` and `this_addr` sit at the SAME offsets FwdMetadata puts them at, and the asserts below are
// what hold them there. The sender reads the command word out of a slot without knowing which mode
// built it, so a tail that moved `cmd` would have it dispatching on dests[6]. Pinning them costs the
// count field: with 8 destinations at top-k 8 there is no room for one, so a zero word is the empty
// slot instead -- a live destination is on another chip and so always has hop >= 1.
struct FanoutMetadata {
    uint32_t src_chip;             // linearized coord of the origin, metadata field 0
    uint32_t token;                // metadata field 1
    uint32_t dests[FO_MAX_DESTS];  // packed page | hop | top-k slot, zero where unused
    uint64_t cmd;
    uint64_t this_addr;
    // How many of this slot's staged deliveries the SENDER writes out before it acts on `cmd`. The
    // reader stages them because only it holds the output accessors; the sender issues them because
    // its port has the headroom -- multicast forwards far fewer pages than it delivers, and the
    // reader's port was the one saturated.
    //
    // The first `local_count` records land on THIS chip and go out as plain NoC writes; the
    // `remote_count` after them land on the NEXT one and go out as fabric packets. Position in the
    // list is the ONLY thing that distinguishes the two, so a stager that emitted them in the other
    // order would write the neighbour's pages here and fabric-send this chip's pages next door. A
    // destination is staged at most once either way, so both counts share one list of FO_MAX_DESTS.
    //
    // These two words sit past the end of FwdMetadata, which is 56 bytes used of the same 64. A tail
    // built by the unicast path therefore leaves them uninitialised, and they are loop bounds for
    // sends -- which is why the sender runs the delivery loops only for the commands fan-out emits.
    uint32_t local_count;
    uint32_t remote_count;
};
static_assert(sizeof(FanoutMetadata) <= FORWARDING_METADATA_SIZE);
// The claim above, held: the counts begin at or after everything FwdMetadata defines, so a unicast
// tail never happens to leave a plausible value in them.
static_assert(offsetof(FanoutMetadata, local_count) >= sizeof(FwdMetadata));

// Bytes the last hop writes to the metadata page: the three words rounded up to a NoC-friendly size.
constexpr uint32_t METADATA_WIRE_BYTES = 16;

// One delivery the sender issues out of a slot: where the token and its metadata go, and where the
// reader staged the metadata words. Every destination has its own record and its own metadata buffer
// because several are in flight from one slot at once. The addresses are page addresses in
// interleaved DRAM whose base is uniform across the mesh, so the same record serves a write on this
// chip and a fabric write to the next one.
struct FanoutDelivery {
    uint64_t payload_addr;
    uint64_t meta_addr;
    uint32_t meta_src;
    uint32_t pad;
};
static_assert(sizeof(FanoutDelivery) == 24);

// Prebuilt packet headers a ring slot needs. A header is read out of L1 asynchronously while the next
// send is being built, so every packet in flight from one slot needs its own or the one still going
// out is torn. Index 0 is the forward, or the payload of a unicast last hop; index 1 is the metadata
// beside it and goes unused under fan-out. From FO_FIRST_DELIVERY_HDR the headers come in pairs, one
// pair per destination a fan-out slot delivers into the next chip -- and that slot may still forward,
// so the pairs cannot reuse index 0.
//
// The host reserves the pool from this same expression. A pool shorter than the kernel's stride puts
// the last slots' headers on top of what follows, which under fan-out is the delivery records those
// very sends read their addresses from.
constexpr uint32_t FO_FIRST_DELIVERY_HDR = 2;
constexpr uint32_t headers_per_slot(bool fanout) { return fanout ? FO_FIRST_DELIVERY_HDR + 2u * FO_MAX_DESTS : 2u; }
// The highest index deliver_remotely can reach, against what the pool provides.
static_assert(FO_FIRST_DELIVERY_HDR + 2u * (FO_MAX_DESTS - 1u) + 1u < headers_per_slot(true));

// Asserted so that whoever changes this layout has to acknowledge they need some other means of ensuring
// every device runs kernels built from the same metadata format.
static_assert(sizeof(FwdMetadata) <= FORWARDING_METADATA_SIZE);
static_assert(offsetof(FwdMetadata, meta) == 0);
static_assert(offsetof(FwdMetadata, final_payload_addr) == 4 * sizeof(uint32_t));
static_assert(offsetof(FwdMetadata, cmd) == FWD_USED_BYTES);
static_assert(FWD_USED_BYTES == 40);
static_assert(FWD_EXTRA_BYTES % 64 == 0);
static_assert(METADATA_WIRE_BYTES <= FORWARDING_METADATA_SIZE);

// The sender reads one command word per slot and neither knows nor cares which mode filled the tail, so
// the two layouts have to agree on where it is. Move either field in either struct and the sender
// dispatches on whatever else happens to sit there -- under fan-out, a packed destination.
static_assert(offsetof(FanoutMetadata, cmd) == offsetof(FwdMetadata, cmd));
static_assert(offsetof(FanoutMetadata, this_addr) == offsetof(FwdMetadata, this_addr));

constexpr uint64_t CMD_END = 0;          // end of stream; the slot carries no token
constexpr uint64_t CMD_FINAL_WRITE = 1;  // this hop is the last: write payload and metadata to their pages
constexpr uint64_t CMD_FORWARD = 2;      // push one page further along the stream
constexpr uint64_t CMD_FORWARD_END = 3;  // as CMD_FORWARD, and the last page of its chunk
// Fan-out only: nothing is left past the chip across the cable, so no PAGE is forwarded. The slot's
// staged deliveries still go out, and up to 2 * FO_MAX_DESTS of them are fabric packets aimed at that
// chip's output pages -- so this is not a silent slot, and the counts in the tail rather than this
// command say what it sends. The slot still has to travel the ring in order, since handing it back
// would reorder the sender's view of it.
constexpr uint64_t CMD_NO_FORWARD = 4;

// One (origin chip, destination chip) term of a stream's forwarding region, narrowed to the share the two
// chips agreed on. Generated identically by the chip that writes the region and the chip that reads it,
// which is what lets the region be dense and carry no addresses.
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

// --- Multicast geometry -------------------------------------------------------------------------
//
// Under fan-out a chunk is (origin, hop) rather than (origin, destination, expert): one page per token
// per DIRECTION, travelling to the farthest destination that way while every chip en route keeps what
// it wants and passes the rest on. A stream therefore carries exactly extent/2 chunks -- its own, plus
// one per upstream origin -- instead of relay_chunks_per_stream(extent) * experts_per_chip.
constexpr uint32_t mc_chunks_per_stream(uint32_t ring_extent) { return ring_extent / 2u; }

// Hops the reach table is indexed by: 1..m are real distances, m + 1 is a terminating zero so that the
// tokens whose farthest destination is exactly m come out of reach[m] - reach[m + 1] like any other
// class. Index 0 is unused and kept only so `hop` indexes directly.
constexpr uint32_t mc_reach_hops(uint32_t ring_extent) { return ring_extent / 2u + 2u; }

// Bound on those hops, so the origin can keep one per-class counter on the stack. A hop is packed into
// FO_HOP_BITS, so nothing beyond this is expressible on the wire either.
constexpr uint32_t MC_MAX_HOPS = (1u << FO_HOP_BITS);

// One reach row per (origin, direction), padded to 64 bytes in L1: a DRAM read needs a 64-byte-aligned
// L1 destination on Blackhole, and a row is mc_reach_hops * 4 bytes, which is not a multiple of 64 at
// any extent this op runs on.
constexpr uint32_t mc_reach_row_bytes(uint32_t ring_extent) { return (mc_reach_hops(ring_extent) * 4u + 63u) & ~63u; }

// --- The reader's L1 control region -------------------------------------------------------------
//
// One ordered list of blocks, sized here and nowhere else. The host reserves the sum and the kernel
// carves the offsets, and a mismatch between those two overruns into the global semaphores with no
// guard but an ASSERT that is compiled out on this hardware -- which has already happened twice, both
// times with a green build. Adding a block here is the only way to add one to either side.
enum ControlBlock : uint32_t {
    kCbIndices,
    kCbOffsets,
    kCbCounts,
    kCbRegion,
    kCbTable,
    kCbExpertSlot,
    kCbAlloc,
    kCbChipExperts,
    kCbBucketFill,
    kCbBucketStart,
    kCbEntries,
    kCbMcEntries,
    kCbMcCount,
    kCbReach,
    kCbInStart,
    kCbOutStart,
    kCbCount
};

struct ControlGeometry {
    uint32_t seq_len = 0;
    uint32_t indices_pad_stride = 0;
    uint32_t extent = 0;
    uint32_t num_routed_experts = 0;
    uint32_t experts_per_chip = 0;
    uint32_t topk = 0;
    uint32_t num_relay = 0;  // relay_chunks_per_stream(extent), the same for every stream
    // The two modes build different routing indexes and only one of them ever runs, so each mode
    // reserves only what it uses. Part of the geometry rather than a switch at the carve, because the
    // host reserves and the kernel carves from this one struct.
    uint32_t fanout = 0;
};

// Chunk-start slots. The two modes carve one region: unicast needs one per (relay chunk, expert),
// multicast one per hop, and which is larger flips with experts_per_chip -- so both sides take the max
// rather than assuming either.
constexpr uint32_t control_chunk_start_slots(const ControlGeometry& g) {
    const uint32_t uni = g.num_relay * g.experts_per_chip;
    const uint32_t mc = mc_chunks_per_stream(g.extent);
    return uni > mc ? uni : mc;
}

// One destination's metadata words, padded so the next one starts aligned as well. A NoC write needs
// its L1 source to agree with its destination modulo the transfer's alignment, and the metadata page
// it lands on is 16-byte aligned: at four bare words the source sits wherever the blocks above happen
// to leave it, and a write from an odd offset arrives rotated -- (src, token, slot) reads back as
// (junk, src, token).
constexpr uint32_t MC_META_SLOT_BYTES = 64;

constexpr uint32_t control_block_raw_bytes(const ControlGeometry& g, uint32_t block) {
    switch (block) {
        case kCbIndices: return g.seq_len * g.indices_pad_stride;
        case kCbOffsets: return 4u * g.extent * g.num_routed_experts;
        case kCbCounts: return 4u * g.num_routed_experts;
        case kCbRegion: return 4u * g.num_routed_experts;
        // The dispatch table carries a trailing sentinel column, so a padded token's unguarded lookup
        // lands on it and resolves to "not in this group".
        case kCbTable: return 4u * (g.num_routed_experts + 1u);
        // Indexed by the same expert id the table is, sentinel column included.
        case kCbExpertSlot: return 4u * (g.num_routed_experts + 1u);
        // Keyed by bucket slot, not by global expert id: only the experts of this dispatch group
        // have an allocator, and the slot is what the per-pick lookup already yields.
        case kCbAlloc: return 4u * g.extent * g.experts_per_chip;
        case kCbChipExperts: return 4u * g.extent * g.experts_per_chip;
        case kCbBucketFill: return 4u * g.extent * g.experts_per_chip;
        // Inclusive prefix sums: a bucket's end is the next bucket's start, which is what bounds the
        // fill, so there is one more of these than there are buckets.
        case kCbBucketStart: return 4u * (g.extent * g.experts_per_chip + 1u);
        // (token, page, top-k slot) per surviving pick. Under fan-out only this chip's OWN
        // destinations go through a bucket, but the worst case is the same: every pick local.
        case kCbEntries: return 4u * g.seq_len * 3u * g.topk;
        case kCbMcEntries: return g.fanout ? 4u * 2u * g.seq_len * fo_entry_words(g.topk) : 0u;
        case kCbMcCount: return 4u * 2u;
        case kCbReach: return g.extent * 2u * mc_reach_row_bytes(g.extent);
        case kCbInStart: return 4u * control_chunk_start_slots(g);
        case kCbOutStart: return 4u * control_chunk_start_slots(g);
        default: return 0u;
    }
}

// Every block starts 64-byte aligned. Two of them are read straight out of DRAM, which needs that on
// Blackhole, and one is the source of a NoC write, which needs to agree with its destination modulo
// the transfer size. Aligning all of them costs under a kilobyte and makes the property hold for
// whatever block is added next, rather than for the ones someone remembered to check.
constexpr uint32_t control_block_bytes(const ControlGeometry& g, uint32_t block) {
    return (control_block_raw_bytes(g, block) + 63u) & ~63u;
}

constexpr uint32_t control_region_bytes(const ControlGeometry& g) {
    uint32_t total = 0;
    for (uint32_t b = 0; b < kCbCount; b++) {
        total += control_block_bytes(g, b);
    }
    return total;
}

}  // namespace dspf2d
