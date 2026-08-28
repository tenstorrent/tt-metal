// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// What both kernel roles and the host agree on: the wire format of a ring slot's forwarding metadata, the
// sizes the compile-time arguments are built from, and the host-side geometry the two argument structs are
// derived from. Each role's arguments live beside its kernel, in combine_fabric2d_sender_ct_args.hpp and
// combine_fabric2d_reader_ct_args.hpp.
//
// Everything the host needs sits behind KERNEL_BUILD, so a kernel translation unit never sees it — neither
// the code nor its includes.

#include <cstddef>
#include <cstdint>

#ifndef KERNEL_BUILD
#include <vector>

#include <tt-metalium/buffer.hpp>

#include "../../combine_fabric2d_placement.hpp"
#include "../../combine_fabric2d_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// Derived geometry. One-liners off the two structs the framework passes everywhere, named because
// `aligned_page_size()` as "the token size" is not self-evident.
uint32_t ring_extent(const CombineFabric2dParams& args) {
    return args.device->shape()[static_cast<int32_t>(args.axis)];
}

// One token's row of the embedding. Read off the tensor rather than taken as a parameter — and off its
// SHAPE, not its page: a ROW_MAJOR dispatched buffer pages by exactly one token, a TILE one does not.
uint32_t token_size_bytes(const CombineFabric2dInputs& tensor_args) {
    return static_cast<uint32_t>(tensor_args.dispatched_buffer.logical_shape()[-1]) *
           tensor_args.dispatched_buffer.element_size();
}

bool dispatched_is_tiled(const CombineFabric2dInputs& tensor_args) {
    return tensor_args.dispatched_buffer.layout() == tt::tt_metal::Layout::TILE;
}

// Tiles across one token, which is also the tiles in the tile-row a batch untilizes.
uint32_t tiles_per_token_row(const CombineFabric2dInputs& tensor_args) {
    return static_cast<uint32_t>(tensor_args.dispatched_buffer.logical_shape()[-1]) /
           tensor_args.dispatched_buffer.tensor_spec().tile().get_width();
}

uint32_t tile_size_bytes(const CombineFabric2dInputs& tensor_args) {
    return static_cast<uint32_t>(tensor_args.dispatched_buffer.tensor_spec().tile().get_tile_hw()) *
           tensor_args.dispatched_buffer.element_size();
}

// Tiles the untilize takes per pack call, and so the width of the input window: as wide as it can be, and a
// divisor of the row so the blocks tile it exactly. Eight is what llk_pack_untilize asserts as its ceiling
// off the dense path; a whole tile-row would not fit L1 on top of the output ring anyway, at 458 kB.
constexpr uint32_t UNTILIZE_MAX_BLOCK_TILES = 8;

uint32_t untilize_block_tiles(const CombineFabric2dInputs& tensor_args) {
    for (uint32_t block = UNTILIZE_MAX_BLOCK_TILES; block > 1; block--) {
        if (tiles_per_token_row(tensor_args) % block == 0) {
            return block;
        }
    }
    return 1;
}

uint32_t num_routed_experts(const CombineFabric2dInputs& tensor_args) {
    return static_cast<uint32_t>(tensor_args.expert_token_counts.logical_shape()[-1]);
}

uint32_t num_dispatch_groups(const CombineFabric2dParams& args, const CombineFabric2dInputs& tensor_args) {
    return num_routed_experts(tensor_args) / (args.experts_per_chip * ring_extent(args));
}

uint32_t my_dg_index(const CombineFabric2dParams& args, const ttnn::MeshCoordinate& coord) {
    return coord[static_cast<int32_t>(args.axis)];
}

struct L1Layout {
    uint32_t pkt_hdr_drain;
    uint32_t drain_sink;
    uint32_t ring;          // num_l1_slots tokens, filled by the reader and drained by the sender
    uint32_t pkt_hdr_ring;  // one prebuilt payload header per ring slot
    // The reader's copy of the control tensors, read once at startup and then indexed from L1.
    // Laid out as [expert_offsets: dispatch_group_size x num_routed_experts][counts: num_routed_experts]
    // [region_offsets: num_routed_experts], all uint32. Unlike everything above it, nothing on another chip
    // addresses this, so it can sit at the end — but it is still computed identically everywhere.
    uint32_t control;
    // An untilizer core's layout, which starts over at the allocator base: it shares no memory with the
    // cores above, and the base is where the framework puts its first circular buffer.
    // Zero where there are no untilizer cores, which is where there is nothing to untilize.
    uint32_t unt_ring = 0;     // the batch ring, which IS cb_out
    uint32_t unt_control = 0;  // its own copy of the control tables, past every circular buffer
};

struct DramBuffers {
    tt::tt_metal::Buffer* in = nullptr;
    tt::tt_metal::Buffer* out = nullptr;
    tt::tt_metal::Buffer* fwd = nullptr;
    tt::tt_metal::Buffer* meta = nullptr;
    tt::tt_metal::Buffer* counts = nullptr;
    tt::tt_metal::Buffer* region = nullptr;
    tt::tt_metal::Buffer* expert_offsets = nullptr;
};

// The per-chip values that had to be worked out rather than read off the arguments, plus the stream.
struct KernelPlan {
    StreamId stream = 0;
    uint32_t my_expert_base = 0;
    uint32_t pages_per_stream = 0;
    uint32_t ring_filled_addr = 0;
    uint32_t ring_freed_addr = 0;
    uint32_t fwd_arrived_addr = 0;
};

// The other end of one untilizer handshake: the core to address, and the counter that core's peer owns there.
struct HandshakePeer {
    CoreCoord noc;
    uint32_t counter_addr = 0;
};

// The untilizer group serving one reader, from that reader's side.
struct ReaderUntilizers {
    uint32_t ring_addr = 0;
    uint32_t my_freed_addr = 0;  // the counter this reader owns on each untilizer core
    std::vector<HandshakePeer> peers;
};

// The per-core values an untilizer needs that are not read off the arguments.
struct UntilizerPlan {
    uint32_t my_expert_base = 0;
    uint32_t my_index = 0;   // position in the group
    uint32_t num_peers = 0;  // cores in the group
    uint32_t walks_down = 0;
    uint32_t control_addr = 0;
    uint32_t produced_addr = 0;  // the counter this core owns on each of its consumers
    std::vector<HandshakePeer> consumers;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d

namespace cmbf2d {
namespace op = ttnn::operations::experimental::deepseek_prefill::combine_fabric2d;
}
#endif

namespace cmbf2d {

// Depth of the reader -> sender L1 ring, in tokens, and the half-ring batch slots move in. Both are
// packed as compile-time args, so this is their one definition.
constexpr uint32_t NUM_L1_SLOTS = 8;
constexpr uint32_t BATCH = NUM_L1_SLOTS / 2;

// Tokens whose routing metadata the reader prefetches in one batch, and the pad each record gets. 64 B
// because a DRAM read needs a 64-byte-aligned L1 destination on Blackhole, which a 12-byte record would not
// keep. The host sizes the control region with these; the reader indexes the pads with them.
constexpr uint32_t META_PREFETCH = 64;
constexpr uint32_t META_PAD_STRIDE = 64;

// Rows one untilize produces, which is one tile-row of the dispatched buffer. The least it can produce,
// whatever the tokens wanted, so it is the unit a group of untilizers hands over.
constexpr uint32_t UNT_BATCH_ROWS = 32;
// Batches an untilizer keeps in flight, so the next one can be built while its consumers work through the
// one before it.
constexpr uint32_t UNT_RING_BATCHES = 2;

// Circular buffer indices on an untilizer core. cb_out is declared FIRST, because the framework lays a
// program's circular buffers out from the L1 allocator base in declaration order -- which is what makes
// cb_out and L1Layout::unt_ring the same memory, and so what lets a reader name a row by core and offset.
constexpr uint32_t UNT_CB_OUT = 0;
constexpr uint32_t UNT_CB_IN = 1;
constexpr uint32_t UNT_CB_BATCHES = 2;
// Words per entry in a handshake block: [noc_x, noc_y, counter_addr].
constexpr uint32_t UNT_PEER_WORDS = 3;

// Words per assignment in the reader's assignment block: [dst_chip_id, dst_dg_index, split_idx, split_count].
constexpr uint32_t ASSIGNMENT_WORDS = 4;
// Words per chunk descriptor.
constexpr uint32_t CHUNK_WORDS = 4;

// One chunk of a stream's forwarding region: whose tokens it carries, for which chip, and the share of each
// run those two chips agreed on. Enough to compute the chunk's token count, and so its page range once every
// chunk before it in the region has been counted too.
//
// Packed by position into the reader's compile-time args. to_words below is the only place that order is
// written down, and from_words mirrors it, so the host that emits a chunk and the kernel that reads it back
// cannot drift apart.
struct ChunkDescriptor {
    uint32_t origin_dg_index = 0;
    uint32_t dst_dg_index = 0;
    uint32_t split_idx = 0;
    uint32_t split_count = 1;

    void to_words(uint32_t* words) const {
        words[0] = origin_dg_index;
        words[1] = dst_dg_index;
        words[2] = split_idx;
        words[3] = split_count;
    }

    static ChunkDescriptor from_words(const uint32_t* words) {
        return ChunkDescriptor{words[0], words[1], words[2], words[3]};
    }

#ifndef KERNEL_BUILD
    void append_to(std::vector<uint32_t>& out) const {
        uint32_t words[CHUNK_WORDS];
        to_words(words);
        out.insert(out.end(), words, words + CHUNK_WORDS);
    }
#endif
};
static_assert(sizeof(ChunkDescriptor) == CHUNK_WORDS * sizeof(uint32_t));
// Marks a schedule entry as "relay forwarding chunk k" rather than "own assignment k".
constexpr uint32_t SCHED_FWD = 0x80000000u;

// Ring slot = token + forwarding metadata. 64 rather than the 32 in use keeps the slot stride DRAM-aligned
// (14336 + 64 = 64 * 225), which lets a fabric write target a (token_size + 64)-byte forwarding page
// directly.
constexpr uint32_t FORWARDING_METADATA_SIZE = 64;

// The forwarding metadata a ring slot carries after its token, at (slot_base + token_size_bytes). The reader
// fills it, the sender consumes it. All uint64_t so the sender needs no sub-word loads.
struct FwdMetadata {
    uint64_t final_addr;  // destination DRAM address on the FINAL destination chip
    uint64_t dst_chip;    // final destination chip id
    uint64_t cmd;
    uint64_t this_addr;  // the address THIS hop writes to
};

// A forwarded packet is the token plus the first two words, sent as one contiguous run, so the layout below
// is wire format between chips rather than a private convenience.
constexpr uint32_t FWD_EXTRA_BYTES = 2 * sizeof(uint64_t);

// Asserted so that whoever changes this layout has to acknowledge they need some other means of ensuring
// every device runs kernels built from the same metadata format.
static_assert(sizeof(FwdMetadata) <= FORWARDING_METADATA_SIZE);
static_assert(offsetof(FwdMetadata, final_addr) == 0);
static_assert(offsetof(FwdMetadata, dst_chip) == sizeof(uint64_t));
static_assert(offsetof(FwdMetadata, cmd) == 2 * sizeof(uint64_t));
static_assert(offsetof(FwdMetadata, this_addr) == 3 * sizeof(uint64_t));

constexpr uint64_t CMD_END = 0;  // end of stream; the slot carries no token
constexpr uint64_t CMD_FINAL_WRITE = 1;
constexpr uint64_t CMD_FORWARD = 2;
// A forward that also ends a chunk. The sender bumps the downstream reader's arrival counter right after
// sending it, because that reader is waiting on exactly this chunk's pages and the residual of a partial
// bump batch would otherwise sit uncounted until end of stream — which, on a ring, is a deadlock.
constexpr uint64_t CMD_FORWARD_END = 3;

}  // namespace cmbf2d
