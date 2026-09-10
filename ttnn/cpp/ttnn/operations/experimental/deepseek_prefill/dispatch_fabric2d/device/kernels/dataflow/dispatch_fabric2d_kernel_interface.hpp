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
    uint32_t pkt_hdr_ring;  // TWO prebuilt headers per slot: the last hop issues two writes from one slot
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
constexpr uint32_t NUM_L1_SLOTS = 8;
constexpr uint32_t BATCH = NUM_L1_SLOTS / 2;

// Tokens whose routing metadata the reader prefetches in one batch, and the pad each record gets. 64 B
// because a DRAM read needs a 64-byte-aligned L1 destination on Blackhole, which a packed record of
// num_experts_per_tok uint16 would not keep -- every token after the first would land at a wrong
// address and the whole bucket index would be built from garbage.
constexpr uint32_t META_PREFETCH = 64;
constexpr uint32_t META_PAD_STRIDE = 64;

// Words per assignment in the reader's assignment block: [dst_chip_id, dst_row, split_idx, split_count].
constexpr uint32_t ASSIGNMENT_WORDS = 4;
// Marks a schedule entry as "relay chunk k" rather than "own assignment k".
constexpr uint32_t SCHED_FWD = 0x80000000u;

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

// Bytes the last hop writes to the metadata page: the three words rounded up to a NoC-friendly size.
constexpr uint32_t METADATA_WIRE_BYTES = 16;

// Asserted so that whoever changes this layout has to acknowledge they need some other means of ensuring
// every device runs kernels built from the same metadata format.
static_assert(sizeof(FwdMetadata) <= FORWARDING_METADATA_SIZE);
static_assert(offsetof(FwdMetadata, meta) == 0);
static_assert(offsetof(FwdMetadata, final_payload_addr) == 4 * sizeof(uint32_t));
static_assert(offsetof(FwdMetadata, cmd) == FWD_USED_BYTES);
static_assert(FWD_USED_BYTES == 40);
static_assert(FWD_EXTRA_BYTES % 64 == 0);
static_assert(METADATA_WIRE_BYTES <= FORWARDING_METADATA_SIZE);

constexpr uint64_t CMD_END = 0;          // end of stream; the slot carries no token
constexpr uint64_t CMD_FINAL_WRITE = 1;  // this hop is the last: write payload and metadata to their pages
constexpr uint64_t CMD_FORWARD = 2;      // push one page further along the stream
constexpr uint64_t CMD_FORWARD_END = 3;  // as CMD_FORWARD, and the last page of its chunk

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

}  // namespace dspf2d
