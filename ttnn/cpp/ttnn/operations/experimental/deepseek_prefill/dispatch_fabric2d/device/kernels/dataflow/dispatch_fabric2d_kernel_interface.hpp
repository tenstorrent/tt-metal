// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

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
    uint64_t final_payload_addr;  // token page address on the FINAL destination chip
    uint64_t final_meta_addr;     // metadata page address on that same chip
    uint64_t dst_chip;            // final destination chip id
    uint32_t meta[3];             // (src chip, token index, top-k slot), the metadata this token carries
    uint32_t pad;                 // keeps the forwarded prefix 8-byte aligned
    uint64_t cmd;
    uint64_t this_addr;  // the address THIS hop writes to
};

// A forwarded packet is the token plus everything up to and including `pad`, sent as one contiguous run,
// so this bound is wire format between chips rather than a private convenience.
constexpr uint32_t FWD_EXTRA_BYTES = 3 * sizeof(uint64_t) + 4 * sizeof(uint32_t);

// Asserted so that whoever changes this layout has to acknowledge they need some other means of ensuring
// every device runs kernels built from the same metadata format.
static_assert(sizeof(FwdMetadata) <= FORWARDING_METADATA_SIZE);
static_assert(offsetof(FwdMetadata, final_payload_addr) == 0);
static_assert(offsetof(FwdMetadata, final_meta_addr) == sizeof(uint64_t));
static_assert(offsetof(FwdMetadata, dst_chip) == 2 * sizeof(uint64_t));
static_assert(offsetof(FwdMetadata, meta) == 3 * sizeof(uint64_t));
static_assert(offsetof(FwdMetadata, cmd) == FWD_EXTRA_BYTES);
static_assert(FWD_EXTRA_BYTES == 40);

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
