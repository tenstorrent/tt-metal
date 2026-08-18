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

#include <cstdint>

#ifndef KERNEL_BUILD
#include <vector>

#include <tt-metalium/buffer.hpp>

#include "../../combine_fabric2d_assignments.hpp"
#include "../../combine_fabric2d_placement.hpp"
#include "../../combine_fabric2d_types.hpp"
#endif

#ifndef KERNEL_BUILD
namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// Derived geometry. One-liners off the two structs the framework passes everywhere, named because
// `aligned_page_size()` as "the token size" is not self-evident.
uint32_t ring_extent(const CombineFabric2dParams& args) {
    return args.device->shape()[static_cast<int32_t>(args.axis)];
}

uint32_t token_size_bytes(const CombineFabric2dInputs& tensor_args) {
    return static_cast<uint32_t>(tensor_args.dispatched_buffer.buffer()->aligned_page_size());
}

uint32_t num_routed_experts(const CombineFabric2dInputs& tensor_args) {
    return static_cast<uint32_t>(tensor_args.expert_token_counts.logical_shape()[-1]);
}

uint32_t num_dispatch_groups(const CombineFabric2dParams& args, const CombineFabric2dInputs& tensor_args) {
    return num_routed_experts(tensor_args) / (args.experts_per_chip * ring_extent(args));
}

uint32_t my_row(const CombineFabric2dParams& args, const ttnn::MeshCoordinate& coord) {
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
    uint32_t pages_per_chunk = 0;
    uint32_t ring_filled_addr = 0;
    uint32_t ring_freed_addr = 0;
    uint32_t fwd_arrived_addr = 0;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
#endif

namespace cmbf2d {

#ifndef KERNEL_BUILD
namespace op = ttnn::operations::experimental::deepseek_prefill::combine_fabric2d;
#endif

// Depth of the reader -> sender L1 ring, in tokens, and the half-ring batch slots move in. Both are
// packed as compile-time args, so this is their one definition.
constexpr uint32_t NUM_L1_SLOTS = 8;
constexpr uint32_t BATCH = NUM_L1_SLOTS / 2;

// Tokens whose routing metadata the reader prefetches in one batch, and the pad each record gets. 64 B
// because a DRAM read needs a 64-byte-aligned L1 destination on Blackhole, which a 12-byte record would not
// keep. The host sizes the control region with these; the reader indexes the pads with them.
constexpr uint32_t META_PREFETCH = 64;
constexpr uint32_t META_PAD_STRIDE = 64;

// Words per assignment in the reader's assignment block: [dst_chip_id, dst_row, split_idx, split_count].
constexpr uint32_t ASSIGNMENT_WORDS = 4;
// Marks a schedule entry as "relay forwarding chunk k" rather than "own assignment k".
constexpr uint32_t SCHED_FWD = 0x80000000u;

// Ring slot = token + forwarding metadata. 64 rather than the 32 in use keeps the slot stride DRAM-aligned
// (14336 + 64 = 64 * 225), which lets a fabric write target a (token_size + 64)-byte forwarding page
// directly.
constexpr uint32_t FORWARDING_METADATA_SIZE = 64;

// Forwarding-metadata words, as uint64_t indices from (slot_base + token_size_bytes). The reader fills them,
// the sender consumes them. All uint64_t so the sender needs no sub-word loads.
constexpr uint32_t FWD_META_FINAL_ADDR = 0;  // destination DRAM address on the FINAL destination chip
constexpr uint32_t FWD_META_DST_CHIP = 1;    // final destination chip id; SENTINEL_DST_CHIP marks a sentinel
constexpr uint32_t FWD_META_CMD = 2;
constexpr uint32_t FWD_META_THIS_ADDR = 3;  // the address THIS hop writes to

constexpr uint64_t CMD_END = 0;  // end of stream; the slot carries no token
constexpr uint64_t CMD_FINAL_WRITE = 1;
constexpr uint64_t CMD_FORWARD = 2;

// A sentinel carries no usable token; it marks the end of a forwarding chunk. UINT64_MAX can never collide
// with a real chip id.
constexpr uint64_t SENTINEL_DST_CHIP = UINT64_MAX;

}  // namespace cmbf2d
