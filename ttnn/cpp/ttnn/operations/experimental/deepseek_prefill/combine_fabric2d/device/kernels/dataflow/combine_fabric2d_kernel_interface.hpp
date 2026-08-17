// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The complete host <-> device interface of this op: the compile-time argument order, the wire format of a
// ring slot's metadata tail, the kernels' readers of those args, and the host's writers of them. Both sides
// address every argument by name from the enums below, so no index is ever written down twice.
//
// Everything the host needs sits behind KERNEL_BUILD, so a kernel translation unit never sees it — neither
// the code nor its includes.
//
// The host fills a fixed-size array by name (`pack()`) and the kernels read by name (`ProducerCtArgs` /
// `ReaderCtArgs`), which is why this is not a byte stream: kernel-side values must be `constexpr` to drive
// `constexpr` branches and template parameters, and C++17 offers no constexpr way to reinterpret bytes as
// a struct.
//
// The reader's variable-length blocks follow its scalars, in this order:
//   [ReaderCtArg::Count ..)                      schedule, `schedule_len` words
//   [schedule end ..)                            assignments, 4 words each
//   [assignments end ..)                         TensorAccessorArgs, chained
// The producer has scalars only: every address it writes arrives per token in the slot's metadata tail.

#include <cstdint>

#ifndef KERNEL_BUILD
#include <array>
#include <bitset>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt_stl/assert.hpp>

#include "../../combine_fabric2d_assignments.hpp"
#include "../../combine_fabric2d_placement.hpp"
#include "../../combine_fabric2d_types.hpp"
#endif

namespace cmbf2d {

enum class ProducerCtArg : uint32_t {
    NumL1Slots,
    TokenSizeBytes,
    ForwardingMetadataSize,
    PeerChipId,
    PeerMeshId,
    RingAddr,
    PktHdrRingAddr,
    PktHdrDrainAddr,
    DrainSinkAddr,
    Batch,
    FilledAddr,
    FreedAddr,
    FwdSemNocX,
    FwdSemNocY,
    FwdSemAddr,
    Count,
};

enum class ReaderCtArg : uint32_t {
    NumL1Slots,
    TokenSizeBytes,
    ForwardingMetadataSize,
    Batch,
    RingAddr,
    FilledAddr,
    FreedAddr,
    DramInBaseAddr,
    DramOutBaseAddr,
    DramFwdBaseAddr,
    FwdChunksPerQuarter,
    FwdPagesPerChunk,
    MyQuarter,
    NumIncomingChunks,
    FwdSemAddr,
    NbrChipId,
    NumAssignments,
    ScheduleLen,
    DramMetaBaseAddr,
    DramCountsBaseAddr,
    DramRegionBaseAddr,
    DramExpertOffsetsBaseAddr,
    NumRoutedExperts,
    ExpertsPerChip,
    MyExpertBase,
    NumExpertsPerTok,
    DispatchGroupSize,
    LocalSplitCount,
    MyRow,
    ControlAddr,
    MetaPrefetchCap,
    Count,
};

// Depth of the reader -> producer L1 ring, in tokens, and the half-ring batch slots move in. Both are
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

// Ring slot = token + metadata tail. 64 rather than the 32 in use keeps the slot stride DRAM-aligned
// (14336 + 64 = 64 * 225), which lets a fabric write target a (token_size + 64)-byte forwarding page
// directly.
constexpr uint32_t FORWARDING_METADATA_SIZE = 64;

// Tail words, as uint64_t indices from (slot_base + token_size_bytes). The reader fills them, the producer
// consumes them. All uint64_t so the producer needs no sub-word loads.
constexpr uint32_t TAIL_FINAL_ADDR = 0;  // destination DRAM address on the FINAL destination chip
constexpr uint32_t TAIL_DST_CHIP = 1;    // final destination chip id; SENTINEL_DST_CHIP marks a sentinel
constexpr uint32_t TAIL_CMD = 2;
constexpr uint32_t TAIL_THIS_ADDR = 3;  // the address THIS hop writes to

constexpr uint64_t CMD_END = 0;  // end of stream; the slot carries no token
constexpr uint64_t CMD_FINAL_WRITE = 1;
constexpr uint64_t CMD_FORWARD = 2;

// A sentinel carries no usable token; it marks the end of a forwarding chunk. UINT64_MAX can never collide
// with a real chip id.
constexpr uint64_t SENTINEL_DST_CHIP = UINT64_MAX;

#ifdef KERNEL_BUILD

#define CMBF2D_CT(kind, name) get_compile_time_arg_val(static_cast<uint32_t>(kind::name))

// A kernel is passed one role's args only, and a `static constexpr` initialiser is evaluated as soon as its
// enclosing class is complete, so reading the other role's indices would fail the CT-arg range assert. Each
// kernel therefore defines its role before including this header.
#ifdef CMBF2D_PRODUCER_KERNEL
struct ProducerCtArgs {
    static constexpr uint32_t num_l1_slots = CMBF2D_CT(ProducerCtArg, NumL1Slots);
    static constexpr uint32_t token_size_bytes = CMBF2D_CT(ProducerCtArg, TokenSizeBytes);
    static constexpr uint32_t forwarding_metadata_size = CMBF2D_CT(ProducerCtArg, ForwardingMetadataSize);
    static constexpr uint32_t peer_chip_id = CMBF2D_CT(ProducerCtArg, PeerChipId);
    static constexpr uint32_t peer_mesh_id = CMBF2D_CT(ProducerCtArg, PeerMeshId);
    static constexpr uint32_t ring_addr = CMBF2D_CT(ProducerCtArg, RingAddr);
    static constexpr uint32_t pkt_hdr_ring_addr = CMBF2D_CT(ProducerCtArg, PktHdrRingAddr);
    static constexpr uint32_t pkt_hdr_drain_addr = CMBF2D_CT(ProducerCtArg, PktHdrDrainAddr);
    static constexpr uint32_t drain_sink_addr = CMBF2D_CT(ProducerCtArg, DrainSinkAddr);
    static constexpr uint32_t batch = CMBF2D_CT(ProducerCtArg, Batch);
    static constexpr uint32_t filled_addr = CMBF2D_CT(ProducerCtArg, FilledAddr);
    static constexpr uint32_t freed_addr = CMBF2D_CT(ProducerCtArg, FreedAddr);
    static constexpr uint32_t fwd_sem_noc_x = CMBF2D_CT(ProducerCtArg, FwdSemNocX);
    static constexpr uint32_t fwd_sem_noc_y = CMBF2D_CT(ProducerCtArg, FwdSemNocY);
    static constexpr uint32_t fwd_sem_addr = CMBF2D_CT(ProducerCtArg, FwdSemAddr);

    static constexpr uint32_t slot_stride = token_size_bytes + forwarding_metadata_size;
};
#endif

#ifdef CMBF2D_READER_KERNEL
struct ReaderCtArgs {
    static constexpr uint32_t num_l1_slots = CMBF2D_CT(ReaderCtArg, NumL1Slots);
    static constexpr uint32_t token_size_bytes = CMBF2D_CT(ReaderCtArg, TokenSizeBytes);
    static constexpr uint32_t forwarding_metadata_size = CMBF2D_CT(ReaderCtArg, ForwardingMetadataSize);
    static constexpr uint32_t batch = CMBF2D_CT(ReaderCtArg, Batch);
    static constexpr uint32_t ring_addr = CMBF2D_CT(ReaderCtArg, RingAddr);
    static constexpr uint32_t filled_addr = CMBF2D_CT(ReaderCtArg, FilledAddr);
    static constexpr uint32_t freed_addr = CMBF2D_CT(ReaderCtArg, FreedAddr);
    static constexpr uint32_t dram_in_base_addr = CMBF2D_CT(ReaderCtArg, DramInBaseAddr);
    static constexpr uint32_t dram_out_base_addr = CMBF2D_CT(ReaderCtArg, DramOutBaseAddr);
    static constexpr uint32_t dram_fwd_base_addr = CMBF2D_CT(ReaderCtArg, DramFwdBaseAddr);
    static constexpr uint32_t fwd_chunks_per_quarter = CMBF2D_CT(ReaderCtArg, FwdChunksPerQuarter);
    static constexpr uint32_t fwd_pages_per_chunk = CMBF2D_CT(ReaderCtArg, FwdPagesPerChunk);
    static constexpr uint32_t my_quarter = CMBF2D_CT(ReaderCtArg, MyQuarter);
    static constexpr uint32_t num_incoming_chunks = CMBF2D_CT(ReaderCtArg, NumIncomingChunks);
    static constexpr uint32_t fwd_sem_addr = CMBF2D_CT(ReaderCtArg, FwdSemAddr);
    static constexpr uint32_t nbr_chip_id = CMBF2D_CT(ReaderCtArg, NbrChipId);
    static constexpr uint32_t num_assignments = CMBF2D_CT(ReaderCtArg, NumAssignments);
    static constexpr uint32_t schedule_len = CMBF2D_CT(ReaderCtArg, ScheduleLen);
    static constexpr uint32_t dram_meta_base_addr = CMBF2D_CT(ReaderCtArg, DramMetaBaseAddr);
    static constexpr uint32_t dram_counts_base_addr = CMBF2D_CT(ReaderCtArg, DramCountsBaseAddr);
    static constexpr uint32_t dram_region_base_addr = CMBF2D_CT(ReaderCtArg, DramRegionBaseAddr);
    static constexpr uint32_t dram_expert_offsets_base_addr = CMBF2D_CT(ReaderCtArg, DramExpertOffsetsBaseAddr);
    static constexpr uint32_t num_routed_experts = CMBF2D_CT(ReaderCtArg, NumRoutedExperts);
    static constexpr uint32_t experts_per_chip = CMBF2D_CT(ReaderCtArg, ExpertsPerChip);
    static constexpr uint32_t my_expert_base = CMBF2D_CT(ReaderCtArg, MyExpertBase);
    static constexpr uint32_t num_experts_per_tok = CMBF2D_CT(ReaderCtArg, NumExpertsPerTok);
    static constexpr uint32_t dispatch_group_size = CMBF2D_CT(ReaderCtArg, DispatchGroupSize);
    static constexpr uint32_t local_split_count = CMBF2D_CT(ReaderCtArg, LocalSplitCount);
    static constexpr uint32_t my_row = CMBF2D_CT(ReaderCtArg, MyRow);
    static constexpr uint32_t control_addr = CMBF2D_CT(ReaderCtArg, ControlAddr);
    static constexpr uint32_t meta_prefetch_cap = CMBF2D_CT(ReaderCtArg, MetaPrefetchCap);

    static constexpr uint32_t schedule_base = static_cast<uint32_t>(ReaderCtArg::Count);
    static constexpr uint32_t assignment_base = schedule_base + schedule_len;
    static constexpr uint32_t accessor_base = assignment_base + ASSIGNMENT_WORDS * num_assignments;
    static constexpr uint32_t slot_stride = token_size_bytes + forwarding_metadata_size;
};
#endif

#undef CMBF2D_CT

#else  // host

// Fill by name, then append to the kernel's compile-time arg vector. Fields are deliberately unordered
// here — position comes from the enum, not from declaration order. `append_to` refuses to emit a partially
// filled record: the array is zero-initialised, so a slot nobody assigned would otherwise pack a silent 0,
// which for an address means the kernel writes to offset 0.
template <typename Arg, Arg CountValue>
class CtArgPacker {
public:
    static constexpr uint32_t count = static_cast<uint32_t>(CountValue);

    uint32_t& operator[](Arg a) {
        written_.set(static_cast<uint32_t>(a));
        return words_[static_cast<uint32_t>(a)];
    }

    void append_to(std::vector<uint32_t>& ct_args) const {
        for (uint32_t i = 0; i < count; i++) {
            TT_FATAL(written_.test(i), "combine_fabric2d: compile-time arg slot {} of {} was never set", i, count);
        }
        ct_args.insert(ct_args.end(), words_.begin(), words_.end());
    }

private:
    std::array<uint32_t, count> words_{};
    std::bitset<count> written_;
};

using ProducerCtArgPacker = CtArgPacker<ProducerCtArg, ProducerCtArg::Count>;
using ReaderCtArgPacker = CtArgPacker<ReaderCtArg, ReaderCtArg::Count>;

#endif  // KERNEL_BUILD

}  // namespace cmbf2d

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
    uint32_t ring;          // num_l1_slots tokens, filled by the reader and drained by the producer
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

std::vector<uint32_t> pack_producer_args(
    const CombineFabric2dInputs& tensor_args,
    const StreamPlacement& self,
    const StreamPlacement& downstream,
    const L1Layout& l1,
    const KernelPlan& plan) {
    cmbf2d::ProducerCtArgPacker a;
    a[cmbf2d::ProducerCtArg::NumL1Slots] = cmbf2d::NUM_L1_SLOTS;
    a[cmbf2d::ProducerCtArg::TokenSizeBytes] = token_size_bytes(tensor_args);
    a[cmbf2d::ProducerCtArg::ForwardingMetadataSize] = cmbf2d::FORWARDING_METADATA_SIZE;
    a[cmbf2d::ProducerCtArg::PeerChipId] = static_cast<uint32_t>(self.downstream_node.chip_id);
    a[cmbf2d::ProducerCtArg::PeerMeshId] = *self.downstream_node.mesh_id;
    a[cmbf2d::ProducerCtArg::RingAddr] = l1.ring;
    a[cmbf2d::ProducerCtArg::PktHdrRingAddr] = l1.pkt_hdr_ring;
    a[cmbf2d::ProducerCtArg::PktHdrDrainAddr] = l1.pkt_hdr_drain;
    a[cmbf2d::ProducerCtArg::DrainSinkAddr] = l1.drain_sink;
    a[cmbf2d::ProducerCtArg::Batch] = cmbf2d::BATCH;
    a[cmbf2d::ProducerCtArg::FilledAddr] = plan.ring_filled_addr;
    a[cmbf2d::ProducerCtArg::FreedAddr] = plan.ring_freed_addr;
    // The kernel needs the placement of its peer downstream worker because it sends the
    // forwarded-token-count semaphore bumps to it, addressed in the fabric packet header. This is also why
    // every worker placement on the mesh is decided before any kernel is built.
    a[cmbf2d::ProducerCtArg::FwdSemNocX] = static_cast<uint32_t>(downstream.worker_virtual.x);
    a[cmbf2d::ProducerCtArg::FwdSemNocY] = static_cast<uint32_t>(downstream.worker_virtual.y);
    a[cmbf2d::ProducerCtArg::FwdSemAddr] = plan.fwd_arrived_addr;
    std::vector<uint32_t> ct;
    a.append_to(ct);
    return ct;
}

std::vector<uint32_t> pack_reader_args(
    const CombineFabric2dParams& args,
    const CombineFabric2dInputs& tensor_args,
    const ttnn::MeshCoordinate& coord,
    const StreamPlacement& self,
    const std::vector<Assignment>& work,
    const L1Layout& l1,
    const KernelPlan& plan,
    const DramBuffers& dram) {
    cmbf2d::ReaderCtArgPacker a;
    a[cmbf2d::ReaderCtArg::NumL1Slots] = cmbf2d::NUM_L1_SLOTS;
    a[cmbf2d::ReaderCtArg::TokenSizeBytes] = token_size_bytes(tensor_args);
    a[cmbf2d::ReaderCtArg::ForwardingMetadataSize] = cmbf2d::FORWARDING_METADATA_SIZE;
    a[cmbf2d::ReaderCtArg::Batch] = cmbf2d::BATCH;
    a[cmbf2d::ReaderCtArg::RingAddr] = l1.ring;
    a[cmbf2d::ReaderCtArg::FilledAddr] = plan.ring_filled_addr;
    a[cmbf2d::ReaderCtArg::FreedAddr] = plan.ring_freed_addr;
    a[cmbf2d::ReaderCtArg::DramInBaseAddr] = static_cast<uint32_t>(dram.in->address());
    a[cmbf2d::ReaderCtArg::DramOutBaseAddr] = static_cast<uint32_t>(dram.out->address());
    a[cmbf2d::ReaderCtArg::DramFwdBaseAddr] = static_cast<uint32_t>(dram.fwd->address());
    a[cmbf2d::ReaderCtArg::FwdChunksPerQuarter] = relay_chunks_per_stream(ring_extent(args));
    a[cmbf2d::ReaderCtArg::FwdPagesPerChunk] = plan.pages_per_chunk;
    // Doubles as this stream's share of the same-chip run, which it copies after the fabric work.
    a[cmbf2d::ReaderCtArg::MyQuarter] = plan.stream;
    a[cmbf2d::ReaderCtArg::NumIncomingChunks] = relay_chunks_per_stream(ring_extent(args));
    a[cmbf2d::ReaderCtArg::FwdSemAddr] = plan.fwd_arrived_addr;
    a[cmbf2d::ReaderCtArg::NbrChipId] = static_cast<uint32_t>(self.downstream_node.chip_id);
    a[cmbf2d::ReaderCtArg::ScheduleLen] = static_cast<uint32_t>(work.size());
    a[cmbf2d::ReaderCtArg::DramMetaBaseAddr] = static_cast<uint32_t>(dram.meta->address());
    a[cmbf2d::ReaderCtArg::DramCountsBaseAddr] = static_cast<uint32_t>(dram.counts->address());
    a[cmbf2d::ReaderCtArg::DramRegionBaseAddr] = static_cast<uint32_t>(dram.region->address());
    a[cmbf2d::ReaderCtArg::DramExpertOffsetsBaseAddr] = static_cast<uint32_t>(dram.expert_offsets->address());
    a[cmbf2d::ReaderCtArg::NumRoutedExperts] = num_routed_experts(tensor_args);
    a[cmbf2d::ReaderCtArg::ExpertsPerChip] = args.experts_per_chip;
    a[cmbf2d::ReaderCtArg::MyExpertBase] = plan.my_expert_base;
    a[cmbf2d::ReaderCtArg::NumExpertsPerTok] = args.num_experts_per_tok;
    a[cmbf2d::ReaderCtArg::DispatchGroupSize] = ring_extent(args);
    a[cmbf2d::ReaderCtArg::LocalSplitCount] = stream_count(args.num_links);
    a[cmbf2d::ReaderCtArg::MyRow] = my_row(args, coord);
    a[cmbf2d::ReaderCtArg::ControlAddr] = l1.control;
    a[cmbf2d::ReaderCtArg::MetaPrefetchCap] = cmbf2d::META_PREFETCH;

    uint32_t num_own = 0;
    for (const auto& w : work) {
        num_own += w.is_relay ? 0u : 1u;
    }
    a[cmbf2d::ReaderCtArg::NumAssignments] = num_own;

    std::vector<uint32_t> ct;
    a.append_to(ct);
    // Schedule: the work order, relays tagged. An own entry carries its index into the table that follows.
    uint32_t own_idx = 0;
    for (const auto& w : work) {
        ct.push_back(w.is_relay ? (cmbf2d::SCHED_FWD | w.relay_chunk) : own_idx++);
    }
    for (const auto& w : work) {
        if (w.is_relay) {
            continue;
        }
        ct.push_back(w.dst_chip_id);
        ct.push_back(w.dst_row);
        ct.push_back(w.split_idx);
        ct.push_back(w.split_count);
    }
    return ct;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
#endif  // KERNEL_BUILD
