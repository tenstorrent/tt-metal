// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The complete host <-> device interface of this op: the compile-time argument order, the wire format of a
// ring slot's forwarding metadata, and one argument struct per kernel role. Each role has a single field
// list; the host constructor takes what the program factory already has and derives every field from it,
// the kernel constructor reads the same fields back out of the compile-time args. Indices are named once,
// in the enums below.
//
// Everything the host needs sits behind KERNEL_BUILD, so a kernel translation unit never sees it — neither
// the code nor its includes. Kernel-side values must be `constexpr` to drive `constexpr` branches and
// template parameters, which is why this is a field list and not a byte stream.
//
// The reader's variable-length blocks follow its scalars, in this order:
//   [ReaderCtArg::Count ..)                      schedule, `schedule_len` words
//   [schedule end ..)                            assignments, 4 words each
//   [assignments end ..)                         TensorAccessorArgs, chained
// The sender has scalars only: every address it writes arrives per token in the slot's forwarding metadata.

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

enum class SenderCtArg : uint32_t {
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

#ifdef KERNEL_BUILD
#define CMBF2D_CT(kind, name) get_compile_time_arg_val(static_cast<uint32_t>(kind::name))
#endif

// Field order is the compile-time argument order, so it must match SenderCtArg.
struct SenderCtArgs {
    uint32_t num_l1_slots;
    uint32_t token_size_bytes;
    uint32_t forwarding_metadata_size;
    uint32_t peer_chip_id;
    uint32_t peer_mesh_id;
    uint32_t ring_addr;
    uint32_t pkt_hdr_ring_addr;
    uint32_t pkt_hdr_drain_addr;
    uint32_t drain_sink_addr;
    uint32_t batch;
    uint32_t filled_addr;
    uint32_t freed_addr;
    uint32_t fwd_sem_noc_x;
    uint32_t fwd_sem_noc_y;
    uint32_t fwd_sem_addr;

#ifndef KERNEL_BUILD
    // `downstream` is the worker serving this stream on the next chip: the sender bumps its
    // forwarded-token-count semaphore through the fabric packet header, which is why every worker placement
    // on the mesh is decided before any kernel is built.
    SenderCtArgs(
        const op::CombineFabric2dInputs& tensor_args,
        const op::StreamPlacement& self,
        const op::StreamPlacement& downstream,
        const op::L1Layout& l1,
        const op::KernelPlan& plan) :
        num_l1_slots(NUM_L1_SLOTS),
        token_size_bytes(op::token_size_bytes(tensor_args)),
        forwarding_metadata_size(FORWARDING_METADATA_SIZE),
        peer_chip_id(static_cast<uint32_t>(self.downstream_node.chip_id)),
        peer_mesh_id(*self.downstream_node.mesh_id),
        ring_addr(l1.ring),
        pkt_hdr_ring_addr(l1.pkt_hdr_ring),
        pkt_hdr_drain_addr(l1.pkt_hdr_drain),
        drain_sink_addr(l1.drain_sink),
        batch(BATCH),
        filled_addr(plan.ring_filled_addr),
        freed_addr(plan.ring_freed_addr),
        fwd_sem_noc_x(static_cast<uint32_t>(downstream.worker_virtual.x)),
        fwd_sem_noc_y(static_cast<uint32_t>(downstream.worker_virtual.y)),
        fwd_sem_addr(plan.fwd_arrived_addr) {}

    std::vector<uint32_t> to_ct_word_arr() const {
        return {
            num_l1_slots,
            token_size_bytes,
            forwarding_metadata_size,
            peer_chip_id,
            peer_mesh_id,
            ring_addr,
            pkt_hdr_ring_addr,
            pkt_hdr_drain_addr,
            drain_sink_addr,
            batch,
            filled_addr,
            freed_addr,
            fwd_sem_noc_x,
            fwd_sem_noc_y,
            fwd_sem_addr};
    }
#elif defined(CMBF2D_SENDER_KERNEL)
    // Only the role that owns these args may read them: `get_ct_arg` static_asserts the index against the
    // kernel's own argument count, and the other role's list is shorter.
    constexpr SenderCtArgs() :
        num_l1_slots(CMBF2D_CT(SenderCtArg, NumL1Slots)),
        token_size_bytes(CMBF2D_CT(SenderCtArg, TokenSizeBytes)),
        forwarding_metadata_size(CMBF2D_CT(SenderCtArg, ForwardingMetadataSize)),
        peer_chip_id(CMBF2D_CT(SenderCtArg, PeerChipId)),
        peer_mesh_id(CMBF2D_CT(SenderCtArg, PeerMeshId)),
        ring_addr(CMBF2D_CT(SenderCtArg, RingAddr)),
        pkt_hdr_ring_addr(CMBF2D_CT(SenderCtArg, PktHdrRingAddr)),
        pkt_hdr_drain_addr(CMBF2D_CT(SenderCtArg, PktHdrDrainAddr)),
        drain_sink_addr(CMBF2D_CT(SenderCtArg, DrainSinkAddr)),
        batch(CMBF2D_CT(SenderCtArg, Batch)),
        filled_addr(CMBF2D_CT(SenderCtArg, FilledAddr)),
        freed_addr(CMBF2D_CT(SenderCtArg, FreedAddr)),
        fwd_sem_noc_x(CMBF2D_CT(SenderCtArg, FwdSemNocX)),
        fwd_sem_noc_y(CMBF2D_CT(SenderCtArg, FwdSemNocY)),
        fwd_sem_addr(CMBF2D_CT(SenderCtArg, FwdSemAddr)) {}
#endif

    constexpr uint32_t slot_stride() const { return token_size_bytes + forwarding_metadata_size; }
};

// Field order is the compile-time argument order, so it must match ReaderCtArg.
struct ReaderCtArgs {
    uint32_t num_l1_slots;
    uint32_t token_size_bytes;
    uint32_t forwarding_metadata_size;
    uint32_t batch;
    uint32_t ring_addr;
    uint32_t filled_addr;
    uint32_t freed_addr;
    uint32_t dram_in_base_addr;
    uint32_t dram_out_base_addr;
    uint32_t dram_fwd_base_addr;
    uint32_t fwd_chunks_per_quarter;
    uint32_t fwd_pages_per_chunk;
    uint32_t my_quarter;
    uint32_t num_incoming_chunks;
    uint32_t fwd_sem_addr;
    uint32_t nbr_chip_id;
    uint32_t num_assignments;
    uint32_t schedule_len;
    uint32_t dram_meta_base_addr;
    uint32_t dram_counts_base_addr;
    uint32_t dram_region_base_addr;
    uint32_t dram_expert_offsets_base_addr;
    uint32_t num_routed_experts;
    uint32_t experts_per_chip;
    uint32_t my_expert_base;
    uint32_t num_experts_per_tok;
    uint32_t dispatch_group_size;
    uint32_t local_split_count;
    uint32_t my_row;
    uint32_t control_addr;
    uint32_t meta_prefetch_cap;

#ifndef KERNEL_BUILD
    ReaderCtArgs(
        const op::CombineFabric2dParams& args,
        const op::CombineFabric2dInputs& tensor_args,
        const ttnn::MeshCoordinate& coord,
        const op::StreamPlacement& self,
        const std::vector<op::Assignment>& work,
        const op::L1Layout& l1,
        const op::KernelPlan& plan,
        const op::DramBuffers& dram) :
        num_l1_slots(NUM_L1_SLOTS),
        token_size_bytes(op::token_size_bytes(tensor_args)),
        forwarding_metadata_size(FORWARDING_METADATA_SIZE),
        batch(BATCH),
        ring_addr(l1.ring),
        filled_addr(plan.ring_filled_addr),
        freed_addr(plan.ring_freed_addr),
        dram_in_base_addr(static_cast<uint32_t>(dram.in->address())),
        dram_out_base_addr(static_cast<uint32_t>(dram.out->address())),
        dram_fwd_base_addr(static_cast<uint32_t>(dram.fwd->address())),
        fwd_chunks_per_quarter(op::relay_chunks_per_stream(op::ring_extent(args))),
        fwd_pages_per_chunk(plan.pages_per_chunk),
        // Doubles as this stream's share of the same-chip run, which it copies after the fabric work.
        my_quarter(plan.stream),
        num_incoming_chunks(op::relay_chunks_per_stream(op::ring_extent(args))),
        fwd_sem_addr(plan.fwd_arrived_addr),
        nbr_chip_id(static_cast<uint32_t>(self.downstream_node.chip_id)),
        num_assignments(count_own_assignments(work)),
        schedule_len(static_cast<uint32_t>(work.size())),
        dram_meta_base_addr(static_cast<uint32_t>(dram.meta->address())),
        dram_counts_base_addr(static_cast<uint32_t>(dram.counts->address())),
        dram_region_base_addr(static_cast<uint32_t>(dram.region->address())),
        dram_expert_offsets_base_addr(static_cast<uint32_t>(dram.expert_offsets->address())),
        num_routed_experts(op::num_routed_experts(tensor_args)),
        experts_per_chip(args.experts_per_chip),
        my_expert_base(plan.my_expert_base),
        num_experts_per_tok(args.num_experts_per_tok),
        dispatch_group_size(op::ring_extent(args)),
        local_split_count(op::stream_count(args.num_links)),
        my_row(op::my_row(args, coord)),
        control_addr(l1.control),
        meta_prefetch_cap(META_PREFETCH) {
        // Schedule: the work order, relays tagged. An own entry carries its index into the table that
        // follows.
        uint32_t own_idx = 0;
        for (const auto& w : work) {
            blocks_.push_back(w.is_relay ? (SCHED_FWD | w.relay_chunk) : own_idx++);
        }
        for (const auto& w : work) {
            if (w.is_relay) {
                continue;
            }
            blocks_.push_back(w.dst_chip_id);
            blocks_.push_back(w.dst_row);
            blocks_.push_back(w.split_idx);
            blocks_.push_back(w.split_count);
        }
    }

    std::vector<uint32_t> to_ct_word_arr() const {
        std::vector<uint32_t> word_arr{
            num_l1_slots,
            token_size_bytes,
            forwarding_metadata_size,
            batch,
            ring_addr,
            filled_addr,
            freed_addr,
            dram_in_base_addr,
            dram_out_base_addr,
            dram_fwd_base_addr,
            fwd_chunks_per_quarter,
            fwd_pages_per_chunk,
            my_quarter,
            num_incoming_chunks,
            fwd_sem_addr,
            nbr_chip_id,
            num_assignments,
            schedule_len,
            dram_meta_base_addr,
            dram_counts_base_addr,
            dram_region_base_addr,
            dram_expert_offsets_base_addr,
            num_routed_experts,
            experts_per_chip,
            my_expert_base,
            num_experts_per_tok,
            dispatch_group_size,
            local_split_count,
            my_row,
            control_addr,
            meta_prefetch_cap};
        word_arr.insert(word_arr.end(), blocks_.begin(), blocks_.end());
        return word_arr;
    }
#elif defined(CMBF2D_READER_KERNEL)
    // Only the role that owns these args may read them: `get_ct_arg` static_asserts the index against the
    // kernel's own argument count, and the other role's list is shorter.
    constexpr ReaderCtArgs() :
        num_l1_slots(CMBF2D_CT(ReaderCtArg, NumL1Slots)),
        token_size_bytes(CMBF2D_CT(ReaderCtArg, TokenSizeBytes)),
        forwarding_metadata_size(CMBF2D_CT(ReaderCtArg, ForwardingMetadataSize)),
        batch(CMBF2D_CT(ReaderCtArg, Batch)),
        ring_addr(CMBF2D_CT(ReaderCtArg, RingAddr)),
        filled_addr(CMBF2D_CT(ReaderCtArg, FilledAddr)),
        freed_addr(CMBF2D_CT(ReaderCtArg, FreedAddr)),
        dram_in_base_addr(CMBF2D_CT(ReaderCtArg, DramInBaseAddr)),
        dram_out_base_addr(CMBF2D_CT(ReaderCtArg, DramOutBaseAddr)),
        dram_fwd_base_addr(CMBF2D_CT(ReaderCtArg, DramFwdBaseAddr)),
        fwd_chunks_per_quarter(CMBF2D_CT(ReaderCtArg, FwdChunksPerQuarter)),
        fwd_pages_per_chunk(CMBF2D_CT(ReaderCtArg, FwdPagesPerChunk)),
        my_quarter(CMBF2D_CT(ReaderCtArg, MyQuarter)),
        num_incoming_chunks(CMBF2D_CT(ReaderCtArg, NumIncomingChunks)),
        fwd_sem_addr(CMBF2D_CT(ReaderCtArg, FwdSemAddr)),
        nbr_chip_id(CMBF2D_CT(ReaderCtArg, NbrChipId)),
        num_assignments(CMBF2D_CT(ReaderCtArg, NumAssignments)),
        schedule_len(CMBF2D_CT(ReaderCtArg, ScheduleLen)),
        dram_meta_base_addr(CMBF2D_CT(ReaderCtArg, DramMetaBaseAddr)),
        dram_counts_base_addr(CMBF2D_CT(ReaderCtArg, DramCountsBaseAddr)),
        dram_region_base_addr(CMBF2D_CT(ReaderCtArg, DramRegionBaseAddr)),
        dram_expert_offsets_base_addr(CMBF2D_CT(ReaderCtArg, DramExpertOffsetsBaseAddr)),
        num_routed_experts(CMBF2D_CT(ReaderCtArg, NumRoutedExperts)),
        experts_per_chip(CMBF2D_CT(ReaderCtArg, ExpertsPerChip)),
        my_expert_base(CMBF2D_CT(ReaderCtArg, MyExpertBase)),
        num_experts_per_tok(CMBF2D_CT(ReaderCtArg, NumExpertsPerTok)),
        dispatch_group_size(CMBF2D_CT(ReaderCtArg, DispatchGroupSize)),
        local_split_count(CMBF2D_CT(ReaderCtArg, LocalSplitCount)),
        my_row(CMBF2D_CT(ReaderCtArg, MyRow)),
        control_addr(CMBF2D_CT(ReaderCtArg, ControlAddr)),
        meta_prefetch_cap(CMBF2D_CT(ReaderCtArg, MetaPrefetchCap)) {}
#endif

    constexpr uint32_t slot_stride() const { return token_size_bytes + forwarding_metadata_size; }
    constexpr uint32_t schedule_base() const { return static_cast<uint32_t>(ReaderCtArg::Count); }
    constexpr uint32_t assignment_base() const { return schedule_base() + schedule_len; }
    constexpr uint32_t accessor_base() const { return assignment_base() + ASSIGNMENT_WORDS * num_assignments; }

#ifndef KERNEL_BUILD
private:
    static uint32_t count_own_assignments(const std::vector<op::Assignment>& work) {
        uint32_t num_own = 0;
        for (const auto& w : work) {
            num_own += w.is_relay ? 0u : 1u;
        }
        return num_own;
    }

    std::vector<uint32_t> blocks_;  // schedule then assignments, appended after the scalars
#endif
};

#ifdef KERNEL_BUILD
#undef CMBF2D_CT
#endif

}  // namespace cmbf2d
