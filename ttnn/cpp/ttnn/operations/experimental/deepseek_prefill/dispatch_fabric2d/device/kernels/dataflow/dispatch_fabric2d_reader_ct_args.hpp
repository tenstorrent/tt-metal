// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The reader kernel's compile-time and runtime arguments. Host and kernel index the SAME enums by name
// rather than counting positions, so the two cannot drift.
//
// Buffer addresses are RUNTIME args, never compile-time: an address describes an allocation, not a
// program, so held compile-time it survives into every program-cache hit and a hit dispatched against
// reallocated buffers reads and writes at the previous call's addresses. TensorAccessor args stay
// compile-time -- those describe the tensor, not the allocation.

#include "dispatch_fabric2d_kernel_interface.hpp"

namespace dspf2d {

// Buffer bindings, rewritten by the framework per dispatch.
struct ReaderRtArg {
    enum Idx : uint32_t {
        kInputAddr,
        kIndicesAddr,
        kExpertOffsetsAddr,
        kDispatchTableAddr,
        kCountsAddr,
        kRegionOffsetsAddr,
        kOutPayloadAddr,
        kOutMetaAddr,
        kFwdAddr,
        kCount,
    };
};

struct ReaderCtArgs {
    enum Idx : uint32_t {
        kNumL1Slots,
        kBatch,
        kTokenSizeBytes,
        kMetadataPageBytes,
        kForwardingMetadataSize,
        kSeqLen,
        kTopK,
        kNumRoutedExperts,
        kExpertsPerChip,
        kExtent,
        kMyRow,
        kMyChipId,
        kNbrChipId,
        kLinearizedCoord,
        kNumLinks,
        kStream,
        kMaxDispatchBufTokens,
        kIndicesPadStride,
        kRingAddr,
        kControlAddr,
        kFilledAddr,
        kFreedAddr,
        kFwdSemAddr,
        kFwdPagesPerStream,
        kNumOwn,
        kNumRelay,
        // Blocks appended after the scalars, in this order. Kept as base indices so a later field can be
        // added without renumbering anything the kernel already reads.
        kRingChipIdsBase,
        kAssignmentBase,
        kScheduleBase,
        kCount,
    };

    uint32_t num_l1_slots;
    uint32_t batch;
    uint32_t token_size_bytes;
    uint32_t metadata_page_bytes;
    uint32_t forwarding_metadata_size;
    uint32_t seq_len;
    uint32_t topk;
    uint32_t num_routed_experts;
    uint32_t experts_per_chip;
    uint32_t extent;
    uint32_t my_row;
    uint32_t my_chip_id;
    uint32_t nbr_chip_id;
    // Metadata field 0 is the source chip as the production op names it, so this op has to agree.
    uint32_t linearized_coord;
    uint32_t num_links;
    uint32_t stream;
    uint32_t max_dispatch_buf_tokens;
    // One 64-byte pad per token: a DRAM read needs a 64-byte-aligned L1 destination on Blackhole, and a
    // packed topk*2-byte record would put every token after the first at a wrong address.
    uint32_t indices_pad_stride;
    uint32_t ring_addr;
    uint32_t control_addr;
    uint32_t filled_addr;
    uint32_t freed_addr;
    uint32_t fwd_sem_addr;
    uint32_t fwd_pages_per_stream;
    uint32_t num_own;
    uint32_t num_relay;
    uint32_t ring_chip_ids_base;
    uint32_t assignment_base;
    uint32_t schedule_base;

#ifndef KERNEL_BUILD
    ReaderCtArgs(
        const op::DispatchFabric2dParams& args,
        uint32_t token_bytes,
        uint32_t metadata_bytes,
        uint32_t linearized,
        uint32_t row,
        uint32_t chip_id,
        uint32_t neighbour_chip_id,
        const op::L1Layout& l1,
        const op::KernelPlan& plan,
        uint32_t own_count,
        uint32_t relay_count) :
        num_l1_slots(NUM_L1_SLOTS),
        batch(BATCH),
        token_size_bytes(token_bytes),
        metadata_page_bytes(metadata_bytes),
        forwarding_metadata_size(FORWARDING_METADATA_SIZE),
        seq_len(args.seq_len_per_chip),
        topk(args.num_experts_per_tok),
        num_routed_experts(args.num_routed_experts),
        experts_per_chip(args.experts_per_chip),
        extent(plan.extent),
        my_row(row),
        my_chip_id(chip_id),
        nbr_chip_id(neighbour_chip_id),
        linearized_coord(linearized),
        num_links(args.num_links),
        stream(plan.stream),
        max_dispatch_buf_tokens(args.max_dispatch_buffer_token_size),
        indices_pad_stride(META_PAD_STRIDE * ((args.num_experts_per_tok * 2 + META_PAD_STRIDE - 1) / META_PAD_STRIDE)),
        ring_addr(l1.ring),
        control_addr(l1.control),
        filled_addr(plan.ring_filled_addr),
        freed_addr(plan.ring_freed_addr),
        fwd_sem_addr(plan.fwd_arrived_addr),
        fwd_pages_per_stream(plan.fwd_pages_per_stream),
        num_own(own_count),
        num_relay(relay_count),
        ring_chip_ids_base(kCount),
        assignment_base(kCount + args.device->shape()[args.axis]),
        schedule_base(kCount + args.device->shape()[args.axis] + own_count * ASSIGNMENT_WORDS) {}

    // Scalars, then ring_chip_ids, then the assignments, then the schedule. The three base indices above
    // are what the kernel walks these with, so they are computed from the same expressions.
    std::vector<uint32_t> to_ct_word_arr(
        const std::vector<uint32_t>& ring_chip_ids,
        const std::vector<uint32_t>& assignment_words,
        const std::vector<uint32_t>& schedule) const {
        constexpr uint32_t kUnset = 0xDEADBEEFu;
        std::vector<uint32_t> w(kCount, kUnset);
        w[kNumL1Slots] = num_l1_slots;
        w[kBatch] = batch;
        w[kTokenSizeBytes] = token_size_bytes;
        w[kMetadataPageBytes] = metadata_page_bytes;
        w[kForwardingMetadataSize] = forwarding_metadata_size;
        w[kSeqLen] = seq_len;
        w[kTopK] = topk;
        w[kNumRoutedExperts] = num_routed_experts;
        w[kExpertsPerChip] = experts_per_chip;
        w[kExtent] = extent;
        w[kMyRow] = my_row;
        w[kMyChipId] = my_chip_id;
        w[kNbrChipId] = nbr_chip_id;
        w[kLinearizedCoord] = linearized_coord;
        w[kNumLinks] = num_links;
        w[kStream] = stream;
        w[kMaxDispatchBufTokens] = max_dispatch_buf_tokens;
        w[kIndicesPadStride] = indices_pad_stride;
        w[kRingAddr] = ring_addr;
        w[kControlAddr] = control_addr;
        w[kFilledAddr] = filled_addr;
        w[kFreedAddr] = freed_addr;
        w[kFwdSemAddr] = fwd_sem_addr;
        w[kFwdPagesPerStream] = fwd_pages_per_stream;
        w[kNumOwn] = num_own;
        w[kNumRelay] = num_relay;
        w[kRingChipIdsBase] = ring_chip_ids_base;
        w[kAssignmentBase] = assignment_base;
        w[kScheduleBase] = schedule_base;
        for (uint32_t i = 0; i < kCount; i++) {
            TT_FATAL(w[i] != kUnset, "dispatch_fabric2d: reader compile-time arg {} was never assigned", i);
        }
        TT_FATAL(
            ring_chip_ids.size() == extent && assignment_words.size() == num_own * ASSIGNMENT_WORDS &&
                schedule.size() == num_own + num_relay,
            "dispatch_fabric2d: reader blocks are {}/{}/{} words but the kernel indexes {}/{}/{}",
            ring_chip_ids.size(),
            assignment_words.size(),
            schedule.size(),
            extent,
            num_own * ASSIGNMENT_WORDS,
            num_own + num_relay);
        w.insert(w.end(), ring_chip_ids.begin(), ring_chip_ids.end());
        w.insert(w.end(), assignment_words.begin(), assignment_words.end());
        w.insert(w.end(), schedule.begin(), schedule.end());
        return w;
    }
#else
    constexpr ReaderCtArgs() :
        num_l1_slots(get_compile_time_arg_val(kNumL1Slots)),
        batch(get_compile_time_arg_val(kBatch)),
        token_size_bytes(get_compile_time_arg_val(kTokenSizeBytes)),
        metadata_page_bytes(get_compile_time_arg_val(kMetadataPageBytes)),
        forwarding_metadata_size(get_compile_time_arg_val(kForwardingMetadataSize)),
        seq_len(get_compile_time_arg_val(kSeqLen)),
        topk(get_compile_time_arg_val(kTopK)),
        num_routed_experts(get_compile_time_arg_val(kNumRoutedExperts)),
        experts_per_chip(get_compile_time_arg_val(kExpertsPerChip)),
        extent(get_compile_time_arg_val(kExtent)),
        my_row(get_compile_time_arg_val(kMyRow)),
        my_chip_id(get_compile_time_arg_val(kMyChipId)),
        nbr_chip_id(get_compile_time_arg_val(kNbrChipId)),
        linearized_coord(get_compile_time_arg_val(kLinearizedCoord)),
        num_links(get_compile_time_arg_val(kNumLinks)),
        stream(get_compile_time_arg_val(kStream)),
        max_dispatch_buf_tokens(get_compile_time_arg_val(kMaxDispatchBufTokens)),
        indices_pad_stride(get_compile_time_arg_val(kIndicesPadStride)),
        ring_addr(get_compile_time_arg_val(kRingAddr)),
        control_addr(get_compile_time_arg_val(kControlAddr)),
        filled_addr(get_compile_time_arg_val(kFilledAddr)),
        freed_addr(get_compile_time_arg_val(kFreedAddr)),
        fwd_sem_addr(get_compile_time_arg_val(kFwdSemAddr)),
        fwd_pages_per_stream(get_compile_time_arg_val(kFwdPagesPerStream)),
        num_own(get_compile_time_arg_val(kNumOwn)),
        num_relay(get_compile_time_arg_val(kNumRelay)),
        ring_chip_ids_base(get_compile_time_arg_val(kRingChipIdsBase)),
        assignment_base(get_compile_time_arg_val(kAssignmentBase)),
        schedule_base(get_compile_time_arg_val(kScheduleBase)) {}
#endif

    constexpr uint32_t slot_stride() const { return token_size_bytes + forwarding_metadata_size; }
};

}  // namespace dspf2d
