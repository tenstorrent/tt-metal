// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The reader kernel's compile-time and runtime arguments. Host and kernel index the same enums by name,
// so the argument order cannot drift between them.
//
// Buffer addresses are runtime args because a program-cache hit may run against reallocated buffers.
// TensorAccessor args describe the tensor and stay compile-time.

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
        // Bound unconditionally so host and kernel never disagree about the arg layout; with no config
        // supplied it points at a stand-in the kernel does not read.
        kPaddingConfigAddr,
        kCount,
    };
};

struct ReaderCtArgs {
    enum Idx : uint32_t {
        kQueueDepth,
        kBatch,
        kTokenSizeBytes,
        kForwardingMetadataSize,
        kSeqLen,
        kTopK,
        kNumRoutedExperts,
        kExpertsPerChip,
        kExtent,
        kMyRow,
        kNbrChipId,
        kLinearizedCoord,
        kNumLinks,
        kStream,
        kMaxDispatchBufTokens,
        kIndicesPadStride,
        kQueueAddr,
        kControlAddr,
        kFilledAddr,
        kFreedAddr,
        kFwdSemAddr,
        kFwdPagesPerStream,
        kNumOwn,
        kNumForward,
        kUntilizeSemAddr,
        kUntilizeTileRows,
        kHasPaddingConfig,
        // Blocks appended after the scalars, in this order. Kept as base indices so a later field can be
        // added without renumbering anything the kernel already reads.
        kRingChipIdsBase,
        kAssignmentBase,
        // (origin_row, dst_row, split_idx, split_count) per descriptor. `in` lists what this stream reads
        // from its own fwd_section, `out` what it writes into the downstream chip's. The host's
        // validate_chunk_agreement checks that `out` here matches `in` on the downstream chip in content
        // and order, so a writer can place a chunk from its own list alone.
        kInChunksBase,
        kOutChunksBase,
        kCount,
    };

    uint32_t queue_depth;
    uint32_t batch;
    uint32_t token_size_bytes;
    uint32_t forwarding_metadata_size;
    uint32_t seq_len;
    uint32_t topk;
    uint32_t num_routed_experts;
    uint32_t experts_per_chip;
    uint32_t extent;
    uint32_t my_row;
    uint32_t nbr_chip_id;
    // Written as metadata field 0, the source chip.
    uint32_t linearized_coord;
    uint32_t num_links;
    uint32_t stream;
    uint32_t max_dispatch_buf_tokens;
    // One 64-byte pad per token: a DRAM read needs a 64-byte-aligned L1 destination on Blackhole, and a
    // packed topk*2-byte record would put every token after the first at a wrong address.
    uint32_t indices_pad_stride;
    uint32_t queue_addr;
    uint32_t control_addr;
    uint32_t filled_addr;
    uint32_t freed_addr;
    uint32_t fwd_sem_addr;
    uint32_t fwd_pages_per_stream;
    uint32_t num_own;
    uint32_t num_forward;
    // A TILE input reaches this reader through a staging buffer the untilizer pool fills. Zero
    // tile rows is the row-major path: the input accessor already points at the tokens.
    uint32_t untilize_sem_addr;
    uint32_t untilize_tile_rows;
    // Read the two padding words and bound the routing pass at the real token count.
    uint32_t has_padding_config;
    uint32_t ring_chip_ids_base;
    uint32_t assignment_base;
    uint32_t in_chunks_base;
    uint32_t out_chunks_base;

#ifndef KERNEL_BUILD
    ReaderCtArgs(
        const op::DispatchFabric2dParams& args,
        uint32_t token_bytes,
        uint32_t linearized,
        uint32_t row,
        uint32_t neighbour_chip_id,
        const op::L1Layout& l1,
        const op::KernelPlan& plan,
        uint32_t own_count,
        uint32_t forward_count) :
        queue_depth(QUEUE_DEPTH),
        batch(BATCH),
        token_size_bytes(token_bytes),
        forwarding_metadata_size(FORWARDING_METADATA_SIZE),
        seq_len(args.seq_len_per_chip),
        topk(args.num_experts_per_tok),
        num_routed_experts(args.num_routed_experts),
        experts_per_chip(args.experts_per_chip),
        extent(plan.extent),
        my_row(row),
        nbr_chip_id(neighbour_chip_id),
        linearized_coord(linearized),
        num_links(args.num_links),
        stream(plan.stream),
        max_dispatch_buf_tokens(args.max_dispatch_buffer_token_size),
        indices_pad_stride(META_PAD_STRIDE * ((args.num_experts_per_tok * 2 + META_PAD_STRIDE - 1) / META_PAD_STRIDE)),
        queue_addr(l1.queue),
        control_addr(l1.control),
        filled_addr(plan.queue_filled_addr),
        freed_addr(plan.queue_freed_addr),
        fwd_sem_addr(plan.fwd_arrived_addr),
        fwd_pages_per_stream(plan.fwd_pages_per_stream),
        num_own(own_count),
        num_forward(forward_count),
        untilize_sem_addr(plan.untilize_sem_addr),
        untilize_tile_rows(plan.untilize_tile_rows),
        has_padding_config(args.has_padding_config ? 1u : 0u),
        ring_chip_ids_base(kCount),
        assignment_base(kCount + args.device->shape()[args.axis]),
        in_chunks_base(assignment_base + own_count * ASSIGNMENT_WORDS),
        out_chunks_base(in_chunks_base + forward_count * ASSIGNMENT_WORDS) {}

    // Scalars, then ring_chip_ids, the assignments and the two chunk-descriptor blocks, at the base
    // indices set in the constructor.
    std::vector<uint32_t> to_ct_word_arr(
        const std::vector<uint32_t>& ring_chip_ids,
        const std::vector<uint32_t>& assignment_words,
        const std::vector<uint32_t>& in_chunks,
        const std::vector<uint32_t>& out_chunks) const {
        constexpr uint32_t kUnset = 0xDEADBEEFu;
        std::vector<uint32_t> w(kCount, kUnset);
        w[kQueueDepth] = queue_depth;
        w[kBatch] = batch;
        w[kTokenSizeBytes] = token_size_bytes;
        w[kForwardingMetadataSize] = forwarding_metadata_size;
        w[kSeqLen] = seq_len;
        w[kTopK] = topk;
        w[kNumRoutedExperts] = num_routed_experts;
        w[kExpertsPerChip] = experts_per_chip;
        w[kExtent] = extent;
        w[kMyRow] = my_row;
        w[kNbrChipId] = nbr_chip_id;
        w[kLinearizedCoord] = linearized_coord;
        w[kNumLinks] = num_links;
        w[kStream] = stream;
        w[kMaxDispatchBufTokens] = max_dispatch_buf_tokens;
        w[kIndicesPadStride] = indices_pad_stride;
        w[kQueueAddr] = queue_addr;
        w[kControlAddr] = control_addr;
        w[kFilledAddr] = filled_addr;
        w[kFreedAddr] = freed_addr;
        w[kFwdSemAddr] = fwd_sem_addr;
        w[kFwdPagesPerStream] = fwd_pages_per_stream;
        w[kNumOwn] = num_own;
        w[kNumForward] = num_forward;
        w[kUntilizeSemAddr] = untilize_sem_addr;
        w[kUntilizeTileRows] = untilize_tile_rows;
        w[kHasPaddingConfig] = has_padding_config;
        w[kRingChipIdsBase] = ring_chip_ids_base;
        w[kAssignmentBase] = assignment_base;
        w[kInChunksBase] = in_chunks_base;
        w[kOutChunksBase] = out_chunks_base;
        for (uint32_t i = 0; i < kCount; i++) {
            TT_FATAL(w[i] != kUnset, "dispatch_fabric2d: reader compile-time arg {} was never assigned", i);
        }
        TT_FATAL(
            ring_chip_ids.size() == extent && assignment_words.size() == num_own * ASSIGNMENT_WORDS,
            "dispatch_fabric2d: reader blocks are {}/{} words but the kernel indexes {}/{}",
            ring_chip_ids.size(),
            assignment_words.size(),
            extent,
            num_own * ASSIGNMENT_WORDS);
        TT_FATAL(
            in_chunks.size() == num_forward * ASSIGNMENT_WORDS && out_chunks.size() == num_forward * ASSIGNMENT_WORDS,
            "dispatch_fabric2d: chunk descriptor blocks are {}/{} words but the kernel indexes {} each",
            in_chunks.size(),
            out_chunks.size(),
            num_forward * ASSIGNMENT_WORDS);
        w.insert(w.end(), ring_chip_ids.begin(), ring_chip_ids.end());
        w.insert(w.end(), assignment_words.begin(), assignment_words.end());
        w.insert(w.end(), in_chunks.begin(), in_chunks.end());
        w.insert(w.end(), out_chunks.begin(), out_chunks.end());
        return w;
    }
#else
    constexpr ReaderCtArgs() :
        queue_depth(get_compile_time_arg_val(kQueueDepth)),
        batch(get_compile_time_arg_val(kBatch)),
        token_size_bytes(get_compile_time_arg_val(kTokenSizeBytes)),
        forwarding_metadata_size(get_compile_time_arg_val(kForwardingMetadataSize)),
        seq_len(get_compile_time_arg_val(kSeqLen)),
        topk(get_compile_time_arg_val(kTopK)),
        num_routed_experts(get_compile_time_arg_val(kNumRoutedExperts)),
        experts_per_chip(get_compile_time_arg_val(kExpertsPerChip)),
        extent(get_compile_time_arg_val(kExtent)),
        my_row(get_compile_time_arg_val(kMyRow)),
        nbr_chip_id(get_compile_time_arg_val(kNbrChipId)),
        linearized_coord(get_compile_time_arg_val(kLinearizedCoord)),
        num_links(get_compile_time_arg_val(kNumLinks)),
        stream(get_compile_time_arg_val(kStream)),
        max_dispatch_buf_tokens(get_compile_time_arg_val(kMaxDispatchBufTokens)),
        indices_pad_stride(get_compile_time_arg_val(kIndicesPadStride)),
        queue_addr(get_compile_time_arg_val(kQueueAddr)),
        control_addr(get_compile_time_arg_val(kControlAddr)),
        filled_addr(get_compile_time_arg_val(kFilledAddr)),
        freed_addr(get_compile_time_arg_val(kFreedAddr)),
        fwd_sem_addr(get_compile_time_arg_val(kFwdSemAddr)),
        fwd_pages_per_stream(get_compile_time_arg_val(kFwdPagesPerStream)),
        num_own(get_compile_time_arg_val(kNumOwn)),
        num_forward(get_compile_time_arg_val(kNumForward)),
        untilize_sem_addr(get_compile_time_arg_val(kUntilizeSemAddr)),
        untilize_tile_rows(get_compile_time_arg_val(kUntilizeTileRows)),
        has_padding_config(get_compile_time_arg_val(kHasPaddingConfig)),
        ring_chip_ids_base(get_compile_time_arg_val(kRingChipIdsBase)),
        assignment_base(get_compile_time_arg_val(kAssignmentBase)),
        in_chunks_base(get_compile_time_arg_val(kInChunksBase)),
        out_chunks_base(get_compile_time_arg_val(kOutChunksBase)) {}
#endif

    constexpr uint32_t entry_stride() const { return token_size_bytes + forwarding_metadata_size; }

#ifdef KERNEL_BUILD
    // The program factory appends the TensorAccessorArgs after the blocks above, in ReaderRtArg order.
    // Their base is derived from the block bases, so adding a scalar or widening a block keeps it right.
    static constexpr uint32_t accessor_base =
        get_compile_time_arg_val(kOutChunksBase) + get_compile_time_arg_val(kNumForward) * ASSIGNMENT_WORDS;
    static constexpr auto in_args = TensorAccessorArgs<accessor_base>();
    static constexpr auto indices_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    static constexpr auto offsets_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    static constexpr auto table_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    static constexpr auto counts_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    static constexpr auto region_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    static constexpr auto out_payload_args = TensorAccessorArgs<region_args.next_compile_time_args_offset()>();
    static constexpr auto out_meta_args = TensorAccessorArgs<out_payload_args.next_compile_time_args_offset()>();
    static constexpr auto fwd_args = TensorAccessorArgs<out_meta_args.next_compile_time_args_offset()>();
    static constexpr auto padding_args = TensorAccessorArgs<fwd_args.next_compile_time_args_offset()>();
#endif
};

}  // namespace dspf2d
