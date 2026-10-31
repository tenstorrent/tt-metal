// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The sender kernel's compile-time arguments. Host and kernel index the same enum by name, so the
// argument order cannot drift between them.
//
// Each address the sender writes a token to arrives with the token in its fwd_meta. Its own DRAM access is the
// zeroing of its share of the metadata output, through the accessor appended after the stream core block.

#include "dispatch_fabric2d_kernel_interface.hpp"

namespace dspf2d {

struct SenderCtArgs {
    enum Idx : uint32_t {
        kQueueDepth,
        kTokenSizeBytes,
        kForwardingMetadataSize,
        kDownstreamChipId,
        kDownstreamMeshId,
        kQueueAddr,
        kPktHdrQueueAddr,
        kPktHdrSignalAddr,
        kBatch,
        kFilledAddr,
        kFreedAddr,
        kDownstreamNocX,
        kDownstreamNocY,
        kFwdSemAddr,
        kCreditNocX,
        kCreditNocY,
        kLaunchCreditAddr,
        kStream,
        kNumStreams,
        kMetaPages,
        kMetaPageBytes,
        // (noc_x, noc_y) of every stream core on this chip follow the fixed args, then the metadata accessor.
        kCount,
    };

    uint32_t queue_depth;
    uint32_t token_size_bytes;
    uint32_t forwarding_metadata_size;
    uint32_t downstream_chip_id;
    uint32_t downstream_mesh_id;
    uint32_t queue_addr;
    uint32_t pkt_hdr_queue_addr;
    uint32_t pkt_hdr_signal_addr;  // signal_downstream's header, reused by drain_fabric
    uint32_t batch;
    uint32_t filled_addr;
    uint32_t freed_addr;
    uint32_t downstream_noc_x;  // the downstream stream core
    uint32_t downstream_noc_y;
    uint32_t fwd_sem_addr;
    uint32_t credit_noc_x;  // the downstream chip's stream core that sends back to this chip
    uint32_t credit_noc_y;
    uint32_t launch_credit_addr;
    uint32_t stream;
    uint32_t num_streams;
    uint32_t meta_pages;
    uint32_t meta_page_bytes;

#ifndef KERNEL_BUILD
    // `downstream` is the core serving this stream on the next chip; the sender signals its arrived-page
    // counter. `returning` is the core on that chip whose sender writes back to this chip; the sender grants
    // it the launch credit. So placement must be decided on every chip before any kernel is built.
    SenderCtArgs(
        uint32_t token_bytes,
        const op::StreamPlacement& self,
        const op::StreamPlacement& downstream,
        const op::StreamPlacement& returning,
        const op::L1Layout& l1,
        const op::KernelPlan& plan,
        uint32_t streams,
        uint32_t metadata_pages,
        uint32_t metadata_page_bytes) :
        queue_depth(QUEUE_DEPTH),
        token_size_bytes(token_bytes),
        forwarding_metadata_size(FORWARDING_METADATA_SIZE),
        downstream_chip_id(static_cast<uint32_t>(self.downstream_node.chip_id)),
        downstream_mesh_id(*self.downstream_node.mesh_id),
        queue_addr(l1.queue),
        pkt_hdr_queue_addr(l1.pkt_hdr_queue),
        pkt_hdr_signal_addr(l1.pkt_hdr_signal),
        batch(BATCH),
        filled_addr(plan.queue_filled_addr),
        freed_addr(plan.queue_freed_addr),
        downstream_noc_x(static_cast<uint32_t>(downstream.worker_virtual.x)),
        downstream_noc_y(static_cast<uint32_t>(downstream.worker_virtual.y)),
        fwd_sem_addr(plan.fwd_arrived_addr),
        credit_noc_x(static_cast<uint32_t>(returning.worker_virtual.x)),
        credit_noc_y(static_cast<uint32_t>(returning.worker_virtual.y)),
        launch_credit_addr(plan.launch_credit_addr),
        stream(plan.stream),
        num_streams(streams),
        meta_pages(metadata_pages),
        meta_page_bytes(metadata_page_bytes) {}

    std::vector<uint32_t> to_ct_word_arr(const std::vector<uint32_t>& stream_noc_words) const {
        constexpr uint32_t kUnset = 0xDEADBEEFu;
        std::vector<uint32_t> w(kCount, kUnset);
        w[kQueueDepth] = queue_depth;
        w[kTokenSizeBytes] = token_size_bytes;
        w[kForwardingMetadataSize] = forwarding_metadata_size;
        w[kDownstreamChipId] = downstream_chip_id;
        w[kDownstreamMeshId] = downstream_mesh_id;
        w[kQueueAddr] = queue_addr;
        w[kPktHdrQueueAddr] = pkt_hdr_queue_addr;
        w[kPktHdrSignalAddr] = pkt_hdr_signal_addr;
        w[kBatch] = batch;
        w[kFilledAddr] = filled_addr;
        w[kFreedAddr] = freed_addr;
        w[kDownstreamNocX] = downstream_noc_x;
        w[kDownstreamNocY] = downstream_noc_y;
        w[kFwdSemAddr] = fwd_sem_addr;
        w[kCreditNocX] = credit_noc_x;
        w[kCreditNocY] = credit_noc_y;
        w[kLaunchCreditAddr] = launch_credit_addr;
        w[kStream] = stream;
        w[kNumStreams] = num_streams;
        w[kMetaPages] = meta_pages;
        w[kMetaPageBytes] = meta_page_bytes;
        for (uint32_t i = 0; i < kCount; i++) {
            TT_FATAL(w[i] != kUnset, "dispatch_fabric2d: sender compile-time arg {} was never assigned", i);
        }
        TT_FATAL(
            stream_noc_words.size() == 2u * num_streams,
            "dispatch_fabric2d: sender stream core block is {} words but the kernel indexes {}",
            stream_noc_words.size(),
            2u * num_streams);
        w.insert(w.end(), stream_noc_words.begin(), stream_noc_words.end());
        return w;
    }
#else
    constexpr SenderCtArgs() :
        queue_depth(get_compile_time_arg_val(kQueueDepth)),
        token_size_bytes(get_compile_time_arg_val(kTokenSizeBytes)),
        forwarding_metadata_size(get_compile_time_arg_val(kForwardingMetadataSize)),
        downstream_chip_id(get_compile_time_arg_val(kDownstreamChipId)),
        downstream_mesh_id(get_compile_time_arg_val(kDownstreamMeshId)),
        queue_addr(get_compile_time_arg_val(kQueueAddr)),
        pkt_hdr_queue_addr(get_compile_time_arg_val(kPktHdrQueueAddr)),
        pkt_hdr_signal_addr(get_compile_time_arg_val(kPktHdrSignalAddr)),
        batch(get_compile_time_arg_val(kBatch)),
        filled_addr(get_compile_time_arg_val(kFilledAddr)),
        freed_addr(get_compile_time_arg_val(kFreedAddr)),
        downstream_noc_x(get_compile_time_arg_val(kDownstreamNocX)),
        downstream_noc_y(get_compile_time_arg_val(kDownstreamNocY)),
        fwd_sem_addr(get_compile_time_arg_val(kFwdSemAddr)),
        credit_noc_x(get_compile_time_arg_val(kCreditNocX)),
        credit_noc_y(get_compile_time_arg_val(kCreditNocY)),
        launch_credit_addr(get_compile_time_arg_val(kLaunchCreditAddr)),
        stream(get_compile_time_arg_val(kStream)),
        num_streams(get_compile_time_arg_val(kNumStreams)),
        meta_pages(get_compile_time_arg_val(kMetaPages)),
        meta_page_bytes(get_compile_time_arg_val(kMetaPageBytes)) {}
#endif

    constexpr uint32_t entry_stride() const { return token_size_bytes + forwarding_metadata_size; }

#ifdef KERNEL_BUILD
    static constexpr uint32_t stream_noc_base = kCount;
    static constexpr auto meta_args = TensorAccessorArgs<kCount + 2u * get_compile_time_arg_val(kNumStreams)>();
#endif
};

}  // namespace dspf2d
