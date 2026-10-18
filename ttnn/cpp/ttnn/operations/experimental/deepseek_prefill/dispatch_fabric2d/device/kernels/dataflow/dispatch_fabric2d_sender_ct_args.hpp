// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The sender kernel's compile-time arguments. Host and kernel index the same enum by name, so the
// argument order cannot drift between them.
//
// The sender takes scalars only: each address it writes to arrives with the token in its fwd_meta.

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
        kDrainSinkAddr,
        kBatch,
        kFilledAddr,
        kFreedAddr,
        kDownstreamNocX,
        kDownstreamNocY,
        kFwdSemAddr,
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
    uint32_t drain_sink_addr;
    uint32_t batch;
    uint32_t filled_addr;
    uint32_t freed_addr;
    uint32_t downstream_noc_x;  // the downstream stream core
    uint32_t downstream_noc_y;
    uint32_t fwd_sem_addr;

#ifndef KERNEL_BUILD
    // `downstream` is the core serving this stream on the next chip; the sender signals its arrived-page
    // counter, so placement must be decided on every chip before any kernel is built.
    SenderCtArgs(
        uint32_t token_bytes,
        const op::StreamPlacement& self,
        const op::StreamPlacement& downstream,
        const op::L1Layout& l1,
        const op::KernelPlan& plan) :
        queue_depth(QUEUE_DEPTH),
        token_size_bytes(token_bytes),
        forwarding_metadata_size(FORWARDING_METADATA_SIZE),
        downstream_chip_id(static_cast<uint32_t>(self.downstream_node.chip_id)),
        downstream_mesh_id(*self.downstream_node.mesh_id),
        queue_addr(l1.queue),
        pkt_hdr_queue_addr(l1.pkt_hdr_queue),
        pkt_hdr_signal_addr(l1.pkt_hdr_signal),
        drain_sink_addr(l1.drain_sink),
        batch(BATCH),
        filled_addr(plan.queue_filled_addr),
        freed_addr(plan.queue_freed_addr),
        downstream_noc_x(static_cast<uint32_t>(downstream.worker_virtual.x)),
        downstream_noc_y(static_cast<uint32_t>(downstream.worker_virtual.y)),
        fwd_sem_addr(plan.fwd_arrived_addr) {}

    std::vector<uint32_t> to_ct_word_arr() const {
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
        w[kDrainSinkAddr] = drain_sink_addr;
        w[kBatch] = batch;
        w[kFilledAddr] = filled_addr;
        w[kFreedAddr] = freed_addr;
        w[kDownstreamNocX] = downstream_noc_x;
        w[kDownstreamNocY] = downstream_noc_y;
        w[kFwdSemAddr] = fwd_sem_addr;
        for (uint32_t i = 0; i < kCount; i++) {
            TT_FATAL(w[i] != kUnset, "dispatch_fabric2d: sender compile-time arg {} was never assigned", i);
        }
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
        drain_sink_addr(get_compile_time_arg_val(kDrainSinkAddr)),
        batch(get_compile_time_arg_val(kBatch)),
        filled_addr(get_compile_time_arg_val(kFilledAddr)),
        freed_addr(get_compile_time_arg_val(kFreedAddr)),
        downstream_noc_x(get_compile_time_arg_val(kDownstreamNocX)),
        downstream_noc_y(get_compile_time_arg_val(kDownstreamNocY)),
        fwd_sem_addr(get_compile_time_arg_val(kFwdSemAddr)) {}
#endif

    constexpr uint32_t entry_stride() const { return token_size_bytes + forwarding_metadata_size; }
};

}  // namespace dspf2d
