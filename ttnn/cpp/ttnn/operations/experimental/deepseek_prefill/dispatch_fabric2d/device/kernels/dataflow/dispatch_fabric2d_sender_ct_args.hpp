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
        kPktHdrCreditAddr,
        kDrainSinkAddr,
        kBatch,
        kFilledAddr,
        kFreedAddr,
        kDownstreamNocX,
        kDownstreamNocY,
        kFinalSemAddr,
        kCreditNocX,
        kCreditNocY,
        kLaunchCreditAddr,
        kNumVcs,
        kVcPages,
        kCreditDoneAddr,
        // One word per channel, appended after the scalars: the downstream reader's arrival counter, the
        // credit counter on the peer's opposite-direction core, and our own reader's outbox. Global
        // semaphores sit at the same address on every chip, so one block serves both ends.
        kVcArrivedBase,
        kVcFreedBase,
        kVcOutboxBase,
        kCount,
    };

    uint32_t queue_depth;
    uint32_t token_size_bytes;
    uint32_t forwarding_metadata_size;
    uint32_t downstream_chip_id;
    uint32_t downstream_mesh_id;
    uint32_t queue_addr;
    uint32_t pkt_hdr_queue_addr;
    uint32_t pkt_hdr_signal_addr;  // bump_downstream's header, reused by drain_fabric
    uint32_t pkt_hdr_credit_addr;
    uint32_t drain_sink_addr;
    uint32_t batch;
    uint32_t filled_addr;
    uint32_t freed_addr;
    uint32_t downstream_noc_x;  // the downstream stream core
    uint32_t downstream_noc_y;
    uint32_t final_sem_addr;  // the downstream reader's last-hop counter
    uint32_t credit_noc_x;  // the downstream chip's stream core that sends back to this chip
    uint32_t credit_noc_y;
    uint32_t launch_credit_addr;
    uint32_t num_vcs;
    uint32_t vc_pages;
    uint32_t credit_done_addr;
    uint32_t vc_arrived_base = kCount;
    uint32_t vc_freed_base;
    uint32_t vc_outbox_base;

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
        const op::KernelPlan& plan) :
        queue_depth(QUEUE_DEPTH),
        token_size_bytes(token_bytes),
        forwarding_metadata_size(FORWARDING_METADATA_SIZE),
        downstream_chip_id(static_cast<uint32_t>(self.downstream_node.chip_id)),
        downstream_mesh_id(*self.downstream_node.mesh_id),
        queue_addr(l1.queue),
        pkt_hdr_queue_addr(l1.pkt_hdr_queue),
        pkt_hdr_signal_addr(l1.pkt_hdr_signal),
        pkt_hdr_credit_addr(l1.pkt_hdr_credit),
        drain_sink_addr(l1.drain_sink),
        batch(BATCH),
        filled_addr(plan.queue_filled_addr),
        freed_addr(plan.queue_freed_addr),
        downstream_noc_x(static_cast<uint32_t>(downstream.worker_virtual.x)),
        downstream_noc_y(static_cast<uint32_t>(downstream.worker_virtual.y)),
        final_sem_addr(plan.final_arrived_addr),
        credit_noc_x(static_cast<uint32_t>(returning.worker_virtual.x)),
        credit_noc_y(static_cast<uint32_t>(returning.worker_virtual.y)),
        launch_credit_addr(plan.launch_credit_addr),
        num_vcs(static_cast<uint32_t>(plan.vc_arrived_addr.size())),
        vc_pages(plan.vc_pages),
        credit_done_addr(plan.credit_done_addr),
        vc_freed_base(vc_arrived_base + num_vcs),
        vc_outbox_base(vc_freed_base + num_vcs) {}

    std::vector<uint32_t> to_ct_word_arr(
        const std::vector<uint32_t>& vc_arrived,
        const std::vector<uint32_t>& vc_freed,
        const std::vector<uint32_t>& vc_outbox) const {
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
        w[kPktHdrCreditAddr] = pkt_hdr_credit_addr;
        w[kDrainSinkAddr] = drain_sink_addr;
        w[kBatch] = batch;
        w[kFilledAddr] = filled_addr;
        w[kFreedAddr] = freed_addr;
        w[kDownstreamNocX] = downstream_noc_x;
        w[kDownstreamNocY] = downstream_noc_y;
        w[kFinalSemAddr] = final_sem_addr;
        w[kCreditNocX] = credit_noc_x;
        w[kCreditNocY] = credit_noc_y;
        w[kLaunchCreditAddr] = launch_credit_addr;
        w[kNumVcs] = num_vcs;
        w[kVcPages] = vc_pages;
        w[kCreditDoneAddr] = credit_done_addr;
        w[kVcArrivedBase] = vc_arrived_base;
        w[kVcFreedBase] = vc_freed_base;
        w[kVcOutboxBase] = vc_outbox_base;
        for (uint32_t i = 0; i < kCount; i++) {
            TT_FATAL(w[i] != kUnset, "dispatch_fabric2d: sender compile-time arg {} was never assigned", i);
        }
        TT_FATAL(
            vc_arrived.size() == num_vcs && vc_freed.size() == num_vcs && vc_outbox.size() == num_vcs,
            "dispatch_fabric2d: sender channel blocks are {}/{}/{} words but the kernel indexes {}",
            vc_arrived.size(),
            vc_freed.size(),
            vc_outbox.size(),
            num_vcs);
        w.insert(w.end(), vc_arrived.begin(), vc_arrived.end());
        w.insert(w.end(), vc_freed.begin(), vc_freed.end());
        w.insert(w.end(), vc_outbox.begin(), vc_outbox.end());
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
        pkt_hdr_credit_addr(get_compile_time_arg_val(kPktHdrCreditAddr)),
        drain_sink_addr(get_compile_time_arg_val(kDrainSinkAddr)),
        batch(get_compile_time_arg_val(kBatch)),
        filled_addr(get_compile_time_arg_val(kFilledAddr)),
        freed_addr(get_compile_time_arg_val(kFreedAddr)),
        downstream_noc_x(get_compile_time_arg_val(kDownstreamNocX)),
        downstream_noc_y(get_compile_time_arg_val(kDownstreamNocY)),
        final_sem_addr(get_compile_time_arg_val(kFinalSemAddr)),
        credit_noc_x(get_compile_time_arg_val(kCreditNocX)),
        credit_noc_y(get_compile_time_arg_val(kCreditNocY)),
        launch_credit_addr(get_compile_time_arg_val(kLaunchCreditAddr)),
        num_vcs(get_compile_time_arg_val(kNumVcs)),
        vc_pages(get_compile_time_arg_val(kVcPages)),
        credit_done_addr(get_compile_time_arg_val(kCreditDoneAddr)),
        vc_arrived_base(get_compile_time_arg_val(kVcArrivedBase)),
        vc_freed_base(get_compile_time_arg_val(kVcFreedBase)),
        vc_outbox_base(get_compile_time_arg_val(kVcOutboxBase)) {}
#endif

    constexpr uint32_t entry_stride() const { return token_size_bytes + forwarding_metadata_size; }

#ifdef KERNEL_BUILD
    uint32_t vc_arrived_addr(uint32_t vc) const { return kernel_compile_time_args[vc_arrived_base + vc]; }
    uint32_t vc_freed_addr(uint32_t vc) const { return kernel_compile_time_args[vc_freed_base + vc]; }
    uint32_t vc_outbox_addr(uint32_t vc) const { return kernel_compile_time_args[vc_outbox_base + vc]; }
#endif
};

}  // namespace dspf2d
