// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The sender kernel's compile-time arguments. Host and kernel index the SAME enum by name rather than
// counting positions, so the two cannot drift: adding a field in the wrong place is a compile error on
// one side instead of a silently misread word on the other.
//
// The sender has scalars only -- every address it writes arrives per token in the slot's routing tail.

#include "dispatch_fabric2d_kernel_interface.hpp"

namespace dspf2d {

struct SenderCtArgs {
    enum Idx : uint32_t {
        kNumL1Slots,
        kTokenSizeBytes,
        kForwardingMetadataSize,
        kMetadataPageBytes,
        kPeerChipId,
        kPeerMeshId,
        kRingAddr,
        kPktHdrRingAddr,
        kPktHdrDrainAddr,
        kDrainSinkAddr,
        kBatch,
        kFilledAddr,
        kFreedAddr,
        kFwdSemNocX,
        kFwdSemNocY,
        kFwdSemAddr,
        kCount,
    };

    uint32_t num_l1_slots;
    uint32_t token_size_bytes;
    uint32_t forwarding_metadata_size;
    // Dispatch lands each token in TWO tensors at one page index, so the final hop is two writes and the
    // sender needs the metadata page size as well as the token's.
    uint32_t metadata_page_bytes;
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
    // arrived-page counter through the fabric packet header, which is why every worker placement on the
    // mesh is decided before any kernel is built.
    SenderCtArgs(
        uint32_t token_bytes,
        uint32_t metadata_bytes,
        const op::StreamPlacement& self,
        const op::StreamPlacement& downstream,
        const op::L1Layout& l1,
        const op::KernelPlan& plan) :
        num_l1_slots(NUM_L1_SLOTS),
        token_size_bytes(token_bytes),
        forwarding_metadata_size(FORWARDING_METADATA_SIZE),
        metadata_page_bytes(metadata_bytes),
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
        constexpr uint32_t kUnset = 0xDEADBEEFu;
        std::vector<uint32_t> w(kCount, kUnset);
        w[kNumL1Slots] = num_l1_slots;
        w[kTokenSizeBytes] = token_size_bytes;
        w[kForwardingMetadataSize] = forwarding_metadata_size;
        w[kMetadataPageBytes] = metadata_page_bytes;
        w[kPeerChipId] = peer_chip_id;
        w[kPeerMeshId] = peer_mesh_id;
        w[kRingAddr] = ring_addr;
        w[kPktHdrRingAddr] = pkt_hdr_ring_addr;
        w[kPktHdrDrainAddr] = pkt_hdr_drain_addr;
        w[kDrainSinkAddr] = drain_sink_addr;
        w[kBatch] = batch;
        w[kFilledAddr] = filled_addr;
        w[kFreedAddr] = freed_addr;
        w[kFwdSemNocX] = fwd_sem_noc_x;
        w[kFwdSemNocY] = fwd_sem_noc_y;
        w[kFwdSemAddr] = fwd_sem_addr;
        for (uint32_t i = 0; i < kCount; i++) {
            TT_FATAL(w[i] != kUnset, "dispatch_fabric2d: sender compile-time arg {} was never assigned", i);
        }
        return w;
    }
#else
    constexpr SenderCtArgs() :
        num_l1_slots(get_compile_time_arg_val(kNumL1Slots)),
        token_size_bytes(get_compile_time_arg_val(kTokenSizeBytes)),
        forwarding_metadata_size(get_compile_time_arg_val(kForwardingMetadataSize)),
        metadata_page_bytes(get_compile_time_arg_val(kMetadataPageBytes)),
        peer_chip_id(get_compile_time_arg_val(kPeerChipId)),
        peer_mesh_id(get_compile_time_arg_val(kPeerMeshId)),
        ring_addr(get_compile_time_arg_val(kRingAddr)),
        pkt_hdr_ring_addr(get_compile_time_arg_val(kPktHdrRingAddr)),
        pkt_hdr_drain_addr(get_compile_time_arg_val(kPktHdrDrainAddr)),
        drain_sink_addr(get_compile_time_arg_val(kDrainSinkAddr)),
        batch(get_compile_time_arg_val(kBatch)),
        filled_addr(get_compile_time_arg_val(kFilledAddr)),
        freed_addr(get_compile_time_arg_val(kFreedAddr)),
        fwd_sem_noc_x(get_compile_time_arg_val(kFwdSemNocX)),
        fwd_sem_noc_y(get_compile_time_arg_val(kFwdSemNocY)),
        fwd_sem_addr(get_compile_time_arg_val(kFwdSemAddr)) {}
#endif

    constexpr uint32_t slot_stride() const { return token_size_bytes + forwarding_metadata_size; }
};

}  // namespace dspf2d
