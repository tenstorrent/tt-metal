// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The sender kernel's compile-time arguments. One field list: the host constructor takes what the program
// factory already has and derives every field from it, the kernel constructor reads the same fields back in
// the same order. The sender has scalars only — every address it writes arrives per token in the slot's
// forwarding metadata.

#include "combine_fabric2d_kernel_interface.hpp"

namespace cmbf2d {

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
    uint32_t filled_slot;
    uint32_t freed_slot;
    uint32_t fwd_sem_noc_x;
    uint32_t fwd_sem_noc_y;
    uint32_t fwd_sem_addr;
    uint32_t final_sem_addr;

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
        filled_slot(plan.ring_filled_slot),
        freed_slot(plan.ring_freed_slot),
        fwd_sem_noc_x(static_cast<uint32_t>(downstream.worker_virtual.x)),
        fwd_sem_noc_y(static_cast<uint32_t>(downstream.worker_virtual.y)),
        fwd_sem_addr(plan.fwd_arrived_addr),
        final_sem_addr(plan.final_arrived_addr) {}

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
            filled_slot,
            freed_slot,
            fwd_sem_noc_x,
            fwd_sem_noc_y,
            fwd_sem_addr,
            final_sem_addr};
    }
#else
    constexpr SenderCtArgs() :
        num_l1_slots(get_compile_time_arg_val(0)),
        token_size_bytes(get_compile_time_arg_val(1)),
        forwarding_metadata_size(get_compile_time_arg_val(2)),
        peer_chip_id(get_compile_time_arg_val(3)),
        peer_mesh_id(get_compile_time_arg_val(4)),
        ring_addr(get_compile_time_arg_val(5)),
        pkt_hdr_ring_addr(get_compile_time_arg_val(6)),
        pkt_hdr_drain_addr(get_compile_time_arg_val(7)),
        drain_sink_addr(get_compile_time_arg_val(8)),
        batch(get_compile_time_arg_val(9)),
        filled_slot(get_compile_time_arg_val(10)),
        freed_slot(get_compile_time_arg_val(11)),
        fwd_sem_noc_x(get_compile_time_arg_val(12)),
        fwd_sem_noc_y(get_compile_time_arg_val(13)),
        fwd_sem_addr(get_compile_time_arg_val(14)),
        final_sem_addr(get_compile_time_arg_val(15)) {}

    // The ring counters live at hand-placed L1 addresses here, so this kernel owns their lifetime and
    // must hand the next launch a zeroed pair. Safe only at the very end: the reader's last act was
    // publishing the CMD_END slot the sender has just drained, so nothing still reads or bumps them.
    volatile tt_l1_ptr uint32_t* filled_ptr() const {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(filled_slot);
    }
    volatile tt_l1_ptr uint32_t* freed_ptr() const {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(freed_slot);
    }
    void reset_ring_counters() const {
        noc_semaphore_set(filled_ptr(), 0);
        noc_semaphore_set(freed_ptr(), 0);
    }
#endif

    constexpr uint32_t slot_stride() const { return token_size_bytes + forwarding_metadata_size; }
};

}  // namespace cmbf2d
