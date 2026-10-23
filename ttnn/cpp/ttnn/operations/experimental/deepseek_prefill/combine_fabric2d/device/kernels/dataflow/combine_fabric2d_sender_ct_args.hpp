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
    uint32_t final_sem_addr;
    uint32_t first_packet_bytes;
    // ONE core, TWO credits. The downstream chip's reverse-direction stream worker is both the core this
    // sender grants a LAUNCH credit to -- the cross-launch guard, which says the region may be written at
    // all -- and the core owed the per-PAGE channel credits this sender couriers for the opposite reader on
    // this chip. They coincide because reverse_stream(s) and s ^ 1 are the same stream, so one pair of NoC
    // coordinates serves both. The two credits are otherwise independent: different addresses, different
    // lifetimes, one per launch against one per freed page.
    uint32_t credit_noc_x;
    uint32_t credit_noc_y;
    uint32_t launch_credit_addr;
    uint32_t num_vcs;
    // Courier duty. This sender carries the credits of the OPPOSITE direction's reader on this chip,
    // because that reader owes them to the chip this sender already points at. Nothing to do with this
    // stream's own tokens -- the two are peers.
    uint32_t pkt_hdr_credit_addr;
    uint32_t credit_done_addr;  // that reader's end-of-stream flag, on this core

#ifndef KERNEL_BUILD
    // `downstream` is the worker serving this stream on the next chip: the sender bumps its
    // forwarded-token-count semaphore through the fabric packet header, and grants the launch credit to
    // `plan.launch_credit_core`, the worker on that chip whose sender writes back to this chip. That is why
    // every worker placement on the mesh is decided before any kernel is built.
    SenderCtArgs(
        const op::CombineFabric2dInputs& tensor_args,
        const op::StreamPlacement& self,
        const op::StreamPlacement& downstream,
        // The opposite direction's worker on the downstream chip: where this sender posts the credits it
        // couriers, since that is the reader whose writes into this chip they free.
        const op::StreamPlacement& downstream_opposite,
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
        final_sem_addr(plan.final_arrived_addr),
        first_packet_bytes(cmbf2d::first_packet_bytes(token_size_bytes)),
        // plan.launch_credit_core is this same core -- the program factory asserts it sends back to us --
        // so the one pair serves the launch credit and the couriered channel credits alike.
        credit_noc_x(static_cast<uint32_t>(downstream_opposite.worker_virtual.x)),
        credit_noc_y(static_cast<uint32_t>(downstream_opposite.worker_virtual.y)),
        launch_credit_addr(plan.launch_credit_addr),
        num_vcs(static_cast<uint32_t>(plan.vc_arrived_addr.size())),
        pkt_hdr_credit_addr(l1.pkt_hdr_credit),
        credit_done_addr(plan.credit_done_addr) {
        // Per channel: where its arrival counter sits on the downstream reader, where the credit we
        // courier lands over there, and where our own courier queue is on this core.
        for (uint32_t v = 0; v < num_vcs; v++) {
            blocks_.push_back(plan.vc_arrived_addr[v]);
            blocks_.push_back(plan.vc_freed_addr[v]);
            blocks_.push_back(plan.vc_outbox_addr[v]);
        }
    }

    std::vector<uint32_t> to_ct_word_arr() const {
        std::vector<uint32_t> word_arr{
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
            final_sem_addr,
            first_packet_bytes,
            credit_noc_x,
            credit_noc_y,
            launch_credit_addr,
            num_vcs,
            pkt_hdr_credit_addr,
            credit_done_addr};
        word_arr.insert(word_arr.end(), blocks_.begin(), blocks_.end());
        return word_arr;
    }

private:
    std::vector<uint32_t> blocks_;

public:
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
        final_sem_addr(get_compile_time_arg_val(14)),
        first_packet_bytes(get_compile_time_arg_val(15)),
        credit_noc_x(get_compile_time_arg_val(16)),
        credit_noc_y(get_compile_time_arg_val(17)),
        launch_credit_addr(get_compile_time_arg_val(18)),
        num_vcs(get_compile_time_arg_val(19)),
        pkt_hdr_credit_addr(get_compile_time_arg_val(20)),
        credit_done_addr(get_compile_time_arg_val(21)) {}

    // Words per channel in the block that follows the scalars.
    static constexpr uint32_t VC_WORDS = 3;
    static constexpr uint32_t vc_base = 22;
    uint32_t vc_arrived_addr(uint32_t v) const { return kernel_compile_time_args[vc_base + v * VC_WORDS + 0]; }
    uint32_t vc_freed_addr(uint32_t v) const { return kernel_compile_time_args[vc_base + v * VC_WORDS + 1]; }
    uint32_t vc_outbox_addr(uint32_t v) const { return kernel_compile_time_args[vc_base + v * VC_WORDS + 2]; }

    // What a ring-counter slot names depends on where the counters live. Overlapped, they are program
    // semaphores the framework re-initialises every launch; alone, they are hand-placed L1 this kernel
    // owns, so it must hand the next launch a zeroed pair. Zeroing is safe only at the very end: the
    // reader's last act was publishing the CMD_END slot the sender has just drained.
#ifdef CMBF2D_OVERLAPPED
    volatile tt_l1_ptr uint32_t* filled_ptr() const {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(filled_slot));
    }
    volatile tt_l1_ptr uint32_t* freed_ptr() const {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(freed_slot));
    }
    void reset_ring_counters() const {}
#else
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
#endif

    constexpr uint32_t slot_stride() const { return token_size_bytes + forwarding_metadata_size; }
    constexpr uint32_t packets_per_token() const { return first_packet_bytes < token_size_bytes ? 2 : 1; }
};

}  // namespace cmbf2d
