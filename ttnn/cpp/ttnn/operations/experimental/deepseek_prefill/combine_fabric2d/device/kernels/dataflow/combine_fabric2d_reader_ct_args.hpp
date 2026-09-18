// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The reader kernel's compile-time arguments. One field list: the host constructor takes what the program
// factory already has and derives every field from it, the kernel constructor reads the same fields back in
// the same order.
//
// Variable-length blocks follow the scalars, in this order:
//   [SCALAR_CT_ARGS ..)          schedule, `schedule_len` words
//   [schedule end ..)            assignments, ASSIGNMENT_WORDS each
//   [assignments end ..)         TensorAccessorArgs, chained on by the program factory

#include "combine_fabric2d_kernel_interface.hpp"

#ifdef KERNEL_BUILD
#include "api/tensor/tensor_accessor_args.h"
#endif

namespace cmbf2d {

// Scalars packed before the variable-length blocks, i.e. the index the schedule starts at.
constexpr uint32_t READER_SCALAR_CT_ARGS = 31;

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
        // Our quarter of the forwarding buffer. (plane, direction) identifies the upstream sender uniquely
        // from the downstream chip's point of view, so the reader WRITES quarter q of the neighbour's buffer
        // and READS quarter q of its own — the same q, because every chip runs the same code. Doubles as
        // this stream's share of the same-chip run, which it copies after the fabric work.
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
#else
    constexpr ReaderCtArgs() :
        num_l1_slots(get_compile_time_arg_val(0)),
        token_size_bytes(get_compile_time_arg_val(1)),
        forwarding_metadata_size(get_compile_time_arg_val(2)),
        batch(get_compile_time_arg_val(3)),
        ring_addr(get_compile_time_arg_val(4)),
        filled_addr(get_compile_time_arg_val(5)),
        freed_addr(get_compile_time_arg_val(6)),
        dram_in_base_addr(get_compile_time_arg_val(7)),
        dram_out_base_addr(get_compile_time_arg_val(8)),
        dram_fwd_base_addr(get_compile_time_arg_val(9)),
        fwd_chunks_per_quarter(get_compile_time_arg_val(10)),
        fwd_pages_per_chunk(get_compile_time_arg_val(11)),
        my_quarter(get_compile_time_arg_val(12)),
        num_incoming_chunks(get_compile_time_arg_val(13)),
        fwd_sem_addr(get_compile_time_arg_val(14)),
        nbr_chip_id(get_compile_time_arg_val(15)),
        num_assignments(get_compile_time_arg_val(16)),
        schedule_len(get_compile_time_arg_val(17)),
        dram_meta_base_addr(get_compile_time_arg_val(18)),
        dram_counts_base_addr(get_compile_time_arg_val(19)),
        dram_region_base_addr(get_compile_time_arg_val(20)),
        dram_expert_offsets_base_addr(get_compile_time_arg_val(21)),
        num_routed_experts(get_compile_time_arg_val(22)),
        experts_per_chip(get_compile_time_arg_val(23)),
        my_expert_base(get_compile_time_arg_val(24)),
        num_experts_per_tok(get_compile_time_arg_val(25)),
        dispatch_group_size(get_compile_time_arg_val(26)),
        local_split_count(get_compile_time_arg_val(27)),
        my_row(get_compile_time_arg_val(28)),
        control_addr(get_compile_time_arg_val(29)),
        meta_prefetch_cap(get_compile_time_arg_val(30)) {}

    static constexpr uint32_t schedule_base = READER_SCALAR_CT_ARGS;
    static constexpr uint32_t assignment_base = schedule_base + get_compile_time_arg_val(17);  // schedule_len
    static constexpr uint32_t accessor_base =
        assignment_base + ASSIGNMENT_WORDS * get_compile_time_arg_val(16);  // num_assignments

    // One accessor per DRAM buffer the program factory chained on, in that order.
    static constexpr auto dram_in_args = TensorAccessorArgs<accessor_base>();
    static constexpr auto dram_out_args = TensorAccessorArgs<dram_in_args.next_compile_time_args_offset()>();
    static constexpr auto dram_fwd_args = TensorAccessorArgs<dram_out_args.next_compile_time_args_offset()>();
    static constexpr auto dram_meta_args = TensorAccessorArgs<dram_fwd_args.next_compile_time_args_offset()>();
    static constexpr auto dram_counts_args = TensorAccessorArgs<dram_meta_args.next_compile_time_args_offset()>();
    static constexpr auto dram_region_args = TensorAccessorArgs<dram_counts_args.next_compile_time_args_offset()>();
    static constexpr auto dram_expert_offsets_args =
        TensorAccessorArgs<dram_region_args.next_compile_time_args_offset()>();
#endif

    constexpr uint32_t slot_stride() const { return token_size_bytes + forwarding_metadata_size; }

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

}  // namespace cmbf2d
