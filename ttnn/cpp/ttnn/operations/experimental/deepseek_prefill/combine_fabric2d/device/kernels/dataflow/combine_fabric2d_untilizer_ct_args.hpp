// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The untilizer kernel's compile-time arguments. One field list: the host constructor takes what the program
// factory already has and derives every field from it, the kernel constructor reads the same fields back in
// the same order.
//
// Variable-length blocks follow the scalars, in this order:
//   [SCALAR_CT_ARGS ..)          own destinations, one dispatch-group index each, in the group's walk order
//   [destinations end ..)        the readers this core feeds, UNT_PEER_WORDS each
//   [consumers end ..)           TensorAccessorArgs, chained on by the program factory

#include "combine_fabric2d_kernel_interface.hpp"

#ifdef KERNEL_BUILD
#include "api/tensor/tensor_accessor_args.h"
#endif

namespace cmbf2d {

// Scalars packed before the variable-length blocks, i.e. the index the destinations start at. Asserted
// against the field list below, so it cannot drift out of step with it.
constexpr uint32_t UNTILIZER_SCALAR_CT_ARGS = 21;

struct UntilizerCtArgs {
    uint32_t token_size_bytes;
    uint32_t dram_in_base_addr;
    uint32_t dram_counts_base_addr;
    uint32_t dram_region_base_addr;
    uint32_t dram_expert_offsets_base_addr;
    uint32_t num_routed_experts;
    uint32_t experts_per_chip;
    uint32_t my_expert_base;
    uint32_t dispatch_group_size;
    uint32_t my_dg_index;
    uint32_t num_destinations;
    uint32_t walks_down;
    // Position in the group and its size. A group's batches are dealt out round robin, so this core takes
    // those where batch % num_peers == my_index and needs nothing else to know which are its own.
    uint32_t my_index;
    uint32_t num_peers;
    uint32_t num_consumers;
    uint32_t ring_batches;
    uint32_t control_addr;
    uint32_t produced_addr;
    // Tile geometry. These cores exist only where there is something to untilize.
    uint32_t tiles_per_row;
    uint32_t tile_bytes;
    uint32_t block_tiles;

#ifndef KERNEL_BUILD
    UntilizerCtArgs(
        const op::CombineFabric2dParams& args,
        const op::CombineFabric2dInputs& tensor_args,
        const ttnn::MeshCoordinate& coord,
        const std::vector<op::Assignment>& work,
        const op::UntilizerPlan& plan,
        const op::DramBuffers& dram) :
        token_size_bytes(op::token_size_bytes(tensor_args)),
        dram_in_base_addr(static_cast<uint32_t>(dram.in->address())),
        dram_counts_base_addr(static_cast<uint32_t>(dram.counts->address())),
        dram_region_base_addr(static_cast<uint32_t>(dram.region->address())),
        dram_expert_offsets_base_addr(static_cast<uint32_t>(dram.expert_offsets->address())),
        num_routed_experts(op::num_routed_experts(tensor_args)),
        experts_per_chip(args.experts_per_chip),
        my_expert_base(plan.my_expert_base),
        dispatch_group_size(op::ring_extent(args)),
        my_dg_index(op::my_dg_index(args, coord)),
        num_destinations(0),  // set from the destination block below, so the two cannot disagree
        walks_down(plan.walks_down),
        my_index(plan.my_index),
        num_peers(plan.num_peers),
        num_consumers(static_cast<uint32_t>(plan.consumers.size())),
        ring_batches(UNT_RING_BATCHES),
        control_addr(plan.control_addr),
        produced_addr(plan.produced_addr),
        tiles_per_row(op::tiles_per_token_row(tensor_args)),
        tile_bytes(op::tile_size_bytes(tensor_args)),
        block_tiles(op::untilize_block_tiles(tensor_args)) {
        // The own destinations in emission order, which is the order the group walks their runs. Taken from
        // the same work list a reader's schedule is built from, so neither side can reorder alone.
        for (const auto& w : work) {
            if (!w.is_relay) {
                blocks_.push_back(w.dst_dg_index);
                num_destinations++;
            }
        }
        for (const auto& c : plan.consumers) {
            blocks_.push_back(c.noc.x);
            blocks_.push_back(c.noc.y);
            blocks_.push_back(c.counter_addr);
        }
    }

    std::vector<uint32_t> to_ct_word_arr() const {
        std::vector<uint32_t> word_arr{
            token_size_bytes,
            dram_in_base_addr,
            dram_counts_base_addr,
            dram_region_base_addr,
            dram_expert_offsets_base_addr,
            num_routed_experts,
            experts_per_chip,
            my_expert_base,
            dispatch_group_size,
            my_dg_index,
            num_destinations,
            walks_down,
            my_index,
            num_peers,
            num_consumers,
            ring_batches,
            control_addr,
            produced_addr,
            tiles_per_row,
            tile_bytes,
            block_tiles};
        word_arr.insert(word_arr.end(), blocks_.begin(), blocks_.end());
        return word_arr;
    }
#else
    constexpr UntilizerCtArgs() :
        token_size_bytes(get_compile_time_arg_val(0)),
        dram_in_base_addr(get_compile_time_arg_val(1)),
        dram_counts_base_addr(get_compile_time_arg_val(2)),
        dram_region_base_addr(get_compile_time_arg_val(3)),
        dram_expert_offsets_base_addr(get_compile_time_arg_val(4)),
        num_routed_experts(get_compile_time_arg_val(5)),
        experts_per_chip(get_compile_time_arg_val(6)),
        my_expert_base(get_compile_time_arg_val(7)),
        dispatch_group_size(get_compile_time_arg_val(8)),
        my_dg_index(get_compile_time_arg_val(9)),
        num_destinations(get_compile_time_arg_val(10)),
        walks_down(get_compile_time_arg_val(11)),
        my_index(get_compile_time_arg_val(12)),
        num_peers(get_compile_time_arg_val(13)),
        num_consumers(get_compile_time_arg_val(14)),
        ring_batches(get_compile_time_arg_val(15)),
        control_addr(get_compile_time_arg_val(16)),
        produced_addr(get_compile_time_arg_val(17)),
        tiles_per_row(get_compile_time_arg_val(18)),
        tile_bytes(get_compile_time_arg_val(19)),
        block_tiles(get_compile_time_arg_val(20)) {}

    static constexpr uint32_t destination_base = UNTILIZER_SCALAR_CT_ARGS;
    static constexpr uint32_t consumer_base = destination_base + get_compile_time_arg_val(10);  // num_destinations
    static constexpr uint32_t accessor_base =
        consumer_base + UNT_PEER_WORDS * get_compile_time_arg_val(14);  // num_consumers

    // One accessor per DRAM buffer the program factory chained on, in that order.
    static constexpr auto dram_in_args = TensorAccessorArgs<accessor_base>();
    static constexpr auto dram_counts_args = TensorAccessorArgs<dram_in_args.next_compile_time_args_offset()>();
    static constexpr auto dram_region_args = TensorAccessorArgs<dram_counts_args.next_compile_time_args_offset()>();
    static constexpr auto dram_expert_offsets_args =
        TensorAccessorArgs<dram_region_args.next_compile_time_args_offset()>();
#endif

#ifndef KERNEL_BUILD
private:
    std::vector<uint32_t> blocks_;  // destinations then consumers, appended after the scalars
#endif
};

#ifdef KERNEL_BUILD
// The scalars are read back by index, so the base the variable-length blocks start at IS the field count.
static_assert(
    sizeof(UntilizerCtArgs) == UNTILIZER_SCALAR_CT_ARGS * sizeof(uint32_t),
    "UNTILIZER_SCALAR_CT_ARGS no longer matches the field list; the blocks after the scalars would be misread");
#endif

}  // namespace cmbf2d
