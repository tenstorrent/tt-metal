// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared argument layout and block-ownership helpers of the g6_to_memory_config movers.
//
// Compile-time args (identical layout for both kernels; the host writes them in this order):
//   0 walk                  WALK_SHARD (owned shard) | WALK_BANK (owned L1 bank stripe) | WALK_BOUNCE
//   1 page_size             bytes per tile page, equal on both sides
//   2 num_pages             P, pages of the flattened padded tile grid
//   3 movers_per_owner      RISCs sharing one owner's block (owner walks only)
//   4 mover_index           this RISC's share of the owner's block, < movers_per_owner
//   5 cb_bounce_pages       bounce staging CB (bounce walk only)
//   6 bounce_block_pages    pages per bounce block (bounce walk only)
//   7 bounce_write_run_pages  cap on pages per bounce write; 0 = the whole run contiguous in the target
//   8.. source TensorAccessorArgs, then target TensorAccessorArgs
// Common runtime args: 0 source base address, 1 target base address.
// Per-core runtime args: WALK_SHARD: [owned_shard_id]; WALK_BANK: none; WALK_BOUNCE: [start_page, core_pages].

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6 {

constexpr uint32_t WALK_SHARD = 0;
constexpr uint32_t WALK_BANK = 1;
constexpr uint32_t WALK_BOUNCE = 2;
constexpr uint32_t NO_SHARD = 0xFFFFFFFFu;
constexpr uint32_t NO_BANK = 0xFFFFFFFFu;

constexpr uint32_t walk = get_compile_time_arg_val(0);
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t num_pages = get_compile_time_arg_val(2);
constexpr uint32_t movers_per_owner = get_compile_time_arg_val(3);
constexpr uint32_t mover_index = get_compile_time_arg_val(4);
constexpr uint32_t cb_bounce_pages = get_compile_time_arg_val(5);
constexpr uint32_t bounce_block_pages = get_compile_time_arg_val(6);
constexpr uint32_t bounce_write_run_pages = get_compile_time_arg_val(7);
constexpr auto src_args = TensorAccessorArgs<8>();
constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

static_assert(walk <= WALK_BOUNCE, "unknown walk");
static_assert(movers_per_owner >= 1 && mover_index < movers_per_owner, "mover_index out of range");

struct Slice {
    uint32_t begin;
    uint32_t end;
};

// This mover's contiguous share of [0, n): contiguity keeps transfers coalescible within a shard.
FORCE_INLINE Slice mover_slice(uint32_t n) {
    const uint32_t per_mover = (n + movers_per_owner - 1) / movers_per_owner;
    const uint32_t begin = mover_index * per_mover < n ? mover_index * per_mover : n;
    const uint32_t end = begin + per_mover < n ? begin + per_mover : n;
    return {begin, end};
}

// The L1 bank this core holds, found by matching each bank's NoC node to our own. The allocator places
// one bank per compute core and shuffles the ids, and no host binding exposes the map; banks past P
// own no page, so the scan stops there.
template <typename Banked>
FORCE_INLINE uint32_t find_own_bank(const Banked& banked) {
    constexpr uint32_t scan_end = num_pages < NUM_L1_BANKS ? num_pages : NUM_L1_BANKS;
    for (uint32_t bank = 0; bank < scan_end; ++bank) {
        if (noc_address_backend::is_local(banked.get_noc_addr(bank), noc_index)) {
            return bank;
        }
    }
    return NO_BANK;
}

// Local L1 address of a page that this core holds; the NoC address must name this core.
FORCE_INLINE uint32_t local_slot(uint64_t noc_addr) {
    ASSERT(noc_address_backend::is_local(noc_addr, noc_index));
    return noc_address_backend::extract_local_address(noc_addr);
}

}  // namespace g6
