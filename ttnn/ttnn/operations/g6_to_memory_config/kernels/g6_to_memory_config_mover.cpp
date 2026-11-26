// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Owner mover (shard_pull, shard_push, bank_pull, bank_push): the core that holds a page in its own
// L1 moves it, so every page crosses the NoC once and no CB exists. A pull lands the page at its
// final address in the output's L1; a push reads it in place from the input's L1. One kernel per
// data-movement RISC (mover 0: NCRISC/NoC0, mover 1: BRISC/NoC1), each taking a contiguous slice of
// the core's page sequence and ending with one barrier.
//
// Raw NoC calls, no kernel_lib helper: see g6_to_memory_config_move.hpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/ttnn/operations/g6_to_memory_config/kernels/g6_to_memory_config_move.hpp"

namespace {

constexpr uint32_t mover_index = get_compile_time_arg_val(0);
constexpr uint32_t movers_per_core = get_compile_time_arg_val(1);
constexpr bool is_pull = get_compile_time_arg_val(2) == 1;
constexpr bool is_bank = get_compile_time_arg_val(3) == 1;
constexpr uint32_t owner_shard_tile_cols = get_compile_time_arg_val(4);
constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(5);
constexpr uint32_t tensor_pages = get_compile_time_arg_val(6);
constexpr uint32_t accessor_args_offset = 7;

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// shard_*: the owner holds exactly one shard. Its page (r, c) sits at
// owner_base + ((r - r0) * owner_shard_tile_cols + (c - c0)) * page_size; the nominal shard width is
// the row stride also for an uneven last shard. The walk is row-major over the mover's slice
// [seq_begin, seq_end) of the core_rows x core_cols rectangle; no run crosses a shard row.
template <typename Remote, typename Owner>
FORCE_INLINE void move_owned_shard(const Remote& remote, const Owner& owner, uint32_t owner_base, uint32_t page_size) {
    uint32_t r0, c0, core_cols, seq_begin, seq_end;
    {
        DeviceZoneScopedN("G6-LOCATE");
        r0 = get_arg_val<uint32_t>(0);
        c0 = get_arg_val<uint32_t>(1);
        core_cols = get_arg_val<uint32_t>(2);
        seq_begin = get_arg_val<uint32_t>(3);
        seq_end = get_arg_val<uint32_t>(4);
    }
    DeviceZoneScopedN("G6-ISSUE");
    uint32_t seq = seq_begin;
    while (seq < seq_end) {
        const uint32_t local_row = seq / core_cols;
        const uint32_t local_col = seq - local_row * core_cols;
        const uint32_t row_pages = min_u32(core_cols - local_col, seq_end - seq);
        const uint32_t page = (r0 + local_row) * tensor_tile_cols + c0 + local_col;
        const uint32_t local_addr = owner_base + (local_row * owner_shard_tile_cols + local_col) * page_size;
        // The host's shard -> core rule must agree with the owner buffer's distribution spec.
        ASSERT(owner.get_noc_addr(page) == get_noc_addr(local_addr));
        g6_to_memory_config::move_page_range<is_pull>(remote, page, row_pages, local_addr, page_size);
        seq += row_pages;
    }
}

// bank_*: the owner side is L1-interleaved with one bank per worker core. The core finds its own bank
// b and owns pages b + k * NUM_L1_BANKS, stored back to back in its bank. Pages of one bank are
// NUM_L1_BANKS apart in page id, so every transfer is one page.
template <typename Remote, typename Owner>
FORCE_INLINE void move_owned_bank(const Remote& remote, const Owner& owner, uint32_t page_size) {
    uint32_t page = 0;
    uint32_t local_addr = 0;
    uint32_t mover_pages = 0;
    {
        DeviceZoneScopedN("G6-LOCATE");
        const uint64_t self = noc_address_backend::self_address(0, noc_index);
        uint32_t bank = NUM_L1_BANKS;
        for (uint32_t b = 0; b < NUM_L1_BANKS; ++b) {
            if (noc_address_backend::bank_address<false>(b, 0, noc_index) == self) {
                bank = b;
                break;
            }
        }
        ASSERT(bank < NUM_L1_BANKS);
        const uint32_t bank_pages =
            (bank < NUM_L1_BANKS && bank < tensor_pages) ? (tensor_pages - bank + NUM_L1_BANKS - 1) / NUM_L1_BANKS : 0;
        const uint32_t half = (bank_pages + movers_per_core - 1) / movers_per_core;
        const uint32_t k_begin = min_u32(bank_pages, mover_index * half);
        const uint32_t k_end = min_u32(bank_pages, k_begin + half);
        mover_pages = k_end - k_begin;
        if (mover_pages > 0) {
            page = bank + k_begin * NUM_L1_BANKS;
            local_addr = noc_address_backend::extract_local_address(owner.get_noc_addr(page));
        }
    }
    DeviceZoneScopedN("G6-ISSUE");
    for (uint32_t k = 0; k < mover_pages; ++k) {
        ASSERT(owner.get_noc_addr(page) == get_noc_addr(local_addr));
        g6_to_memory_config::issue_transfer<is_pull>(remote, page, local_addr, page_size);
        page += NUM_L1_BANKS;
        local_addr += page_size;
    }
}

template <typename Remote, typename Owner>
FORCE_INLINE void move_owned_block(const Remote& remote, const Owner& owner, uint32_t owner_base, uint32_t page_size) {
    if constexpr (is_bank) {
        move_owned_bank(remote, owner, page_size);
    } else {
        move_owned_shard(remote, owner, owner_base, page_size);
    }
}

}  // namespace

void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<accessor_args_offset>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const uint32_t src_base = get_common_arg_val<uint32_t>(0);
    const uint32_t dst_base = get_common_arg_val<uint32_t>(1);
    const auto src = TensorAccessor(src_args, src_base);
    const auto dst = TensorAccessor(dst_args, dst_base);
    const uint32_t page_size = src.get_aligned_page_size();
    ASSERT(page_size == dst.get_aligned_page_size());

    if constexpr (is_pull) {
        move_owned_block(src, dst, dst_base, page_size);
    } else {
        move_owned_block(dst, src, src_base, page_size);
    }

    // BB-03: nothing may be in flight when the next program starts; writes need their acks.
    DeviceZoneScopedN("G6-DRAIN");
    if constexpr (is_pull) {
        noc_async_read_barrier();
    } else {
        noc_async_write_barrier();
    }
}
