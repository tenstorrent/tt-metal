// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Owner mover of g6_to_memory_config (regimes shard_pull, shard_push, bank_pull, bank_push).
//
// The core that holds a page in its own L1 moves it: a pull reads the remote source page straight to
// its final address in this core's output shard / L1 bank; a push writes this core's input shard /
// L1 bank page straight to its remote target. One NoC crossing per page, no staging buffer, no CB.
// The same source runs as mover 0 (NCRISC, NoC0) and mover 1 (BRISC, NoC1); each takes a contiguous
// slice of the core's page sequence, issues every transfer, then drains with one barrier.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_memory_config_move.hpp"

namespace {

constexpr bool kIsPull = get_compile_time_arg_val(0) == 1;
constexpr bool kOwnerIsBank = get_compile_time_arg_val(1) == 1;
constexpr uint32_t kMoverIndex = get_compile_time_arg_val(2);
constexpr uint32_t kMoversPerCore = get_compile_time_arg_val(3);
constexpr uint32_t kPageSize = get_compile_time_arg_val(4);
constexpr uint32_t kTensorPages = get_compile_time_arg_val(5);
constexpr uint32_t kTensorTileCols = get_compile_time_arg_val(6);
constexpr uint32_t kShardTileCols = get_compile_time_arg_val(7);

// Shard owner: walk sequence indices [seq_begin, seq_end) of the core_rows x core_cols rectangle at
// (r0, c0) in row-major order. Local rows are kShardTileCols pages apart (nominal width, also for the
// uneven last shard); a run never crosses a row of the owner's shard.
template <typename Owner, typename Remote>
FORCE_INLINE void move_shard_slice(
    [[maybe_unused]] const Owner& owner,
    const Remote& remote,
    uint32_t local_base,
    uint32_t r0,
    uint32_t c0,
    uint32_t core_cols,
    uint32_t seq_begin,
    uint32_t seq_end) {
    uint32_t q = seq_begin;
    uint32_t row = q / core_cols;
    uint32_t col = q - row * core_cols;
    uint32_t local_row = local_base + row * kShardTileCols * kPageSize;
    while (q < seq_end) {
        uint32_t span = core_cols - col;
        if (span > seq_end - q) {
            span = seq_end - q;
        }
        const uint32_t page = (r0 + row) * kTensorTileCols + c0 + col;
        const uint32_t local = local_row + col * kPageSize;
        ASSERT(local == g6_to_memory_config::local_part(owner.get_noc_addr(page)));
        ASSERT(g6_to_memory_config::is_this_core(owner.get_noc_addr(page)));
        g6_to_memory_config::move_span<kIsPull>(remote, page, span, local, kPageSize);
        q += span;
        ++row;
        col = 0;
        local_row += kShardTileCols * kPageSize;
    }
}

// Bank owner: pages bank + k * NUM_L1_BANKS for k in [k_begin, k_end); an interleaved remote end is
// never contiguous across consecutive page ids, so each transfer is one page.
template <typename Owner, typename Remote>
FORCE_INLINE void move_bank_slice(
    [[maybe_unused]] const Owner& owner, const Remote& remote, uint32_t bank, uint32_t k_begin, uint32_t k_end) {
    uint32_t page = bank + k_begin * NUM_L1_BANKS;
    uint32_t local = g6_to_memory_config::local_part(owner.get_noc_addr(bank)) + k_begin * kPageSize;
    for (uint32_t k = k_begin; k < k_end; ++k) {
        ASSERT(local == g6_to_memory_config::local_part(owner.get_noc_addr(page)));
        const uint64_t remote_addr = remote.get_noc_addr(page);
        if constexpr (kIsPull) {
            noc_async_read(remote_addr, local, kPageSize);
        } else {
            noc_async_write(local, remote_addr, kPageSize);
        }
        page += NUM_L1_BANKS;
        local += kPageSize;
    }
}

template <typename Owner, typename Remote>
FORCE_INLINE void run_mover(const Owner& owner, const Remote& remote, uint32_t owner_base) {
    if constexpr (kOwnerIsBank) {
        uint32_t bank;
        uint32_t k_begin;
        uint32_t k_end;
        {
            DeviceZoneScopedN("G6-LOCATE");
            bank = g6_to_memory_config::find_own_l1_bank(owner);
            ASSERT(bank < NUM_L1_BANKS);
            const uint32_t bank_pages =
                bank < kTensorPages ? (kTensorPages - bank + NUM_L1_BANKS - 1) / NUM_L1_BANKS : 0;
            const uint32_t half = (bank_pages + kMoversPerCore - 1) / kMoversPerCore;
            k_begin = kMoverIndex * half < bank_pages ? kMoverIndex * half : bank_pages;
            k_end = k_begin + half < bank_pages ? k_begin + half : bank_pages;
        }
        if (k_begin == k_end) {
            return;
        }
        {
            DeviceZoneScopedN("G6-ISSUE");
            move_bank_slice(owner, remote, bank, k_begin, k_end);
        }
    } else {
        uint32_t r0;
        uint32_t c0;
        uint32_t core_cols;
        uint32_t seq_begin;
        uint32_t seq_end;
        {
            DeviceZoneScopedN("G6-LOCATE");
            r0 = get_arg_val<uint32_t>(0);
            c0 = get_arg_val<uint32_t>(1);
            core_cols = get_arg_val<uint32_t>(2);
            seq_begin = get_arg_val<uint32_t>(3);
            seq_end = get_arg_val<uint32_t>(4);
        }
        if (seq_begin == seq_end) {
            return;
        }
        {
            DeviceZoneScopedN("G6-ISSUE");
            move_shard_slice(owner, remote, owner_base, r0, c0, core_cols, seq_begin, seq_end);
        }
    }
    {
        DeviceZoneScopedN("G6-DRAIN");
        // A full barrier, not writes_flushed: the next program must not start with acks outstanding.
        if constexpr (kIsPull) {
            noc_async_read_barrier();
        } else {
            noc_async_write_barrier();
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<8>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const uint32_t src_base = get_common_arg_val<uint32_t>(0);
    const uint32_t dst_base = get_common_arg_val<uint32_t>(1);
    const auto src = TensorAccessor(src_args, src_base, kPageSize);
    const auto dst = TensorAccessor(dst_args, dst_base, kPageSize);

    if constexpr (kIsPull) {
        run_mover(dst, src, dst_base);
    } else {
        run_mover(src, dst, src_base);
    }
}
