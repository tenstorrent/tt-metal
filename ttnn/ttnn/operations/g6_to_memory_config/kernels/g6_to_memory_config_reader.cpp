// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather kernel (RISCV_1, NoC0): NoC reads from the source into local L1.
//   R1 pull:        local = this core's target shard or target L1 bank(s); reads land in the output itself.
//   R3 bounce read: local = the staging CB; one block per slot, read barrier, publish.
// Raw NoC + TensorAccessor: no kernel_lib helper moves whole pages between placements (op_design.md, API Mapping).
// Argument layout: g6_to_memory_config_walk.hpp.

#include "api/dataflow/dataflow_api.h"
#include "g6_to_memory_config_walk.hpp"

namespace {

template <typename Walk, typename Accessor>
void bounce_gather(
    Walk& walk,
    const Accessor& src,
    uint32_t src_stride,
    uint32_t page_size,
    uint32_t max_segment_pages,
    uint32_t block_pages,
    uint32_t count,
    uint32_t cb_staging_pages) {
    for (uint32_t done = 0; done < count; done += block_pages) {
        const uint32_t n = g6::min_u32(block_pages, count - done);
        {
            DeviceZoneScopedN("G6-SLOT-WAIT");
            cb_reserve_back(cb_staging_pages, block_pages);
        }
        {
            DeviceZoneScopedN("G6-ISSUE");
            g6::move_positions<true, g6::STAGING, false>(
                walk, 1, src, src_stride, get_write_ptr(cb_staging_pages), page_size, max_segment_pages, n);
        }
        {
            DeviceZoneScopedN("G6-BLOCK-BARRIER");
            noc_async_read_barrier();
        }
        // Nominal count even for a ragged last block keeps the ring aligned to whole slots.
        cb_push_back(cb_staging_pages, block_pages);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t local_kind = get_compile_time_arg_val(0);
    constexpr bool walk_sharded = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t tensor_pages = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_rows = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_cols = get_compile_time_arg_val(4);
    constexpr uint32_t walk_shard_rows = get_compile_time_arg_val(5);
    constexpr uint32_t walk_shard_cols = get_compile_time_arg_val(6);
    constexpr uint32_t walk_num_shard_cols = get_compile_time_arg_val(7);
    constexpr uint32_t max_segment_pages = get_compile_time_arg_val(8);
    constexpr uint32_t block_pages = get_compile_time_arg_val(9);
    constexpr uint32_t cb_staging_pages = get_compile_time_arg_val(10);
    constexpr auto src_args = TensorAccessorArgs<11>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const auto src = TensorAccessor(src_args, get_common_arg_val<uint32_t>(0));
    const auto dst = TensorAccessor(dst_args, get_common_arg_val<uint32_t>(1));
    const uint32_t page_size = src.get_aligned_page_size();
    const uint32_t src_stride = src.contiguous_page_stride();

    const uint32_t walk_start = get_arg_val<uint32_t>(0);
    const uint32_t walk_offset = get_arg_val<uint32_t>(1);
    const uint32_t num_positions = get_arg_val<uint32_t>(2);

    if constexpr (local_kind == g6::OWNED_SHARD) {
        g6::ShardWalk walk{tensor_rows, tensor_cols, walk_shard_rows, walk_shard_cols, walk_num_shard_cols};
        uint32_t shard_base = 0;
        {
            DeviceZoneScopedN("G6-RESOLVE");
            if (num_positions > 0) {
                walk.enter(walk_start, walk_offset);
                const uint64_t shard_noc_addr = dst.get_noc_addr(walk.row0 * tensor_cols + walk.col0);
                ASSERT(noc_address_backend::is_local(shard_noc_addr, noc_index));
                shard_base = noc_address_backend::extract_local_address(shard_noc_addr);
            }
        }
        {
            DeviceZoneScopedN("G6-ISSUE");
            g6::move_owned<true, g6::OWNED_SHARD>(
                walk, 1, src, src_stride, shard_base, page_size, max_segment_pages, block_pages, num_positions);
        }
        {
            DeviceZoneScopedN("G6-DRAIN");
            noc_async_read_barrier();
        }
    } else if constexpr (local_kind == g6::OWNED_BANKS) {
        uint32_t banks[g6::MAX_OWNED_BANKS];
        uint32_t num_owned;
        {
            DeviceZoneScopedN("G6-RESOLVE");
            num_owned = g6::find_owned_l1_banks(tensor_pages, banks);
        }
        {
            DeviceZoneScopedN("G6-ISSUE");
            for (uint32_t i = 0; i < num_owned; ++i) {
                g6::LinearWalk walk{banks[i], NUM_L1_BANKS};
                const uint32_t bank_base = noc_address_backend::extract_local_address(dst.get_noc_addr(banks[i]));
                g6::move_owned<true, g6::OWNED_BANKS>(
                    walk,
                    NUM_L1_BANKS,
                    src,
                    src_stride,
                    bank_base,
                    page_size,
                    max_segment_pages,
                    block_pages,
                    g6::bank_page_count(banks[i], tensor_pages));
            }
        }
        {
            DeviceZoneScopedN("G6-DRAIN");
            noc_async_read_barrier();
        }
    } else {
        if constexpr (walk_sharded) {
            g6::ShardWalk walk{tensor_rows, tensor_cols, walk_shard_rows, walk_shard_cols, walk_num_shard_cols};
            if (num_positions > 0) {
                walk.enter(walk_start, walk_offset);
            }
            bounce_gather(
                walk, src, src_stride, page_size, max_segment_pages, block_pages, num_positions, cb_staging_pages);
        } else {
            g6::LinearWalk walk{walk_start, 1};
            bounce_gather(
                walk, src, src_stride, page_size, max_segment_pages, block_pages, num_positions, cb_staging_pages);
        }
    }
}
