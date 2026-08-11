// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Kernel that:
// if (is_reader) {
//     iterate over local shards of sharded src tensor, and write them to the dst tensor.
// } else {
//     iterate over local shards of sharded dst tensor, and read them from the src tensor.
// }

void kernel_main() {
    constexpr uint32_t src_page_size = get_arg(args::src_page_size);
    constexpr uint32_t dst_page_size = get_arg(args::dst_page_size);
    constexpr uint32_t is_reader = get_arg(args::is_reader);
    constexpr uint32_t logical_width = get_arg(args::logical_width);
    constexpr uint32_t src_width = get_arg(args::src_width);
    constexpr uint32_t dst_width = get_arg(args::dst_width);
    constexpr uint32_t transfer_size = get_arg(args::transfer_size);

    const uint32_t num_shards = get_arg(args::num_shards);
    const uint32_t shard_id_stride = get_arg(args::shard_id_stride);

    const uint32_t first_shard_id = get_arg(args::first_shard_id);

    auto accessor_src = TensorAccessor(tensor::src);
    auto accessor_dst = TensorAccessor(tensor::dst);

    Noc noc;

    for (uint32_t shard_id = first_shard_id; shard_id < num_shards; shard_id += shard_id_stride) {
        if constexpr (is_reader) {
            auto shard_pages = accessor_src.shard_pages(shard_id);
            for (const auto& src_page : shard_pages) {
                auto src_page_id = src_page.page_id();
                // Local shard page: low 32 bits of the noc_addr are the L1 address on this core.
                const uint32_t src_l1_addr_base = static_cast<uint32_t>(src_page.noc_addr());
                const uint32_t transfers_per_page = src_page_size / transfer_size;
                for (uint32_t i = 0; i < transfers_per_page; i++) {
                    const uint32_t src_offset = i * transfer_size;
                    const uint32_t global_offset = src_page_id * src_page_size + src_offset;

                    const uint32_t row_idx = global_offset / src_width;
                    const uint32_t col_idx = global_offset % src_width;

                    // Skip if we're in padding area at end of sharded row
                    if (col_idx >= logical_width) {
                        continue;
                    }
                    // Calculate destination using logical width
                    uint32_t dst_global_offset = row_idx * dst_width + col_idx;

                    const uint32_t dst_row_idx = dst_global_offset / dst_width;
                    const uint32_t dst_col_idx = dst_global_offset % dst_width;
                    if (dst_col_idx >= logical_width) {
                        // update dst_global_offset to skip padding
                        dst_global_offset = (row_idx + 1) * dst_width;
                    }

                    const uint32_t dst_page_id = dst_global_offset / dst_page_size;
                    const uint32_t dst_offset = dst_global_offset % dst_page_size;

                    CoreLocalMem<uint32_t> src_mem(src_l1_addr_base + src_offset);
                    noc.async_write(
                        src_mem,
                        accessor_dst,
                        transfer_size,
                        {.offset_bytes = 0},
                        {.page_id = dst_page_id, .offset_bytes = dst_offset});
                }
                noc.async_writes_flushed();
            }
        } else {
            auto shard_pages = accessor_dst.shard_pages(shard_id);
            for (const auto& page : shard_pages) {
                // Local shard page: low 32 bits of the noc_addr are the L1 address on this core.
                const uint32_t dst_l1_addr = static_cast<uint32_t>(page.noc_addr());
                CoreLocalMem<uint32_t> dst_mem(dst_l1_addr);
                noc.async_read(
                    accessor_src,
                    dst_mem,
                    src_page_size,
                    {.page_id = page.page_id(), .offset_bytes = 0},
                    {.offset_bytes = 0});
            }
        }
    }

    if constexpr (is_reader) {
        noc.async_write_barrier();
    } else {
        noc.async_read_barrier();
    }
}
