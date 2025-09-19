// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/tensor_accessor.h"
#include "dataflow_api.h"

// Kernel that:
// if (is_reader) {
//     iterate over local shards of sharded src tensor, and write them to the dst tensor.
// } else {
//     iterate over local shards of sharded dst tensor, and read them from the src tensor.
// }

void kernel_main() {
    auto args_src = TensorAccessorArgs<0, 0>();
    auto args_dst =
        TensorAccessorArgs<args_src.next_compile_time_args_offset(), args_src.next_common_runtime_args_offset()>();
    constexpr uint32_t base_idx_cta = args_dst.next_compile_time_args_offset();
    constexpr uint32_t base_idx_crta = args_dst.next_common_runtime_args_offset();

    constexpr uint32_t src_page_size = get_compile_time_arg_val(base_idx_cta);
    constexpr uint32_t dst_page_size = get_compile_time_arg_val(base_idx_cta + 1);
    constexpr uint32_t is_reader = get_compile_time_arg_val(base_idx_cta + 2);
    constexpr uint32_t logical_width = get_compile_time_arg_val(base_idx_cta + 3);
    constexpr uint32_t src_width = get_compile_time_arg_val(base_idx_cta + 4);
    constexpr uint32_t dst_width = get_compile_time_arg_val(base_idx_cta + 5);
    constexpr uint32_t transfer_size = get_compile_time_arg_val(base_idx_cta + 6);

    const uint32_t bank_base_address_src = get_common_arg_val<uint32_t>(base_idx_crta);
    const uint32_t bank_base_address_dst = get_common_arg_val<uint32_t>(base_idx_crta + 1);
    const uint32_t num_shards = get_common_arg_val<uint32_t>(base_idx_crta + 2);
    const uint32_t shard_id_stride = get_common_arg_val<uint32_t>(base_idx_crta + 3);

    const uint32_t first_shard_id = get_arg_val<uint32_t>(0);

    auto accessor_src = TensorAccessor(args_src, bank_base_address_src, src_page_size);
    auto accessor_dst = TensorAccessor(args_dst, bank_base_address_dst, dst_page_size);

    for (uint32_t shard_id = first_shard_id; shard_id < num_shards; shard_id += shard_id_stride) {
        if constexpr (is_reader) {
                auto shard_pages = accessor_src.shard_pages(shard_id);
                for (const auto& src_page : shard_pages) {
                    auto src_page_id = src_page.page_id();
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

                        uint64_t source_address = accessor_src.get_noc_addr(src_page_id, src_offset);
                        uint64_t destination_address = accessor_dst.get_noc_addr(dst_page_id, dst_offset);
                        noc_async_write(source_address, destination_address, transfer_size);
                    }
                    noc_async_writes_flushed();
                }
        } else {
            auto shard_pages = accessor_dst.shard_pages(shard_id);
            for (const auto& page : shard_pages) {
                noc_async_read_page(page.page_id(), accessor_src, page.noc_addr());
            }
        }
    }

    if constexpr (is_reader) {
        noc_async_write_barrier();
    } else {
        noc_async_read_barrier();
    }
}
