// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/tensor_accessor.h"
#include "dataflow_api.h"
#include "nd_reshard_common.hpp"

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

    constexpr uint32_t page_size = get_compile_time_arg_val(base_idx_cta);
    constexpr uint32_t is_reader = get_compile_time_arg_val(base_idx_cta + 1);

    const uint32_t bank_base_address_src = get_common_arg_val<uint32_t>(base_idx_crta);
    const uint32_t bank_base_address_dst = get_common_arg_val<uint32_t>(base_idx_crta + 1);
    const uint32_t num_shards = get_common_arg_val<uint32_t>(base_idx_crta + 2);
    const uint32_t shard_id_stride = get_common_arg_val<uint32_t>(base_idx_crta + 3);

    const uint32_t first_shard_id = get_arg_val<uint32_t>(0);

    auto accessor_src = TensorAccessor(args_src, bank_base_address_src, page_size);
    auto accessor_dst = TensorAccessor(args_dst, bank_base_address_dst, page_size);

    for (uint32_t shard_id = first_shard_id; shard_id < num_shards; shard_id += shard_id_stride) {
        if constexpr (is_reader) {
            iterate_pages_in_shard(accessor_src, shard_id, [&](uint32_t page_id) {
                // TODO: local noc_addr calculation can be optimized
                ASSERT(accessor_src.is_local_page(page_id));
                noc_async_write_page(page_id, accessor_dst, accessor_src.get_noc_addr(page_id));
                noc_async_writes_flushed();
            });
        } else {
            iterate_pages_in_shard(accessor_dst, shard_id, [&](uint32_t page_id) {
                // TODO: local noc_addr calculation can be optimized
                ASSERT(accessor_dst.is_local_page(page_id));
                noc_async_read_page(page_id, accessor_src, accessor_dst.get_noc_addr(page_id));
            });
        }
    }

    if constexpr (is_reader) {
        noc_async_write_barrier();
    } else {
        noc_async_read_barrier();
    }
}
