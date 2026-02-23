// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "elt_nd_sharded_add_reader_args.hpp"
#include "ttnn/kernel/kernel_utils.hpp"

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel::eltwise::add_nd_sharded_args;
    auto args = make_runtime_struct_from_args<ElemwiseReaderKernelArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<CompileTimeReaderKernelArgs>();

    constexpr auto element_size = 16;  // TODO: DONT HARDCODE THIS

    constexpr auto a_tensor_args = TensorAccessorArgs<amount_of_fields<CompileTimeReaderKernelArgs>()>();
    constexpr auto b_tensor_args = TensorAccessorArgs<a_tensor_args.next_compile_time_args_offset()>();

    const auto a_tensor = TensorAccessor(a_tensor_args, args.a_tensor_base_addr, get_tile_size(c_args.a_tensor_cb));
    const auto b_tensor = TensorAccessor(b_tensor_args, args.b_tensor_base_addr, get_tile_size(c_args.b_tensor_cb));

    const auto page_size = get_tile_size(c_args.a_tensor_cb);
    const uint32_t cb_num_pages = c_args.num_tiles_per_cycle;

    for (uint32_t i = 0; i < args.num_shards; ++i) {
        uint32_t shard_id = args.shard_id + i * args.next_shard_offset;
        auto a_tensor_shard_pages = a_tensor.shard_pages(shard_id);
        auto b_tensor_shard_pages = b_tensor.shard_pages(shard_id);

        auto a_page_iter = a_tensor_shard_pages.begin();
        auto b_page_iter = b_tensor_shard_pages.begin();
        const auto a_end = a_tensor_shard_pages.end();
        // Read in batches of cb_num_pages. Consecutive pages in iterator have consecutive NOC addrs (see operator++).
        while (a_page_iter != a_end) {
            uint64_t a_addr = a_page_iter->noc_addr();
            uint64_t b_addr = b_page_iter->noc_addr();
            noc_async_read(a_addr, get_write_ptr(c_args.a_tensor_cb), cb_num_pages * page_size);
            noc_async_read(b_addr, get_write_ptr(c_args.b_tensor_cb), cb_num_pages * page_size);
            noc_async_read_barrier();
            cb_push_back(c_args.a_tensor_cb, cb_num_pages);
            cb_push_back(c_args.b_tensor_cb, cb_num_pages);
            for (uint32_t k = 0; k < cb_num_pages && a_page_iter != a_end; ++k) {
                ++a_page_iter;
                ++b_page_iter;
            }
        }
    }

    DPRINT << "Reader kernel completed" << ENDL();
}
