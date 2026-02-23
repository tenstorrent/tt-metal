// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "elt_nd_sharded_add_reader_args.hpp"
#include "ttnn/kernel/kernel_utils.hpp"
#include "../tiles_config.hpp"

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel::eltwise::add_nd_sharded_args;
    auto args = make_runtime_struct_from_args<ElemwiseWriterKernelArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<CompileTimeWriterKernelArgs>();

    constexpr auto dst_args = TensorAccessorArgs<amount_of_fields<CompileTimeWriterKernelArgs>()>();
    const auto dst_tensor = TensorAccessor(dst_args, args.dst_base_addr, get_tile_size(c_args.cb_dst));

    constexpr uint32_t num_tiles_per_cycle = c_args.num_tiles_per_cycle;

    const auto page_size = get_tile_size(c_args.cb_dst);
    const uint32_t cb_num_pages = c_args.num_tiles_per_cycle;

    for (uint32_t i = 0; i < args.num_shards; ++i) {
        uint32_t shard_id = args.shard_id + i * args.next_shard_offset;
        for (uint32_t cycle = 0; cycle < args.num_cycles_per_shard; ++cycle) {
            uint32_t offset_bytes = cycle * cb_num_pages * page_size;
            uint64_t output_tensor_noc_addr = dst_tensor.get_shard_noc_addr(shard_id, offset_bytes);

            DPRINT << "Writing shard " << shard_id << " cycle " << cycle << ENDL();
            cb_wait_front(c_args.cb_dst, cb_num_pages);
            uint32_t l1_read_addr = get_read_ptr(c_args.cb_dst);
            noc_async_write(l1_read_addr, output_tensor_noc_addr, cb_num_pages * page_size);
            noc_async_write_barrier();
            cb_pop_front(c_args.cb_dst, cb_num_pages);
        }
    }
}
