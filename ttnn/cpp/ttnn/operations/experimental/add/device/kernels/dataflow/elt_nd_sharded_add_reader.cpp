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

    constexpr auto element_size = 16;

    constexpr auto a_tensor_args = TensorAccessorArgs<2>();
    constexpr auto b_tensor_args = TensorAccessorArgs<a_tensor_args.next_compile_time_args_offset()>();

    const auto a_tensor = TensorAccessor(a_tensor_args, args.a_tensor_base_addr, get_tile_size(c_args.a_tensor_cb));
    const auto b_tensor = TensorAccessor(b_tensor_args, args.b_tensor_base_addr, get_tile_size(c_args.b_tensor_cb));

    constexpr uint32_t cb_page_cnt = c_args.num_tiles_per_cycle;
    for (uint32_t shard_id = args.shard_id; shard_id < args.num_shards; shard_id += args.next_shard_offset) {
        uint64_t a_tensor_noc_addr = a_tensor.get_shard_noc_addr(shard_id);
        uint64_t b_tensor_noc_addr = b_tensor.get_shard_noc_addr(shard_id);

        noc_async_read(
            a_tensor_noc_addr, get_write_ptr(c_args.a_tensor_cb), a_tensor.dspec().shard_shape()[-1] * element_size);

        noc_async_read(
            b_tensor_noc_addr, get_write_ptr(c_args.b_tensor_cb), b_tensor.dspec().shard_shape()[-1] * element_size);

        noc_async_read_barrier();

        cb_push_back(c_args.a_tensor_cb, cb_page_cnt);
        cb_push_back(c_args.b_tensor_cb, cb_page_cnt);
    }

    DPRINT << "Reader kernel completed" << ENDL();
}
