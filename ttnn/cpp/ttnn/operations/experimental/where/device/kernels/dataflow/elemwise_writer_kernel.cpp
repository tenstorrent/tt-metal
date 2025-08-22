// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "elemwise_writer_kernel_args.hpp"
#include "ttnn/kernel/kernel_utils.hpp"

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel::eltwise::where_args;
    auto args = make_runtime_struct_from_args<ElemwiseWriterKernelArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<CompileTimeWriterKernelArgs>();

    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto dst_tensor = TensorAccessor(dst_args, args.dst_base_addr, get_tile_size(c_args.cb_dst));

    constexpr uint32_t onetile = 1;
    uint32_t end_id = args.tile_ofs + args.num_tiles;
    for (uint32_t i = args.tile_ofs; i < end_id; ++i) {
        cb_wait_front(c_args.cb_dst, onetile);
        uint32_t l1_read_addr = get_read_ptr(c_args.cb_dst);
        noc_async_write_tile(i, dst_tensor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(c_args.cb_dst, onetile);
    }
}
