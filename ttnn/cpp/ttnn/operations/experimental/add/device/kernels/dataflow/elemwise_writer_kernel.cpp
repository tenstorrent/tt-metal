// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "elemwise_writer_kernel_args.hpp"
#include "ttnn/kernel/kernel_utils.hpp"
#include "../tiles_config.hpp"

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel::eltwise::add_args;
    auto args = make_runtime_struct_from_args<ElemwiseWriterKernelArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<CompileTimeWriterKernelArgs>();

    constexpr auto dst_args = TensorAccessorArgs<amount_of_fields<CompileTimeWriterKernelArgs>()>();
    const auto dst_tensor = TensorAccessor(dst_args, args.dst_base_addr, get_tile_size(c_args.cb_dst));

    constexpr uint32_t num_tiles_per_cycle = c_args.num_tiles_per_cycle;
    const uint32_t tile_size = get_tile_size(c_args.cb_dst);
    uint32_t end_id = args.tile_ofs + args.num_tiles;

    for (uint32_t i = args.tile_ofs; i < end_id; i += num_tiles_per_cycle) {
        DPRINT << "Writing tile " << i << " to output circular buffer" << ENDL();
        cb_wait_front(c_args.cb_dst, num_tiles_per_cycle);
        uint32_t l1_read_addr = get_read_ptr(c_args.cb_dst);
        for (uint32_t k = 0; k < num_tiles_per_cycle; ++k) {
            noc_async_write_tile(i + k, dst_tensor, l1_read_addr + k * tile_size);
        }
        noc_async_write_barrier();
        cb_pop_front(c_args.cb_dst, num_tiles_per_cycle);
    }
}
