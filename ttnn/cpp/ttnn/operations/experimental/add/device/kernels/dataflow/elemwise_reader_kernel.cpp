// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "elemwise_reader_kernel_args.hpp"
#include "ttnn/kernel/kernel_utils.hpp"
#include "../tiles_config.hpp"

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel::eltwise::add_args;
    auto args = make_runtime_struct_from_args<ElemwiseReaderKernelArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<CompileTimeReaderKernelArgs>();

    constexpr auto a_tensor_args = TensorAccessorArgs<amount_of_fields<CompileTimeReaderKernelArgs>()>();
    constexpr auto b_tensor_args = TensorAccessorArgs<a_tensor_args.next_compile_time_args_offset()>();

    const auto a_tensor = TensorAccessor(a_tensor_args, args.a_tensor_base_addr, get_tile_size(c_args.a_tensor_cb));
    const auto b_tensor = TensorAccessor(b_tensor_args, args.b_tensor_base_addr, get_tile_size(c_args.b_tensor_cb));

    constexpr uint32_t num_tiles_per_cycle = c_args.num_tiles_per_cycle;
    const uint32_t a_tile_size = get_tile_size(c_args.a_tensor_cb);
    const uint32_t b_tile_size = get_tile_size(c_args.b_tensor_cb);
    for (uint32_t tile_id = args.tile_ofs; tile_id < args.tile_ofs + args.num_tiles; tile_id += num_tiles_per_cycle) {
        DPRINT << "Reading tile " << tile_id << " from input circular buffers" << ENDL();
        cb_reserve_back(c_args.a_tensor_cb, num_tiles_per_cycle);
        uint32_t a_write_ptr = get_write_ptr(c_args.a_tensor_cb);
        cb_reserve_back(c_args.b_tensor_cb, num_tiles_per_cycle);
        uint32_t b_write_ptr = get_write_ptr(c_args.b_tensor_cb);

        for (uint32_t k = 0; k < num_tiles_per_cycle; ++k) {
            noc_async_read_tile(tile_id + k, a_tensor, a_write_ptr + k * a_tile_size);
            noc_async_read_tile(tile_id + k, b_tensor, b_write_ptr + k * b_tile_size);
        }
        noc_async_read_barrier();

        cb_push_back(c_args.a_tensor_cb, num_tiles_per_cycle);
        cb_push_back(c_args.b_tensor_cb, num_tiles_per_cycle);
    }
    DPRINT << "Reader kernel completed" << ENDL();
}
