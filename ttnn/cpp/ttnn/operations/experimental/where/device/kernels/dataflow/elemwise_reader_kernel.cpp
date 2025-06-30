// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "elemwise_reader_kernel_args.hpp"
#include "ttnn/kernel/kernel_utils.hpp"

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel::eltwise::where_args;
    auto args = make_runtime_struct_from_args<ElemwiseReaderKernelArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<CompileTimeReaderKernelArgs>();

    const InterleavedAddrGenFast<c_args.is_cond_tensor_in_dram> condition_tensor_addr_gen = {
        .bank_base_address = args.condition_tensor_base_addr,
        .page_size = get_tile_size(c_args.condition_cb),
        .data_format = get_dataformat(c_args.condition_cb)};

    const InterleavedAddrGenFast<c_args.is_true_tensor_in_dram> true_tensor_addr_gen = {
        .bank_base_address = args.true_tensor_base_addr,
        .page_size = get_tile_size(c_args.true_tensor_cb),
        .data_format = get_dataformat(c_args.true_tensor_cb)};

    const InterleavedAddrGenFast<c_args.is_false_tensor_in_dram> false_tensor_addr_gen = {
        .bank_base_address = args.false_tensor_base_addr,
        .page_size = get_tile_size(c_args.false_tensor_cb),
        .data_format = get_dataformat(c_args.false_tensor_cb)};

    constexpr uint32_t tile_cnt = 1;
    for (uint32_t tile_id = args.tile_ofs; tile_id < args.tile_ofs + args.num_tiles; tile_id++) {
        cb_reserve_back(c_args.condition_cb, tile_cnt);
        noc_async_read_tile(tile_id, condition_tensor_addr_gen, get_write_ptr(c_args.condition_cb));

        cb_reserve_back(c_args.true_tensor_cb, tile_cnt);
        noc_async_read_tile(tile_id, true_tensor_addr_gen, get_write_ptr(c_args.true_tensor_cb));

        cb_reserve_back(c_args.false_tensor_cb, tile_cnt);
        noc_async_read_tile(tile_id, false_tensor_addr_gen, get_write_ptr(c_args.false_tensor_cb));

        noc_async_read_barrier();

        cb_push_back(c_args.condition_cb, tile_cnt);
        cb_push_back(c_args.true_tensor_cb, tile_cnt);
        cb_push_back(c_args.false_tensor_cb, tile_cnt);
    }
}
