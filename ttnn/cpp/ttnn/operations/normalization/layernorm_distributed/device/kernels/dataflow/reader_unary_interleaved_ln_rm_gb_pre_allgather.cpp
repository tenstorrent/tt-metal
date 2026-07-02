// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs from interleaved dram.
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/debug/assert.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);     // Source address in dram
    const uint32_t NCHt = get_arg_val<uint32_t>(1);         // Number of NCH tiles
    const uint32_t Wt = get_arg_val<uint32_t>(2);           // Width in tiles
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);  // Tile offset for this core

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_inp);

    constexpr uint32_t blk = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_reduce, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    const auto src_a = TensorAccessor(src_args, src_addr);

    Noc noc;
    DataflowBuffer cb_inp_buf(cb_inp);

#if FUSE_PRE_ADD
    const uint32_t res_addr = get_arg_val<uint32_t>(4);  // Residual source address in dram
    constexpr uint32_t cb_res = tt::CBIndex::c_5;
    const uint32_t src1_tile_bytes = get_tile_size(cb_res);
    constexpr auto res_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src_b = TensorAccessor(res_args, res_addr);
    DataflowBuffer cb_res_buf(cb_res);
#endif

    uint32_t inp_tile_idx = tile_offset;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            for (uint32_t r = 0; r < blk; r++) {
                cb_inp_buf.reserve_back(1);
                noc.async_read(src_a, cb_inp_buf, src0_tile_bytes, {.page_id = inp_tile_idx}, {.offset_bytes = 0});
#if FUSE_PRE_ADD
                cb_res_buf.reserve_back(1);
                noc.async_read(src_b, cb_res_buf, src1_tile_bytes, {.page_id = inp_tile_idx}, {.offset_bytes = 0});
#endif
                inp_tile_idx++;
                noc.async_read_barrier();
                cb_inp_buf.push_back(1);
#if FUSE_PRE_ADD
                cb_res_buf.push_back(1);
#endif
            }
        }  // wt loop
    }  // ncht loop
}
