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
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto NCHt = get_arg(args::NCHt);                // Number of NCH tiles
    const auto Wt = get_arg(args::Wt);                    // Width in tiles
    const auto tile_offset = get_arg(args::tile_offset);  // Tile offset for this core

    constexpr auto blk = get_arg(args::blk);

    Noc noc;
    // Input tiles, consumed downstream by the compute kernel.
    DataflowBuffer dfb_inp_buf(dfb::inp);

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = dfb_inp_buf.get_tile_size();

    // The reduce-scalar tile the compute kernel's row reduction multiplies by.
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::reduce, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    const auto src_a = TensorAccessor(tensor::src);

#ifdef FUSE_PRE_ADD
    // Residual tiles, added to the input by the compute kernel before the statistics pass.
    DataflowBuffer dfb_res_buf(dfb::res);
    const uint32_t src1_tile_bytes = dfb_res_buf.get_tile_size();
    const auto src_b = TensorAccessor(tensor::res_src);
#endif

    uint32_t inp_tile_idx = tile_offset;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            for (uint32_t r = 0; r < blk; r++) {
                dfb_inp_buf.reserve_back(1);
                noc.async_read(src_a, dfb_inp_buf, src0_tile_bytes, {.page_id = inp_tile_idx}, {.offset_bytes = 0});
#ifdef FUSE_PRE_ADD
                dfb_res_buf.reserve_back(1);
                noc.async_read(src_b, dfb_res_buf, src1_tile_bytes, {.page_id = inp_tile_idx}, {.offset_bytes = 0});
#endif
                inp_tile_idx++;
                noc.async_read_barrier();
                dfb_inp_buf.push_back(1);
#ifdef FUSE_PRE_ADD
                dfb_res_buf.push_back(1);
#endif
            }
        }  // wt loop
    }  // ncht loop
}
