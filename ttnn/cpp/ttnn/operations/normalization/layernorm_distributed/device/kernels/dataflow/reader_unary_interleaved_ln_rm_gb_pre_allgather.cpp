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

    uint32_t inp_tile_idx = tile_offset;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            for (uint32_t r = 0; r < blk; r++) {
                dfb_inp_buf.reserve_back(1);
                noc.async_read(src_a, dfb_inp_buf, src0_tile_bytes, {.page_id = inp_tile_idx}, {.offset_bytes = 0});
                // Residual tiles, added to the input by the compute kernel before the statistics pass.
                with_nullable_token(dfb::res, [&](const DFBBindingToken& res_tok) {
                    with_nullable_token(tensor::res_src, [&](const auto& res_src_tok) {
                        DataflowBuffer dfb_res_buf(res_tok);
                        const auto src_b = TensorAccessor(res_src_tok);
                        dfb_res_buf.reserve_back(1);
                        noc.async_read(
                            src_b,
                            dfb_res_buf,
                            dfb_res_buf.get_tile_size(),
                            {.page_id = inp_tile_idx},
                            {.offset_bytes = 0});
                    });
                });
                inp_tile_idx++;
                noc.async_read_barrier();
                dfb_inp_buf.push_back(1);
                with_nullable_token(dfb::res, [&](const DFBBindingToken& res_tok) {
                    DataflowBuffer dfb_res_buf(res_tok);
                    dfb_res_buf.push_back(1);
                });
            }
        }  // wt loop
    }  // ncht loop
}
