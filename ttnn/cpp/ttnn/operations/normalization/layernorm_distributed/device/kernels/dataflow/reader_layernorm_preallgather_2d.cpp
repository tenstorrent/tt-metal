// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs from interleaved dram.
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/debug/assert.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto NCHt = get_arg(args::NCHt);                // Number of NCH tiles
    const auto Wt = get_arg(args::Wt);                    // Width in tiles
    const auto tile_offset = get_arg(args::tile_offset);  // Tile offset for this core
    const bool is_merge_core = get_arg(args::is_merge_core);
    const auto reduce_core_noc_x = get_arg(args::reduce_core_noc_x);
    const auto reduce_core_noc_y = get_arg(args::reduce_core_noc_y);
    const auto y = get_arg(args::y);

    const uint32_t onetile = 1;

    constexpr auto blk = get_arg(args::blk);
    constexpr auto num_cores_to_wait = get_arg(args::num_cores_to_wait);

    const auto src_a = TensorAccessor(tensor::src);

    Noc noc;
    // Input tiles, consumed downstream by the compute kernel.
    DataflowBuffer dfb_inp_buf(dfb::inp);
    // This core's partial statistic: produced by compute, then shipped over the NoC to the merge core.
    DataflowBuffer dfb_out_buf(dfb::out);
    // Gather buffer on the merge core: every core in the column lands its partial here.
    DataflowBuffer dfb_x2_merge_buf(dfb::x2_merge);
    Semaphore<> reducer_sem(sem::reducer);

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = dfb_inp_buf.get_tile_size();

#ifdef FUSE_PRE_ADD
    // Residual tiles, added to the input by the compute kernel before the statistics pass.
    DataflowBuffer dfb_res_buf(dfb::res);
    const uint32_t src1_tile_bytes = dfb_res_buf.get_tile_size();
    const auto src_b = TensorAccessor(tensor::res_src);
#endif

    // Generate constant tiles for reduce scalar
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        dfb::reduce,
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    if (is_merge_core) {
        dataflow_kernel_lib::prepare_zero_tile<dfb::zero>();
    }

    uint32_t inp_tile_idx = tile_offset;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // read input tiles
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            dfb_inp_buf.reserve_back(blk);
#ifdef FUSE_PRE_ADD
            dfb_res_buf.reserve_back(blk);
#endif

            for (uint32_t r = 0; r < blk; r++) {
                noc.async_read(
                    src_a,
                    dfb_inp_buf,
                    src0_tile_bytes,
                    {.page_id = inp_tile_idx},
                    {.offset_bytes = r * src0_tile_bytes});
#ifdef FUSE_PRE_ADD
                noc.async_read(
                    src_b,
                    dfb_res_buf,
                    src1_tile_bytes,
                    {.page_id = inp_tile_idx},
                    {.offset_bytes = r * src1_tile_bytes});
#endif
                inp_tile_idx++;
            }
            noc.async_read_barrier();

            dfb_inp_buf.push_back(blk);
#ifdef FUSE_PRE_ADD
            dfb_res_buf.push_back(blk);
#endif

        }  // wt loop

    }  // ncht loop

    // wait on the partial output and then write it to the merge core over the NoC
    dfb_out_buf.wait_front(onetile);

    // Partial statistics use the intermediate format, which is Float32 with fp32_dest_acc_en.
    uint32_t o_write_size = dfb_out_buf.get_tile_size();
    uint32_t worker_offset = o_write_size * y;

    UnicastEndpoint reduce_ep;
    noc.async_write(
        dfb_out_buf,
        reduce_ep,
        o_write_size,
        {.offset_bytes = 0},
        {.noc_x = reduce_core_noc_x,
         .noc_y = reduce_core_noc_y,
         // The gather buffer is laid out identically on every core in the column, so this core's own
         // write pointer gives the same base address the merge core's instance has. The write itself
         // lands on the remote core, not here.
         .addr = dfb_x2_merge_buf.get_write_ptr() + worker_offset});
    noc.async_write_barrier();
    dfb_out_buf.pop_front(onetile);

    // increase semaphore
    reducer_sem.up(noc, reduce_core_noc_x, reduce_core_noc_y, 1);
    noc.async_atomic_barrier();

    if (is_merge_core) {
        reducer_sem.wait(num_cores_to_wait);
        dfb_x2_merge_buf.push_back(num_cores_to_wait);
        reducer_sem.set(0);
    }
}
