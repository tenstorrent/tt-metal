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
    const auto row_stride = get_arg(args::row_stride);    // Tiles between the starts of consecutive input rows
    const bool is_merge_core = get_arg(args::is_merge_core);
    const auto reduce_core_noc_x = get_arg(args::reduce_core_noc_x);
    const auto reduce_core_noc_y = get_arg(args::reduce_core_noc_y);
    const auto y = get_arg(args::y);
    // Merge core only: the NoC rectangle of the other cores in its column, which it signals when the
    // gather buffer is free for the next row's partials.
    const auto workers_noc_x_start = get_arg(args::workers_noc_x_start);
    const auto workers_noc_y_start = get_arg(args::workers_noc_y_start);
    const auto workers_noc_x_end = get_arg(args::workers_noc_x_end);
    const auto workers_noc_y_end = get_arg(args::workers_noc_y_end);

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
    // On the merge core: counts the partials that have landed in its gather buffer for the current row.
    Semaphore reducer_sem(sem::reducer);
    // On the other cores: incremented by the merge core when its gather buffer can take this core's next partial.
    Semaphore gather_free_sem(sem::gather_free);

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

    // Partial statistics use the intermediate format, which is Float32 with fp32_dest_acc_en.
    const uint32_t o_write_size = dfb_out_buf.get_tile_size();
    const uint32_t worker_offset = o_write_size * y;
    // The gather buffer is laid out identically on every core in the column, so this core's own
    // write pointer gives the same base address the merge core's instance has. Only the merge core
    // pushes into its gather buffer, and it always pushes the full buffer depth (num_cores_to_wait
    // tiles), so its write pointer is back at that base at the start of every row.
    const uint32_t gather_base_addr = dfb_x2_merge_buf.get_write_ptr();
    UnicastEndpoint reduce_ep;

    // Sends one row's partial statistic to the merge core and, on the merge core, gathers the
    // column's partials for that row into the gather buffer. The gather buffer holds one row, so the
    // merge core first waits until its compute has consumed the previous row and then releases the
    // other cores in the column to write. A core increments the merge core's semaphore only after
    // receiving that release, so the merge core's reset of the semaphore after each row cannot drop
    // an increment that belongs to the next row.
    auto send_partial = [&]() {
        if (is_merge_core) {
            dfb_x2_merge_buf.reserve_back(num_cores_to_wait);
            if constexpr (num_cores_to_wait > 1) {
                gather_free_sem.inc_multicast(
                    noc,
                    workers_noc_x_start,
                    workers_noc_y_start,
                    workers_noc_x_end,
                    workers_noc_y_end,
                    1,
                    num_cores_to_wait - 1);
            }
        } else {
            gather_free_sem.wait(1);
            gather_free_sem.set(0);
        }

        // wait on the partial output and then write it to the merge core over the NoC. The write
        // itself lands on the remote core, not here.
        dfb_out_buf.wait_front(onetile);
        noc.async_write(
            dfb_out_buf,
            reduce_ep,
            o_write_size,
            {.offset_bytes = 0},
            {.noc_x = reduce_core_noc_x, .noc_y = reduce_core_noc_y, .addr = gather_base_addr + worker_offset});
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
    };

    uint32_t row_start_tile_idx = tile_offset;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // read input tiles
        uint32_t inp_tile_idx = row_start_tile_idx;
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
        row_start_tile_idx += row_stride;

        // Send the previous row's partial only after this row's input is in flight to compute, so
        // the compute kernel does not wait for input while the reader sends the partial and waits
        // for the column's partials to be gathered.
        if (ncht > 0) {
            send_partial();
        }
    }  // ncht loop

    // Send the last row's partial; the loop sends each row's partial one row late.
    if (NCHt > 0) {
        send_partial();
    }
}
