// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    // A deferred residual write to settle into the stream. Where there is none the
    // accessor below aliases running_sum, so the offsets hold either way and nothing
    // reads through it.
    constexpr bool fuse_add = get_compile_time_arg_val(1) == 1;
    constexpr auto prefix_args = TensorAccessorArgs<2>();
    constexpr auto partial_args = TensorAccessorArgs<prefix_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<partial_args.next_compile_time_args_offset()>();
    constexpr auto pending_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();

    // runtime args. The two passes are split across different core counts — statistics
    // by token row, the fold by output tile — so a core carries a run of each and the
    // runs are unrelated. `num_stat_rows` is zero on a core that only folds.
    const auto prefix_addr = get_arg_val<uint32_t>(0);
    const auto partial_addr = get_arg_val<uint32_t>(1);
    const auto q_addr = get_arg_val<uint32_t>(2);
    const auto num_stat_rows = get_arg_val<uint32_t>(3);
    const auto stat_start_row = get_arg_val<uint32_t>(4);
    const auto num_fold_tiles = get_arg_val<uint32_t>(5);
    const auto fold_start_tile = get_arg_val<uint32_t>(6);
    const auto pending_addr = get_arg_val<uint32_t>(7);

    // The partial's read site, in whole Ht*Wt planes. Every core reads the same
    // site, and the host re-patches it in place on a program-cache hit.
    const auto partial_page_offset = get_common_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_scaler = 1;
    constexpr uint32_t cb_id_q = 2;
    constexpr uint32_t cb_id_prefix = 3;
    constexpr uint32_t cb_id_partial = 0;
    constexpr uint32_t cb_id_pending = 10;

    constexpr uint32_t prefix_tile_bytes = get_tile_size(cb_id_prefix);
    constexpr uint32_t partial_tile_bytes = get_tile_size(cb_id_partial);
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_id_q);
    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer prefix_buf(cb_id_prefix);
    DataflowBuffer partial_buf(cb_id_partial);
    DataflowBuffer q_buf(cb_id_q);
    DataflowBuffer pending_buf(cb_id_pending);

    auto prefix_accessor = TensorAccessor(prefix_args, prefix_addr);
    auto partial_accessor = TensorAccessor(partial_args, partial_addr);
    auto q_accessor = TensorAccessor(q_args, q_addr);
    auto pending_accessor = TensorAccessor(pending_args, pending_addr);

    // Pass one, on the cores that carry token rows. The scaler and q feed only this
    // pass, so a fold-only core must not push them either: compute pops them exactly
    // when it runs the pass, and an unmatched push leaves the buffer full for the
    // rest of the program.
    if (num_stat_rows > 0) {
        // Both reductions are plain sums; the mean this feeds is taken against the
        // full unsharded `d` downstream, not against this rank's share.
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<cb_id_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

        // q spans the same d as a row of the stream and is the same for every row
        // this core owns, so it is read once and left resident.
        q_buf.reserve_back(Wt);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            noc.async_read(q_accessor, q_buf, q_tile_bytes, {.page_id = wt}, {.offset_bytes = wt * q_tile_bytes});
        }
        noc.async_read_barrier();
        q_buf.push_back(Wt);

        for (uint32_t r = 0; r < num_stat_rows; ++r) {
            const uint32_t base_page = (stat_start_row + r) * Wt;
            prefix_buf.reserve_back(Wt);
            if constexpr (fuse_add) {
                pending_buf.reserve_back(Wt);
            }
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                noc.async_read(
                    prefix_accessor,
                    prefix_buf,
                    prefix_tile_bytes,
                    {.page_id = base_page + wt},
                    {.offset_bytes = wt * prefix_tile_bytes});
                if constexpr (fuse_add) {
                    noc.async_read(
                        pending_accessor,
                        pending_buf,
                        prefix_tile_bytes,
                        {.page_id = base_page + wt},
                        {.offset_bytes = wt * prefix_tile_bytes});
                }
            }
            noc.async_read_barrier();
            prefix_buf.push_back(Wt);
            if constexpr (fuse_add) {
                pending_buf.push_back(Wt);
            }
        }
    }

    // Pass two: a contiguous run of output tiles, the same indexing the output has.
    // `running_sum` is read a second time here rather than held across the exchange —
    // this core's fold run and its statistics run are different token rows now, so
    // there is nothing to hold.
    //
    // A deferred write is read again too rather than the settled sum being read back:
    // the sum is parked in DRAM by whichever core reduced that row, and waiting for it
    // to land would put every fold core's prefetch behind the exchange. The fold
    // distributes the row weight over the two addends instead.
    for (uint32_t i = fold_start_tile; i < fold_start_tile + num_fold_tiles; ++i) {
        partial_buf.reserve_back(onetile);
        prefix_buf.reserve_back(onetile);
        if constexpr (fuse_add) {
            pending_buf.reserve_back(onetile);
        }
        noc.async_read(
            partial_accessor,
            partial_buf,
            partial_tile_bytes,
            {.page_id = i + partial_page_offset},
            {.offset_bytes = 0});
        noc.async_read(prefix_accessor, prefix_buf, prefix_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        if constexpr (fuse_add) {
            noc.async_read(pending_accessor, pending_buf, prefix_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        partial_buf.push_back(onetile);
        prefix_buf.push_back(onetile);
        if constexpr (fuse_add) {
            pending_buf.push_back(onetile);
        }
    }
}
