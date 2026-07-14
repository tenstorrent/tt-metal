// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm statistics.
For rmsnorm it computes E(x**2) and returns it as a one tile wide output
 */

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/dataflow/circular_buffer.h"  // CircularBuffer (was transitively via pre_add.h)
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // add

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t NCHt = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t blk = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_y = get_compile_time_arg_val(3);
    bool is_merge_core = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce_id = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t cb_x2_id = tt::CBIndex::c_6;  // x**2
    constexpr uint32_t cb_zero_id = tt::CBIndex::c_13;
    constexpr uint32_t cb_res_id = tt::CBIndex::c_5;  // residual b (unused when !FUSE_PRE_ADD)
    constexpr uint32_t cb_inp_id = FUSE_PRE_ADD ? tt::CBIndex::c_3 : cb_in0_id;  // fused a + b, or just a

    if constexpr (FUSE_PRE_ADD) {
        binary_op_init_common(cb_in0_id, cb_res_id, cb_inp_id);
    } else {
        binary_op_init_common(cb_inp_id, cb_reduce_id, cb_x2_id);
    }

    CircularBuffer cb_in0(cb_in0_id);
    CircularBuffer cb_res(cb_res_id);
    CircularBuffer cb_inp(cb_inp_id);
    CircularBuffer cb_x2(cb_x2_id);
    CircularBuffer cb_reduce(cb_reduce_id);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        if constexpr (FUSE_PRE_ADD) {
            ckl::add<
                cb_in0_id,
                cb_res_id,
                cb_inp_id,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::Bulk,
                ckl::InputLifecycle::Bulk,
                ckl::OutputLifecycle::Bulk,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::PackTileReconfig::Output,
                ckl::OperandKind::Block,
                ckl::OperandKind::Block>(ckl::EltwiseShape::of(Wt / blk, blk));
        }

        /*
         * x**2
         */
        reconfig_data_format(cb_inp_id, cb_inp_id);
        pack_reconfig_data_format(cb_x2_id);
        mul_tiles_init(cb_inp_id, cb_inp_id);

        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_inp.wait_front(wt + blk);  // cumulative wait

            tile_regs_acquire();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles(cb_inp_id, cb_inp_id, wt + wtr, wt + wtr, wtr);
            }
            tile_regs_commit();

            cb_x2.reserve_back(blk);

            tile_regs_wait();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                pack_tile(wtr, cb_x2_id, wt + wtr);
            }
            tile_regs_release();

            cb_x2.push_back(blk);
        }

        /*
         * sum(x**2)
         */
        // BulkWaitBulkPop: All Wt tiles already in CB (see cumulative wait above)
        ckl::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            cb_x2_id,
            cb_reduce_id,
            cb_out,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(ckl::ReduceInputBlockShape::row(Wt));
        cb_inp.pop_front(Wt);
        cb_reduce.pop_front(1);
    }

    // if merge core, we need to do a final sum on the tile in cb_x2_id and then write the result to cb_out_final_id
    if (is_merge_core) {
        constexpr uint32_t cb_x2_merge_id = tt::CBIndex::c_15;
        constexpr uint32_t cb_out_final_id = tt::CBIndex::c_14;

        // Initialize accumulation
        binary_op_init_common(cb_x2_merge_id, cb_zero_id, cb_out_final_id);
        using Accumulate = ckl::BinaryFpu<
            cb_x2_merge_id,
            cb_zero_id,
            ckl::BinaryFpuOp::Add,
            ckl::BroadcastDim::None,
            ckl::InputLifecycle::Bulk,
            ckl::InputLifecycle::Bulk,
            ckl::BinaryDataFormatReconfig::Input,
            ckl::Dst::D0,
            ckl::OperandKind::Block,
            ckl::OperandKind::Scalar,
            ckl::TileOffset::Unset,
            ckl::TileOffset::Unset,
            ckl::DestAccumulation::Enabled>;
        using Pack = ckl::PackTile<
            cb_out_final_id,
            ckl::OutputLifecycle::DestAccumulation,
            ckl::PackTileReconfig::Output,
            ckl::Dst::D0>;

        ckl::eltwise_chain(ckl::EltwiseShape::tiles(num_cores_y, num_cores_y), Accumulate{}, Pack{});
    }
}
