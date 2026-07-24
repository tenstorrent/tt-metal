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
#include "api/dataflow/circular_buffer.h"
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
        compute_kernel_hw_startup(cb_in0_id, cb_res_id, cb_inp_id);
    } else {
        compute_kernel_hw_startup(cb_inp_id, cb_reduce_id, cb_x2_id);
    }

    CircularBuffer cb_in0(cb_in0_id);
    CircularBuffer cb_res(cb_res_id);
    CircularBuffer cb_inp(cb_inp_id);
    CircularBuffer cb_reduce(cb_reduce_id);

    constexpr auto squaring_shape = ckl::EltwiseShape::tiles(Wt, blk);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        if constexpr (FUSE_PRE_ADD) {
            ckl::add<
                ckl::input(cb_in0_id, ckl::InputLifecycle::Chunked, ckl::OperandKind::Block),
                ckl::input(cb_res_id, ckl::InputLifecycle::Chunked, ckl::OperandKind::Block),
                ckl::output(cb_inp_id, ckl::OutputLifecycle::Chunked),
                ckl::BroadcastDim::None>(squaring_shape);
        }

        ckl::square<
            ckl::input(cb_inp_id, ckl::InputLifecycle::HeldCumulative, ckl::OperandKind::Block),
            ckl::output(cb_x2_id, ckl::OutputLifecycle::Chunked)>(squaring_shape);

        // sum(x**2)
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
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(num_cores_y),
            ckl::BinaryFpu<
                ckl::input(cb_x2_merge_id, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::input(cb_zero_id, ckl::InputLifecycle::HeldBulk),
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None,
                ckl::Dst::D0,
                ckl::DestAccumulation::Enabled>{},
            ckl::PackTile<ckl::output(
                cb_out_final_id,
                ckl::OutputLifecycle::DestAccumulation,
                ckl::DataFormatReconfig::Enabled,
                ckl::PackRelu::Disabled,
                ckl::L1Accumulation::Disabled,
                ckl::DestAccumulation::Enabled)>{});
    }
}
