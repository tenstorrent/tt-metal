// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm statistics.
 * For rmsnorm we compute E(x**2) and return it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_14;

    constexpr uint32_t cb_x2 = tt::CBIndex::c_6;
    constexpr uint32_t cb_res = tt::CBIndex::c_5;
    constexpr uint32_t cb_inp = FUSE_PRE_ADD ? tt::CBIndex::c_3 : cb_in0;

    if constexpr (FUSE_PRE_ADD) {
        binary_op_init_common(cb_in0, cb_res, cb_inp);
    } else {
        binary_op_init_common(cb_inp, cb_reduce, cb_x2);
    }

    constexpr auto squaring_shape = ckl::EltwiseShape::of(Wt / blk, blk);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        if constexpr (FUSE_PRE_ADD) {
            ckl::add<
                ckl::input(cb_in0, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::input(cb_res, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::output(cb_inp, ckl::OutputLifecycle::Bulk),
                ckl::BroadcastDim::None>(squaring_shape);
        }

        ckl::square<
            ckl::input(cb_inp, ckl::InputLifecycle::Pipelined, ckl::OperandKind::Block),
            ckl::output(cb_x2, ckl::OutputLifecycle::Bulk)>(squaring_shape);

        ckl::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            cb_x2,
            cb_reduce,
            cb_out,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(ckl::ReduceInputBlockShape::row(Wt));
    }
    cb_pop_front(cb_reduce, 1);
}
