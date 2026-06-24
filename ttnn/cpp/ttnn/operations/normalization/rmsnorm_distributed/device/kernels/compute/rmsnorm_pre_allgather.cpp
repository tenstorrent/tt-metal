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

    constexpr uint32_t cb_x2 = tt::CBIndex::c_6;                           // x**2
    constexpr uint32_t cb_res = tt::CBIndex::c_5;                          // residual b (unused when !FUSE_PRE_ADD)
    constexpr uint32_t cb_inp = FUSE_PRE_ADD ? tt::CBIndex::c_3 : cb_in0;  // fused a + b, or just a

    if constexpr (FUSE_PRE_ADD) {
        binary_op_init_common(cb_in0, cb_res, cb_inp);
    } else {
        binary_op_init_common(cb_inp, cb_reduce, cb_x2);
    }

    // 2D walk for the squaring chain: outer = Wt/blk subblocks, inner = blk tiles.
    // Total iterations = Wt. Wt is assumed divisible by blk (matches the original
    // loop `for (wt = 0; wt < Wt; wt += blk)` which has no tail handling).
    constexpr auto squaring_shape = ckl::EltwiseShape::of(Wt / blk, blk);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Fuse pre-add: cb_inp = cb_in0 + cb_res (no-op when !FUSE_PRE_ADD). Migrated from
        // pre_add::one_row to eltwise_chain: per-block (blk) bulk add over Wt tiles via
        // Bulk + Block index, reproducing one_row's wait/pop/reserve/push(blk) loop. Reconfig
        // Input (reconfig_data_format + add_tiles_init) + Output (pack_reconfig_data_format).
        if constexpr (FUSE_PRE_ADD) {
            ckl::add<
                cb_in0,
                cb_res,
                cb_inp,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::Bulk,
                ckl::InputLifecycle::Bulk,
                ckl::OutputLifecycle::Bulk,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::PackTileReconfig::Output,
                ckl::OperandKind::Block,
                ckl::OperandKind::Block>(squaring_shape);
        }

        // x**2 — same-CB FPU mul. cb_inp lifecycle: InputLifecycle::HeldCumulative (chain emits
        // cumulative `cb_wait_front(cb_inp, (i+1)*blk)` per blk-chunk; never pops).
        // The caller pops Wt from cb_inp after the reduce below. cb_x2 lifecycle:
        // OutputLifecycle::Chunked (chain emits reserve_back(blk) + push_back(blk) per chunk;
        // pack writes absolute slots via Block index).
        ckl::square<
            cb_inp,
            cb_x2,
            ckl::InputLifecycle::HeldCumulative,
            ckl::OutputLifecycle::Bulk,
            ckl::BinaryDataFormatReconfig::Input,
            ckl::PackTileReconfig::Output,
            ckl::OperandKind::Block>(squaring_shape);

        // sum(x**2) — BulkWaitBulkPop: all Wt tiles already in CB.
        ckl::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            cb_x2,
            cb_reduce,
            cb_out,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(ckl::ReduceInputBlockShape::row(Wt));
        cb_pop_front(cb_inp, Wt);
    }
    cb_pop_front(cb_reduce, 1);
}
