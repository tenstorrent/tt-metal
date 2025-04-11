// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/softmax.h"
#include "compute_kernel_api/reduce.h"

#include "debug/dprint.h"
#include "debug/dprint_tensix.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch

void calc_numeric_stable(
    uint32_t Wt, uint32_t ndst, uint32_t cb_in, uint32_t cb_bcast_scaler, uint32_t cb_max, uint32_t cb_out) {
    // calculate max val per row
    ACQ();
    reconfig_data_format(cb_in, cb_bcast_scaler);
    cb_reserve_back(cb_max, 1);
    cb_wait_front(cb_bcast_scaler, 1);
    reduce_init_delta<false, PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_bcast_scaler, cb_max);
    for (uint32_t wt = 0; wt < Wt; wt++) {
        cb_wait_front(cb_in, wt + 1);
        constexpr uint32_t bcast_scaler0 = 0;
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_bcast_scaler, wt, bcast_scaler0, 0);
    }
    reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_max);
    pack_tile(0, cb_max);
    cb_push_back(cb_max, 1);
    REL();

    // calculate x-max(x)
    exp_tile_init<EXP_APPROX>();
    reconfig_data_format_srcb(cb_max);
    cb_wait_front(cb_max, 1);
    sub_bcast_cols_init_short(cb_in, cb_max);
    for (uint32_t wt = 0; wt < Wt; wt += ndst) {
        ACQ();
        for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
            sub_tiles_bcast_cols(cb_in, cb_max, wt + wt8, 0, wt8);
        }
        cb_reserve_back(cb_out, ndst);
        for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
            exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
            pack_tile(wt8, cb_out);     // reuse the exps buffer again, this time in a circular manner
        }
        cb_push_back(cb_out, ndst);
        REL();
    }
    cb_pop_front(cb_in, Wt);
    cb_pop_front(cb_max, 1);
    cb_wait_front(cb_out, Wt);
}

namespace NAMESPACE {
void MAIN {
    const uint32_t NCHt = get_arg_val<uint32_t>(0);
    const uint32_t Ht = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t ndst = get_arg_val<uint32_t>(3);
    const uint32_t start_ht = get_arg_val<uint32_t>(4);
    const uint32_t mask_padded_data = get_arg_val<uint32_t>(5);
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_6);

    constexpr uint32_t onetile = 1;
    // reserve one tile for zeros on cb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto cb_bcast_scaler = tt::CBIndex::c_2;
    constexpr auto cb_fused_scale = tt::CBIndex::c_3;
    constexpr auto cb_fused_attn = tt::CBIndex::c_4;
    constexpr auto cb_mask_padded = tt::CBIndex::c_5;
    constexpr auto cb_exps = tt::CBIndex::c_6;
    constexpr auto cb_scale_mask = tt::CBIndex::c_9;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_7;
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_11;
#ifdef NUMERIC_STABLE
    constexpr auto cb_max = tt::CBIndex::c_8;
    constexpr auto cb_x = tt::CBIndex::c_10;
#else
    constexpr auto cb_x = cb_exps;
#endif

    cb_wait_front(cb_bcast_scaler, 1);  // comes from the reader

#if FUSED_SCALE_MASK
    cb_wait_front(cb_fused_scale, 1);
#endif
    uint32_t Wt_blocks = Wt / blocks;
    Wt_blocks += Wt % blocks == 0 ? 0 : 1;

    // First loop is to parse and find the sum
    constexpr int dst0 = 0;
    uint32_t ht = start_ht;
    bool wait_mask = true;
    // Current kenrel restrictions, RM and needs to be disible by ndst and wt_block
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#ifdef NUMERIC_STABLE
        // finds the max, through a max redue
        // packconfig
        // unpackconfig
        // reduce_init
        // extra cb for holding partial reduces
        for (uint32_t wt_block = 0; wt_block < Wt_blocks; wt_block += blocks) {
            for (uint_32t b = 0; b < block and wt_block + b < Wt; b += ndst) {
                // cb_wait_fronts
                for (int dst_reg = 0; dst_reg < ndst; dst_reg++) {
                    // reduced
                }
                // cb_pop_front
            }
        }
        // reduce_revert_delta
#endif
        // Finds the denominator
        for (uint32_t wt_block = 0; wt_block < Wt_blocks; wt_block += blocks) {
            for (uint_32t b = 0; b < block and wt_block + b < Wt; b += ndst) {
                for (int dst_reg = 0; dst_reg < ndst; dst_reg++) {
                }
            }
        }
        // Finds the final value
        for (uint32_t wt_block = 0; wt_block < Wt_blocks; wt_block += blocks) {
            for (uint_32t b = 0; b < block and wt_block + b < Wt; b += ndst) {
            }
        }
    }
}
}  // namespace NAMESPACE
