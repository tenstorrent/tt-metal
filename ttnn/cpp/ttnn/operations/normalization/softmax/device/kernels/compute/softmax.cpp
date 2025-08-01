// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_bcast_scaler, cb_max);
    for (uint32_t wt = 0; wt < Wt; wt++) {
        cb_wait_front(cb_in, wt + 1);
        constexpr uint32_t bcast_scaler0 = 0;
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_bcast_scaler, wt, bcast_scaler0, 0);
    }
    reduce_uninit();
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

    constexpr int dst0 = 0;
    uint32_t ht = start_ht;
    bool wait_mask = true;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#if FUSED_SCALE_MASK
        reconfig_data_format(cb_in0, cb_fused_scale);
        pack_reconfig_data_format(cb_scale_mask);
        mul_tiles_bcast_scalar_init_short(cb_in0, cb_fused_scale);
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            // apply fused scale [*= 1/sqrt(...)]
            ACQ();
            cb_wait_front(cb_in0, ndst);
            cb_reserve_back(cb_scale_mask, ndst);
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                mul_tiles_bcast_scalar(cb_in0, cb_fused_scale, wt8, 0, wt8);  // mul bcast-HW -> DST[wt8]
                pack_tile(wt8, cb_scale_mask);                                // reuse exps buffer
            }
            cb_push_back(cb_scale_mask, ndst);
            cb_pop_front(cb_in0, ndst);
            REL();
        }
        reconfig_data_format(cb_scale_mask, cb_fused_attn);

#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif

#ifdef CAUSAL_MASK
        add_tiles_init(cb_scale_mask, cb_fused_attn);
#else
        add_bcast_rows_init_short(cb_scale_mask, cb_fused_attn);
#endif
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            ACQ();
            cb_wait_front(cb_scale_mask, ndst);
#ifdef CAUSAL_MASK
            cb_wait_front(cb_fused_attn, wt + ndst);  // cumulative wait for up to Wt tiles
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                add_tiles(cb_scale_mask, cb_fused_attn, wt8, wt + wt8, wt8);  // tile *= 1/(sum(exp(x)))
            }
#else
            if (wait_mask) {
                cb_wait_front(cb_fused_attn, wt + ndst);  // cumulative wait for up to Wt tiles, only at first ht
            }

            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                add_tiles_bcast_rows(cb_scale_mask, cb_fused_attn, wt8, wt + wt8, wt8);  // tile *= 1/(sum(exp(x)))
            }
#endif
            cb_pop_front(cb_scale_mask, ndst);
            cb_reserve_back(cb_x, ndst);
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
#ifndef NUMERIC_STABLE
                exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
#endif
                pack_tile(wt8, cb_x);  // reuse the exps buffer again, this time in a circular manner
            }
            cb_push_back(cb_x, ndst);
            REL();
        }

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        calc_numeric_stable(Wt, ndst, cb_x, cb_bcast_scaler, cb_max, cb_exps);
#endif

#ifdef CAUSAL_MASK
        cb_pop_front(cb_fused_attn, Wt);
#else
        if (wait_mask) {
            wait_mask = false;
        }
        ht++;
        if (ht == Ht) {
            cb_pop_front(cb_fused_attn, Wt);
            ht = 0;
            wait_mask = true;
        }
#endif  // CAUSAL_MASK

        reconfig_data_format(cb_exps, cb_bcast_scaler);
#else
        reconfig_data_format(cb_in0, cb_in0);
        pack_reconfig_data_format(cb_exps);
        copy_tile_to_dst_init_short(cb_in0);  // need to copy from CB to DST to be able to run sfpu math
#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif
        if (mask_padded_data) {
            for (uint32_t wt = 0; wt < Wt; wt += ndst) {
                ACQ();
                cb_wait_front(cb_in0, ndst);
                for (uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                    if (wt == (Wt - ndst) && (wt8 == ndst - 1)) {
                        reconfig_data_format(cb_in0, cb_mask_padded);
                        add_bcast_rows_init_short(cb_in0, cb_mask_padded);
                        cb_wait_front(cb_mask_padded, 1);
                        add_tiles_bcast_rows(cb_in0, cb_mask_padded, wt8, 0, wt8);
                    } else {
                        copy_tile(cb_in0, wt8, wt8);  // copy from c_in[0] to DST[0]
                    }
                }
                cb_pop_front(cb_in0, ndst);

                cb_reserve_back(cb_x, ndst);
                for (uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
#ifndef NUMERIC_STABLE
                    exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
#endif
                    pack_tile(wt8, cb_x);  // DST[0]->cb_id[wt]
                }
                cb_push_back(cb_x, ndst);
                REL();
            }

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable(Wt, ndst, cb_x, cb_bcast_scaler, cb_max, cb_exps);
#endif

        } else {
// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable(Wt, ndst, cb_in0, cb_bcast_scaler, cb_max, cb_exps);
#else
            for (uint32_t wt = 0; wt < Wt; wt += ndst) {
                ACQ();
                cb_wait_front(cb_in0, ndst);
                for (uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                    copy_tile(cb_in0, wt8, wt8);  // copy from c_in[0] to DST[0]
                }
                cb_pop_front(cb_in0, ndst);

                cb_reserve_back(cb_exps, ndst);
                for (uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                    exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
                    pack_tile(wt8, cb_exps);    // DST[0]->cb_id[wt]
                }
                cb_push_back(cb_exps, ndst);
                REL();
            }
#endif
        }

        reconfig_data_format(cb_exps, cb_bcast_scaler);
#endif

        ACQ();
        cb_reserve_back(cb_recipsumexps, onetile);
        reduce_init(cb_exps, cb_bcast_scaler, cb_recipsumexps);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            cb_wait_front(cb_exps, wt + 1);        // must be a cumulative wait for correctness
            constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
            reduce_tile(cb_exps, cb_bcast_scaler, wt, bcast_scaler0, dst0);
        }
        reduce_uninit();
        recip_tile_init();
        recip_tile(dst0);  // DST[0] = 1/sum(exp(x))
        pack_tile(dst0, cb_recipsumexps);
        cb_push_back(cb_recipsumexps, 1);

        REL();

        cb_wait_front(cb_recipsumexps, 1);  // will reuse Wt times for bcast

        reconfig_data_format(cb_exps, cb_recipsumexps);
        pack_reconfig_data_format(cb_out0);
        // now cb_sumexps has exp tiles, need to multiply by our DST[2]
        // by now we already did a umulative wait for Wt tiles in cb_exps
        mul_bcast_cols_init_short(cb_exps, cb_recipsumexps);
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            ACQ();
            cb_reserve_back(cb_out0, ndst);
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                // wt+wt8 since we pop Wt after the entire loop
                mul_tiles_bcast<BroadcastType::COL>(
                    cb_exps, cb_recipsumexps, wt + wt8, 0, wt8);  // tile *= 1/(sum(exp(x)))
                pack_tile(wt8, cb_out0);
            }
            cb_push_back(cb_out0, ndst);
            REL();
        }
        cb_pop_front(cb_recipsumexps, 1);
        cb_pop_front(cb_exps, Wt);
    }  // NCHt loop
    // cb_pop_front(cb_bcast_scaler, 1); // we don't actually have to do this
    // cb_pop_front(cb_fused_scale, 1); // we don't actually have to do this
}
}  // namespace NAMESPACE
