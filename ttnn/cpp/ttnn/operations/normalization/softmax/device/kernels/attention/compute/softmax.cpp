// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch

template <uint32_t dfb_in, uint32_t dfb_max_scaler, uint32_t dfb_max, uint32_t dfb_out>
void calc_numeric_stable(uint32_t Wt, uint32_t ndst) {
    auto dfb_in_obj = DataflowBuffer(dfb_in);
    auto dfb_max_obj = DataflowBuffer(dfb_max);
    auto dfb_out_obj = DataflowBuffer(dfb_out);

    // calculate max val per row
    compute_kernel_lib::reduce<
        PoolType::MAX,
        ReduceDim::REDUCE_ROW,
        dfb_in,
        dfb_max_scaler,
        dfb_max,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));

    // calculate x-max(x)
    exp_tile_init<EXP_APPROX>();
    reconfig_data_format_srcb(dfb_max);
    dfb_max_obj.wait_front(1);
    sub_bcast_cols_init_short(dfb_in, dfb_max);
    for (uint32_t wt = 0; wt < Wt; wt += ndst) {
        const uint32_t rem = (wt + ndst > Wt) ? (Wt - wt) : ndst;  // clamped final block
        tile_regs_acquire();
        for (uint32_t wt8 = 0; wt8 < rem; wt8++) {
            sub_tiles_bcast_cols(dfb_in, dfb_max, wt + wt8, 0, wt8);
        }
        dfb_out_obj.reserve_back(rem);
        for (uint32_t wt8 = 0; wt8 < rem; wt8++) {
            exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t wt8 = 0; wt8 < rem; wt8++) {
            pack_tile(wt8, dfb_out);  // reuse the exps buffer again, this time in a circular manner
        }
        tile_regs_release();
        dfb_out_obj.push_back(rem);
    }
    dfb_in_obj.pop_front(Wt);
    dfb_max_obj.pop_front(1);
    dfb_out_obj.wait_front(Wt);
}

// CB consumers cannot wrap mid-fifo: pops in one cycle must land exactly on fifo_limit.
// After a partial row (Wt not a multiple of the CB capacity), rd/wr sit at that offset.
// Push/pop `pad` tiles to complete the cycle and return pointers to the CB base.
// Kept identical to the copy in softmax_large_tensor.cpp.
ALWI void cycle_dfb_pad(uint32_t dfb_id, uint32_t pad) {
    if (pad == 0) {
        return;
    }
    DataflowBuffer dfb(dfb_id);
    dfb.reserve_back(pad);
    dfb.push_back(pad);
    dfb.wait_front(pad);
    dfb.pop_front(pad);
}

// Same, for CBs whose padding tiles the reader already pushed: only consume them.
ALWI void drain_dfb_pad(uint32_t dfb_id, uint32_t pad) {
    if (pad == 0) {
        return;
    }
    DataflowBuffer dfb(dfb_id);
    dfb.wait_front(pad);
    dfb.pop_front(pad);
}

void kernel_main() {
    const uint32_t NCHt = get_arg_val<uint32_t>(0);
    const uint32_t Ht = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t ndst = get_arg_val<uint32_t>(3);
    const uint32_t start_ht = get_arg_val<uint32_t>(4);
    const uint32_t mask_padded_data = get_arg_val<uint32_t>(5);
    const uint32_t in0_t = get_arg_val<uint32_t>(6);  // factory: in0 CB tile capacity
    const uint32_t out0_t = ndst * 2;                 // matches factory out0_t = block_size * 2 (double-buffered)
    const uint32_t exps_t = ((Wt + ndst - 1) / ndst) * ndst;  // dfb_exps/dfb_x capacity, rounded to ndst
    // Tiles needed after Wt to finish each CB's cycle, named for the capacity they align.
    const uint32_t in0_pad = (in0_t > 0 && Wt > 0) ? ((in0_t - (Wt % in0_t)) % in0_t) : 0;
    const uint32_t out0_pad = (out0_t > 0 && Wt > 0) ? ((out0_t - (Wt % out0_t)) % out0_t) : 0;
    const uint32_t exps_pad = (exps_t > Wt) ? (exps_t - Wt) : 0;
    const uint32_t attn_pad = exps_pad;  // in4_t is also round_up(Wt, ndst); reader pushes the pad
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_6);

    constexpr uint32_t onetile = 1;
    // reserve one tile for zeros on cb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto dfb_max_scaler = tt::CBIndex::c_2;
    constexpr auto dfb_sum_scaler = tt::CBIndex::c_13;
    constexpr auto dfb_fused_scale = tt::CBIndex::c_3;
    constexpr auto dfb_fused_attn = tt::CBIndex::c_4;
    constexpr auto dfb_mask_padded = tt::CBIndex::c_5;
    constexpr auto dfb_exps = tt::CBIndex::c_6;
    constexpr auto dfb_scale_mask = tt::CBIndex::c_9;
    constexpr auto dfb_recipsumexps = tt::CBIndex::c_7;
    constexpr auto dfb_in0 = tt::CBIndex::c_0;
    constexpr auto dfb_out0 = tt::CBIndex::c_11;
    DataflowBuffer dfb_max_scaler_obj(dfb_max_scaler);
    DataflowBuffer dfb_sum_scaler_obj(dfb_sum_scaler);
    DataflowBuffer dfb_fused_scale_obj(dfb_fused_scale);
    DataflowBuffer dfb_fused_attn_obj(dfb_fused_attn);
    DataflowBuffer dfb_mask_padded_obj(dfb_mask_padded);
    DataflowBuffer dfb_exps_obj(dfb_exps);
    DataflowBuffer dfb_scale_mask_obj(dfb_scale_mask);
    DataflowBuffer dfb_recipsumexps_obj(dfb_recipsumexps);
    DataflowBuffer dfb_in0_obj(dfb_in0);
    DataflowBuffer dfb_out0_obj(dfb_out0);
#ifdef NUMERIC_STABLE
    constexpr auto dfb_max = tt::CBIndex::c_8;
    constexpr auto dfb_x = tt::CBIndex::c_10;
    DataflowBuffer dfb_max_obj(dfb_max);
#else
    constexpr auto dfb_x = dfb_exps;
#endif
    DataflowBuffer dfb_x_obj(dfb_x);

    dfb_max_scaler_obj.wait_front(1);  // comes from the reader
    dfb_sum_scaler_obj.wait_front(1);  // comes from the reader

#if FUSED_SCALE_MASK
    dfb_fused_scale_obj.wait_front(1);
#endif

    constexpr int dst0 = 0;
    uint32_t ht = start_ht;
    bool wait_mask = true;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#if FUSED_SCALE_MASK
        reconfig_data_format(dfb_in0, dfb_fused_scale);
        pack_reconfig_data_format(dfb_scale_mask);
        mul_tiles_bcast_scalar_init_short(dfb_in0, dfb_fused_scale);
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            const uint32_t rem = (wt + ndst > Wt) ? (Wt - wt) : ndst;  // clamped final block
            // apply fused scale [*= 1/sqrt(...)]
            tile_regs_acquire();
            dfb_in0_obj.wait_front(rem);
            dfb_scale_mask_obj.reserve_back(rem);
            for (uint32_t wt8 = 0; wt8 < rem; wt8++) {
                mul_tiles_bcast_scalar(dfb_in0, dfb_fused_scale, wt8, 0, wt8);  // mul bcast-HW -> DST[wt8]
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t wt8 = 0; wt8 < rem; wt8++) {
                pack_tile(wt8, dfb_scale_mask);  // reuse exps buffer
            }
            tile_regs_release();
            dfb_scale_mask_obj.push_back(rem);
            dfb_in0_obj.pop_front(rem);
        }
        reconfig_data_format(dfb_scale_mask, dfb_fused_attn);

#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif

#ifdef CAUSAL_MASK
        add_tiles_init(dfb_scale_mask, dfb_fused_attn);
#else
        add_bcast_rows_init_short(dfb_scale_mask, dfb_fused_attn);
#endif
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            const uint32_t rem = (wt + ndst > Wt) ? (Wt - wt) : ndst;  // clamped final block
            tile_regs_acquire();
            dfb_scale_mask_obj.wait_front(rem);
#ifdef CAUSAL_MASK
            dfb_fused_attn_obj.wait_front(wt + rem);  // cumulative wait for up to Wt tiles
            for (uint32_t wt8 = 0; wt8 < rem; wt8++) {
                add_tiles(dfb_scale_mask, dfb_fused_attn, wt8, wt + wt8, wt8);  // tile *= 1/(sum(exp(x)))
            }
#else
            if (wait_mask) {
                dfb_fused_attn_obj.wait_front(wt + rem);  // cumulative wait for up to Wt tiles, only at first ht
            }

            for (uint32_t wt8 = 0; wt8 < rem; wt8++) {
                add_tiles_bcast_rows(dfb_scale_mask, dfb_fused_attn, wt8, wt + wt8, wt8);  // tile *= 1/(sum(exp(x)))
            }
#endif
            dfb_scale_mask_obj.pop_front(rem);
            dfb_x_obj.reserve_back(rem);
#ifndef NUMERIC_STABLE
            for (uint32_t wt8 = 0; wt8 < rem; wt8++) {
                exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t wt8 = 0; wt8 < rem; wt8++) {
                pack_tile(wt8, dfb_x);  // reuse the exps buffer again, this time in a circular manner
            }
            tile_regs_release();
            dfb_x_obj.push_back(rem);
        }

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        calc_numeric_stable<dfb_x, dfb_max_scaler, dfb_max, dfb_exps>(Wt, ndst);
#endif

#ifdef CAUSAL_MASK
        dfb_fused_attn_obj.pop_front(Wt);
        drain_dfb_pad(dfb_fused_attn, attn_pad);
#else
        if (wait_mask) {
            wait_mask = false;
        }
        ht++;
        if (ht == Ht) {
            dfb_fused_attn_obj.pop_front(Wt);
            drain_dfb_pad(dfb_fused_attn, attn_pad);
            ht = 0;
            wait_mask = true;
        }
#endif  // CAUSAL_MASK

        reconfig_data_format(dfb_exps, dfb_sum_scaler);
#else
        reconfig_data_format(dfb_in0, dfb_in0);
        pack_reconfig_data_format(dfb_exps);
        copy_tile_to_dst_init_short(dfb_in0);  // need to copy from CB to DST to be able to run sfpu math
#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif
        if (mask_padded_data) {
            for (uint32_t wt = 0; wt < Wt; wt += ndst) {
                const uint32_t rem = (wt + ndst > Wt) ? (Wt - wt) : ndst;  // clamped final block
                tile_regs_acquire();
                dfb_in0_obj.wait_front(rem);
                for (uint32_t wt8 = 0; wt8 < rem; ++wt8) {
                    if (wt + wt8 == Wt - 1) {  // last tile of the row gets the -inf padding mask
                        reconfig_data_format(dfb_in0, dfb_mask_padded);
                        add_bcast_rows_init_short(dfb_in0, dfb_mask_padded);
                        dfb_mask_padded_obj.wait_front(1);
                        add_tiles_bcast_rows(dfb_in0, dfb_mask_padded, wt8, 0, wt8);
                    } else {
                        copy_tile(dfb_in0, wt8, wt8);  // copy from c_in[0] to DST[0]
                    }
                }
                dfb_in0_obj.pop_front(rem);

                dfb_x_obj.reserve_back(rem);
#ifndef NUMERIC_STABLE
                for (uint32_t wt8 = 0; wt8 < rem; ++wt8) {
                    exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
                }
#endif
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t wt8 = 0; wt8 < rem; ++wt8) {
                    pack_tile(wt8, dfb_x);  // DST[0]->dfb_id[wt]
                }
                tile_regs_release();
                dfb_x_obj.push_back(rem);
            }

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<dfb_x, dfb_max_scaler, dfb_max, dfb_exps>(Wt, ndst);
#endif

        } else {
// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<dfb_in0, dfb_max_scaler, dfb_max, dfb_exps>(Wt, ndst);
#else
            for (uint32_t wt = 0; wt < Wt; wt += ndst) {
                const uint32_t rem = (wt + ndst > Wt) ? (Wt - wt) : ndst;  // clamped final block
                tile_regs_acquire();
                dfb_in0_obj.wait_front(rem);
                for (uint32_t wt8 = 0; wt8 < rem; ++wt8) {
                    copy_tile(dfb_in0, wt8, wt8);  // copy from c_in[0] to DST[0]
                }
                dfb_in0_obj.pop_front(rem);

                dfb_exps_obj.reserve_back(rem);
                for (uint32_t wt8 = 0; wt8 < rem; ++wt8) {
                    exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t wt8 = 0; wt8 < rem; ++wt8) {
                    pack_tile(wt8, dfb_exps);  // DST[0]->dfb_id[wt]
                }
                tile_regs_release();
                dfb_exps_obj.push_back(rem);
            }
#endif
        }
#endif

        // SUM reduce with reciprocal post-processing (1/sum)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            dfb_exps,
            dfb_sum_scaler,
            dfb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            compute_kernel_lib::ReduceInputBlockShape::row(Wt),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t) {
                recip_tile_init();
                recip_tile(0);
            });

        dfb_recipsumexps_obj.wait_front(1);  // will reuse Wt times for bcast

        reconfig_data_format(dfb_exps, dfb_recipsumexps);
        pack_reconfig_data_format(dfb_out0);
        // now cb_sumexps has exp tiles, need to multiply by our DST[2]
        // by now we already did a cumulative wait for Wt tiles in cb_exps
        mul_bcast_cols_init_short(dfb_exps, dfb_recipsumexps);
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            const uint32_t rem = (wt + ndst > Wt) ? (Wt - wt) : ndst;  // clamped final block
            tile_regs_acquire();
            dfb_out0_obj.reserve_back(rem);
            for (uint32_t wt8 = 0; wt8 < rem; wt8++) {
                // wt+wt8 since we pop Wt after the entire loop
                mul_tiles_bcast<BroadcastType::COL>(
                    dfb_exps, dfb_recipsumexps, wt + wt8, 0, wt8);  // tile *= 1/(sum(exp(x)))
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t wt8 = 0; wt8 < rem; wt8++) {
                pack_tile(wt8, dfb_out0);
            }
            tile_regs_release();
            dfb_out0_obj.push_back(rem);
        }
        dfb_recipsumexps_obj.pop_front(1);
        dfb_exps_obj.pop_front(Wt);

        // Realign CBs before the next row when Wt does not fill them exactly.
        drain_dfb_pad(dfb_in0, in0_pad);
        cycle_dfb_pad(dfb_exps, exps_pad);
#if FUSED_SCALE_MASK
        cycle_dfb_pad(dfb_scale_mask, exps_pad + ndst);  // im3_t = exps_t + ndst
#ifdef NUMERIC_STABLE
        // Without NUMERIC_STABLE, dfb_x aliases dfb_exps; cycling it again would drift it per row.
        cycle_dfb_pad(dfb_x, exps_pad);
#endif
#elif defined(NUMERIC_STABLE)
        // Without NUMERIC_STABLE, dfb_x aliases dfb_exps (already cycled).
        if (mask_padded_data) {
            cycle_dfb_pad(dfb_x, exps_pad);
        }
#endif
        if (out0_pad > 0) {
            dfb_out0_obj.reserve_back(out0_pad);
            dfb_out0_obj.push_back(out0_pad);  // writer drains, does not write to DRAM
        }
    }  // NCHt loop
    // The scaler tiles are each waited once and reused across the whole NCHt loop; pop them at
    // the end so the CBs are left balanced.
    dfb_max_scaler_obj.pop_front(1);
    dfb_sum_scaler_obj.pop_front(1);
#if FUSED_SCALE_MASK
    dfb_fused_scale_obj.pop_front(1);
#endif
}
