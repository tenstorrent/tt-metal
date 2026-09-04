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
#include "experimental/kernel_args.h"

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch

template <std::uint32_t dfb_in, std::uint32_t dfb_max_scaler, std::uint32_t dfb_max, std::uint32_t dfb_out>
void calc_numeric_stable(std::uint32_t Wt, std::uint32_t ndst) {
    DataflowBuffer dfb_in_obj(dfb_in);
    DataflowBuffer dfb_max_obj(dfb_max);
    DataflowBuffer dfb_out_obj(dfb_out);

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
    sub_bcast_cols_init(dfb_in, dfb_max);
    for (std::uint32_t wt = 0; wt < Wt; wt += ndst) {
        const std::uint32_t rem = (wt + ndst > Wt) ? (Wt - wt) : ndst;  // clamped final block
        tile_regs_acquire();
        for (std::uint32_t wt8 = 0; wt8 < rem; wt8++) {
            sub_tiles_bcast_cols(dfb_in, dfb_max, wt + wt8, 0, wt8);
        }
        dfb_out_obj.reserve_back(rem);
        for (std::uint32_t wt8 = 0; wt8 < rem; wt8++) {
            exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
        }
        tile_regs_commit();
        tile_regs_wait();
        for (std::uint32_t wt8 = 0; wt8 < rem; wt8++) {
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
ALWI void cycle_dfb_pad(std::uint32_t dfb_id, std::uint32_t pad) {
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
ALWI void drain_dfb_pad(std::uint32_t dfb_id, std::uint32_t pad) {
    if (pad == 0) {
        return;
    }
    DataflowBuffer dfb(dfb_id);
    dfb.wait_front(pad);
    dfb.pop_front(pad);
}

void kernel_main() {
    const std::uint32_t NCHt = get_arg(args::num_rows);
    const std::uint32_t Ht = get_arg(args::Ht);
    const std::uint32_t Wt = get_arg(args::Wt);
    const std::uint32_t ndst = get_arg(args::blk);
    const std::uint32_t start_ht = get_arg(args::start_ht);
    // The pad-mask data (W > W_unpadded) is host-known; it is carried as the MASK_PADDED_DATA compile-time
    // define (not a runtime arg), which lets the c_5 pad-mask DFB and the c_10 intermediate be bound only
    // on the paths that use them.
    constexpr std::uint32_t in0_t = get_arg(args::in0_t);          // in0 DFB tile capacity
    const std::uint32_t out0_t = ndst * 2;                         // matches factory out0_t = block_size * 2
    const std::uint32_t exps_t = ((Wt + ndst - 1) / ndst) * ndst;  // dfb_exps/dfb_x capacity, rounded to ndst
    // Every CB capacity is a multiple of ndst, so when ndst divides Wt the blocks tile each fifo
    // exactly: a row already ends on the base and no pad is needed. Zero guards keep the modulo safe.
    const bool pad_to_fifo_base = Wt > 0 && ndst > 0 && (Wt % ndst) != 0;
    // Tiles needed after Wt to finish each CB's cycle, named for the capacity they align.
    const std::uint32_t in0_pad = pad_to_fifo_base ? ((in0_t - (Wt % in0_t)) % in0_t) : 0;
    const std::uint32_t out0_pad = pad_to_fifo_base ? ((out0_t - (Wt % out0_t)) % out0_t) : 0;
    const std::uint32_t exps_pad = pad_to_fifo_base ? (exps_t - Wt) : 0;
    const std::uint32_t attn_pad = exps_pad;  // in4_t is also round_up(Wt, ndst); reader pushes the pad
    const std::uint32_t scale_mask_pad = pad_to_fifo_base ? (exps_pad + ndst) : 0;  // im3_t = exps_t + ndst

    constexpr std::uint32_t onetile = 1;
    // reserve one tile for zeros on dfb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto dfb_max_scaler = dfb::max_scaler;
    constexpr auto dfb_sum_scaler = dfb::sum_scaler;
    constexpr auto dfb_exps = dfb::exps;
    constexpr auto dfb_recipsumexps = dfb::recip_sum_exps;
    constexpr auto dfb_in0 = dfb::in0;
    constexpr auto dfb_out0 = dfb::out0;
    DataflowBuffer dfb_max_scaler_obj(dfb_max_scaler);
    DataflowBuffer dfb_sum_scaler_obj(dfb_sum_scaler);
    DataflowBuffer dfb_exps_obj(dfb_exps);
    DataflowBuffer dfb_recipsumexps_obj(dfb_recipsumexps);
    DataflowBuffer dfb_in0_obj(dfb_in0);
    DataflowBuffer dfb_out0_obj(dfb_out0);
#if FUSED_SCALE_MASK
    // fused_scale/fused_attn/scale_mask are bound only on the fused scale-mask path.
    constexpr auto dfb_fused_scale = dfb::fused_scale;
    constexpr auto dfb_fused_attn = dfb::fused_attn;
    constexpr auto dfb_scale_mask = dfb::scale_mask;
    DataflowBuffer dfb_fused_scale_obj(dfb_fused_scale);
    DataflowBuffer dfb_fused_attn_obj(dfb_fused_attn);
    DataflowBuffer dfb_scale_mask_obj(dfb_scale_mask);
#endif
#ifdef MASK_PADDED_DATA
    constexpr auto dfb_mask_padded = dfb::mask_padded;
    DataflowBuffer dfb_mask_padded_obj(dfb_mask_padded);
#endif

    compute_kernel_hw_startup(dfb_in0, dfb_max_scaler, dfb_exps);
#ifdef NUMERIC_STABLE
    constexpr auto dfb_max = dfb::max;
#if defined(FUSED_SCALE_MASK) || defined(MASK_PADDED_DATA)
    // dfb_x is a distinct intermediate (c_10) only on the numeric-stable paths that post-process a masked
    // buffer; otherwise the reads go straight from dfb_in0 (see the calc_numeric_stable<dfb_in0,...> call).
    constexpr auto dfb_x = dfb::x;
    DataflowBuffer dfb_x_obj(dfb_x);
#endif
#else
    // Without numeric_stable, dfb_x aliases dfb_exps (Same-FIFO reuse) so exp results circulate in one buffer.
    constexpr auto dfb_x = dfb_exps;
    DataflowBuffer dfb_x_obj(dfb_x);
#endif

    dfb_max_scaler_obj.wait_front(1);  // comes from the reader
    dfb_sum_scaler_obj.wait_front(1);  // comes from the reader

#if FUSED_SCALE_MASK
    dfb_fused_scale_obj.wait_front(1);
#endif

    constexpr int dst0 = 0;
    std::uint32_t ht = start_ht;
    bool wait_mask = true;
    for (std::uint32_t ncht = 0; ncht < NCHt; ncht++) {
#if FUSED_SCALE_MASK
        reconfig_data_format(dfb_in0, dfb_fused_scale);
        pack_reconfig_data_format(dfb_scale_mask);
        mul_bcast_scalar_init(dfb_in0, dfb_fused_scale);
        for (std::uint32_t wt = 0; wt < Wt; wt += ndst) {
            const std::uint32_t rem = (wt + ndst > Wt) ? (Wt - wt) : ndst;  // clamped final block
            // apply fused scale [*= 1/sqrt(...)]
            tile_regs_acquire();
            dfb_in0_obj.wait_front(rem);
            dfb_scale_mask_obj.reserve_back(rem);
            for (std::uint32_t wt8 = 0; wt8 < rem; wt8++) {
                mul_tiles_bcast_scalar(dfb_in0, dfb_fused_scale, wt8, 0, wt8);  // mul bcast-HW -> DST[wt8]
            }
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t wt8 = 0; wt8 < rem; wt8++) {
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
        add_init(dfb_scale_mask, dfb_fused_attn);
#else
        add_bcast_rows_init(dfb_scale_mask, dfb_fused_attn);
#endif
        for (std::uint32_t wt = 0; wt < Wt; wt += ndst) {
            const std::uint32_t rem = (wt + ndst > Wt) ? (Wt - wt) : ndst;  // clamped final block
            tile_regs_acquire();
            dfb_scale_mask_obj.wait_front(rem);
#ifdef CAUSAL_MASK
            dfb_fused_attn_obj.wait_front(wt + rem);  // cumulative wait for up to Wt tiles
            for (std::uint32_t wt8 = 0; wt8 < rem; wt8++) {
                add_tiles(dfb_scale_mask, dfb_fused_attn, wt8, wt + wt8, wt8);  // tile *= 1/(sum(exp(x)))
            }
#else
            if (wait_mask) {
                dfb_fused_attn_obj.wait_front(wt + rem);  // cumulative wait for up to Wt tiles, only at first ht
            }

            for (std::uint32_t wt8 = 0; wt8 < rem; wt8++) {
                add_tiles_bcast_rows(dfb_scale_mask, dfb_fused_attn, wt8, wt + wt8, wt8);  // tile *= 1/(sum(exp(x)))
            }
#endif
            dfb_scale_mask_obj.pop_front(rem);
            dfb_x_obj.reserve_back(rem);
#ifndef NUMERIC_STABLE
            for (std::uint32_t wt8 = 0; wt8 < rem; wt8++) {
                exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t wt8 = 0; wt8 < rem; wt8++) {
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
        copy_init(dfb_in0);  // need to copy from CB to DST to be able to run sfpu math
#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif
#ifdef MASK_PADDED_DATA
        {
            for (std::uint32_t wt = 0; wt < Wt; wt += ndst) {
                const std::uint32_t rem = (wt + ndst > Wt) ? (Wt - wt) : ndst;  // clamped final block
                tile_regs_acquire();
                dfb_in0_obj.wait_front(rem);
                for (std::uint32_t wt8 = 0; wt8 < rem; ++wt8) {
                    if (wt + wt8 == Wt - 1) {  // last tile of the row gets the -inf padding mask
                        reconfig_data_format(dfb_in0, dfb_mask_padded);
                        add_bcast_rows_init(dfb_in0, dfb_mask_padded);
                        dfb_mask_padded_obj.wait_front(1);
                        add_tiles_bcast_rows(dfb_in0, dfb_mask_padded, wt8, 0, wt8);
                    } else {
                        copy_tile(dfb_in0, wt8, wt8);  // copy from c_in[0] to DST[0]
                    }
                }
                dfb_in0_obj.pop_front(rem);

                dfb_x_obj.reserve_back(rem);
#ifndef NUMERIC_STABLE
                for (std::uint32_t wt8 = 0; wt8 < rem; ++wt8) {
                    exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
                }
#endif
                tile_regs_commit();
                tile_regs_wait();
                for (std::uint32_t wt8 = 0; wt8 < rem; ++wt8) {
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
        }
#else
        {
// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<dfb_in0, dfb_max_scaler, dfb_max, dfb_exps>(Wt, ndst);
#else
            for (std::uint32_t wt = 0; wt < Wt; wt += ndst) {
                const std::uint32_t rem = (wt + ndst > Wt) ? (Wt - wt) : ndst;  // clamped final block
                tile_regs_acquire();
                dfb_in0_obj.wait_front(rem);
                for (std::uint32_t wt8 = 0; wt8 < rem; ++wt8) {
                    copy_tile(dfb_in0, wt8, wt8);  // copy from c_in[0] to DST[0]
                }
                dfb_in0_obj.pop_front(rem);

                dfb_exps_obj.reserve_back(rem);
                for (std::uint32_t wt8 = 0; wt8 < rem; ++wt8) {
                    exp_tile<EXP_APPROX>(wt8);  // exp on DST[0]
                }
                tile_regs_commit();
                tile_regs_wait();
                for (std::uint32_t wt8 = 0; wt8 < rem; ++wt8) {
                    pack_tile(wt8, dfb_exps);  // DST[0]->dfb_id[wt]
                }
                tile_regs_release();
                dfb_exps_obj.push_back(rem);
            }
#endif
        }
#endif  // MASK_PADDED_DATA
#endif  // FUSED_SCALE_MASK

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
            [](std::uint32_t) {
                recip_tile_init();
                recip_tile(0);
            });

        dfb_recipsumexps_obj.wait_front(1);  // will reuse Wt times for bcast

        reconfig_data_format(dfb_exps, dfb_recipsumexps);
        pack_reconfig_data_format(dfb_out0);
        // now dfb_sumexps has exp tiles, need to multiply by our DST[2]
        // by now we already did a cumulative wait for Wt tiles in dfb_exps
        mul_bcast_cols_init(dfb_exps, dfb_recipsumexps);
        for (std::uint32_t wt = 0; wt < Wt; wt += ndst) {
            const std::uint32_t rem = (wt + ndst > Wt) ? (Wt - wt) : ndst;  // clamped final block
            tile_regs_acquire();
            dfb_out0_obj.reserve_back(rem);
            for (std::uint32_t wt8 = 0; wt8 < rem; wt8++) {
                // wt+wt8 since we pop Wt after the entire loop
                mul_tiles_bcast<BroadcastType::COL>(
                    dfb_exps, dfb_recipsumexps, wt + wt8, 0, wt8);  // tile *= 1/(sum(exp(x)))
            }
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t wt8 = 0; wt8 < rem; wt8++) {
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
        cycle_dfb_pad(dfb_scale_mask, scale_mask_pad);
#ifdef NUMERIC_STABLE
        // Without NUMERIC_STABLE, dfb_x aliases dfb_exps; cycling it again would drift it per row.
        cycle_dfb_pad(dfb_x, exps_pad);
#endif
#elif defined(NUMERIC_STABLE) && defined(MASK_PADDED_DATA)
        // dfb_x is a distinct buffer only here; without NUMERIC_STABLE it aliases dfb_exps (already cycled).
        cycle_dfb_pad(dfb_x, exps_pad);
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
