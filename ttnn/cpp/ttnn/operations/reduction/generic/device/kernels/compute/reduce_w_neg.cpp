// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "llk_math_eltwise_binary.h"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto NC = get_arg(args::NC);
#ifdef REDUCE_POST_MUL
    // Packed fp32 user scalar applied via mul_unary_tile after the reduce+negate finishes.
    constexpr auto post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
#endif

    DataflowBuffer cb_input_obj(dfb::in_dfb);
    DataflowBuffer cb_scaler_obj(dfb::scaler_dfb);
    DataflowBuffer cb_output_obj(dfb::out_dfb);
    DataflowBuffer cb_acc_obj(dfb::acc_dfb);
    DataflowBuffer cb_ineg_obj(dfb::ineg_dfb);

    compute_kernel_hw_startup(dfb::in_dfb, dfb::scaler_dfb, dfb::out_dfb);

    cb_scaler_obj.wait_front(1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_input_obj.wait_front(onetile);
                tile_regs_acquire();
                copy_tile_init(dfb::in_dfb);
                copy_tile(dfb::in_dfb, 0, dst_idx);
                negative_tile_init();
                negative_tile(dst_idx);
                tile_regs_wait();
                cb_input_obj.pop_front(onetile);
                cb_ineg_obj.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, dfb::ineg_dfb);
                tile_regs_release();
                cb_ineg_obj.push_back(onetile);

                tile_regs_acquire();
                if (wt > 0) {
                    cb_acc_obj.wait_front(onetile);
                    copy_tile_init(dfb::acc_dfb);
                    copy_tile(dfb::acc_dfb, 0, dst_idx);
                }

                cb_ineg_obj.wait_front(onetile);
                reduce_init<REDUCE_OP, REDUCE_DIM>(dfb::ineg_dfb, dfb::scaler_dfb, dfb::acc_dfb);
                reduce_tile<REDUCE_OP, REDUCE_DIM>(dfb::ineg_dfb, dfb::scaler_dfb, 0, 0, dst_idx);
                reduce_uninit();
                tile_regs_wait();
                cb_ineg_obj.pop_front(onetile);
                if (wt > 0) {
                    cb_acc_obj.pop_front(onetile);
                }
                cb_acc_obj.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, dfb::acc_dfb);
                tile_regs_release();
                cb_acc_obj.push_back(onetile);
            }  // wt

            cb_acc_obj.wait_front(onetile);
            tile_regs_acquire();
            copy_tile_init(dfb::acc_dfb);
            copy_tile(dfb::acc_dfb, 0, dst_idx);
            negative_tile_init();
            negative_tile(dst_idx);
#ifdef REDUCE_POST_MUL
            // GMPOOL only respects the scaler's exponent for MAX/MIN, so the host requests reduction
            // with scaler=1.0 and then applies the user scalar via mul_unary_tile (SFPU) on each
            // output DEST register.
            binop_with_scalar_tile_init();
            mul_unary_tile(dst_idx, post_mul_scaler_bits);
#endif
            tile_regs_wait();
            cb_acc_obj.pop_front(onetile);
            cb_output_obj.reserve_back(onetile);
            tile_regs_commit();
            pack_tile(dst_idx, dfb::out_dfb);
            tile_regs_release();
            cb_output_obj.push_back(onetile);
        }  // ht
    }  // nc
}
