// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reduce HW with negation, ported to Metal 2.0.

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t NC = get_arg(args::NC);
#ifdef REDUCE_POST_MUL
    constexpr uint32_t post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
#endif

    DataflowBuffer cb_input_obj(dfb::input);
    DataflowBuffer cb_scaler_obj(dfb::scaler);
    DataflowBuffer cb_output_obj(dfb::output);
    DataflowBuffer cb_acc_obj(dfb::acc);
    DataflowBuffer cb_ineg_obj(dfb::ineg);

    compute_kernel_hw_startup(dfb::input, dfb::scaler, dfb::output);
    cb_scaler_obj.wait_front(1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                acquire_dst();
                cb_input_obj.wait_front(onetile);
                copy_tile_init(dfb::input);
                copy_tile(dfb::input, 0, reduce_dst_idx);
                negative_tile_init();
                negative_tile(reduce_dst_idx);
                cb_input_obj.pop_front(onetile);
                cb_ineg_obj.reserve_back(onetile);
                pack_tile(reduce_dst_idx, dfb::ineg);
                cb_ineg_obj.push_back(onetile);
                release_dst();

                acquire_dst();
                if (wt > 0 || ht > 0) {
                    cb_acc_obj.wait_front(onetile);
                    copy_tile_init(dfb::acc);
                    copy_tile(dfb::acc, 0, reduce_dst_idx);
                }

                cb_ineg_obj.wait_front(onetile);
                reduce_init<REDUCE_OP, REDUCE_DIM>(dfb::ineg, dfb::scaler, dfb::acc);
                reduce_tile<REDUCE_OP, REDUCE_DIM>(dfb::ineg, dfb::scaler, 0, 0, reduce_dst_idx);
                reduce_uninit();
                cb_ineg_obj.pop_front(onetile);
                if (wt > 0 || ht > 0) {
                    cb_acc_obj.pop_front(onetile);
                }
                cb_acc_obj.reserve_back(onetile);
                pack_tile(reduce_dst_idx, dfb::acc);
                cb_acc_obj.push_back(onetile);
                release_dst();
            }  // wt
        }  // ht

        acquire_dst();
        cb_acc_obj.wait_front(onetile);
        copy_tile_init(dfb::acc);
        copy_tile(dfb::acc, 0, reduce_dst_idx);
        negative_tile_init();
        negative_tile(reduce_dst_idx);
#ifdef REDUCE_POST_MUL
        // GMPOOL only respects the scaler's exponent for MAX/MIN, so the host requests reduction
        // with scaler=1.0 and then applies the user scalar via mul_unary_tile (SFPU) on each
        // output DEST register.
        binop_with_scalar_tile_init();
        mul_unary_tile(reduce_dst_idx, post_mul_scaler_bits);
#endif
        cb_acc_obj.pop_front(onetile);
        cb_output_obj.reserve_back(onetile);
        pack_tile(reduce_dst_idx, dfb::output);
        cb_output_obj.push_back(onetile);
        release_dst();
    }  // nc
}
