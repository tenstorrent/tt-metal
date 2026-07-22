// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reduce_hw_neg.cpp (negate / MIN HW reduce). CB indices → dfb:: bindings, CTAs →
// named args. The intermediate `ineg` and accumulator `acc` DFBs are bound by THIS compute kernel as
// both producer and consumer (self-loop); the factory declares them INTRA (see pool_generic precedent).
// The legacy reduce_hw_neg.cpp is retained for not-yet-ported reduce factories.

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    uint32_t Ht = get_arg(args::Ht);
    uint32_t Wt = get_arg(args::Wt);
    uint32_t NC = get_arg(args::NC);
#ifdef REDUCE_POST_MUL
    // Packed fp32 user scalar applied via mul_unary_tile after the reduce+negate finishes.
    constexpr uint32_t post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
#endif

    constexpr auto cb_input = dfb::in;
    constexpr auto cb_scaler = dfb::scaler;
    constexpr auto cb_output = dfb::out;
    constexpr auto cb_acc = dfb::acc;
    constexpr auto cb_ineg = dfb::ineg;

    DataflowBuffer cb_input_obj(cb_input);
    DataflowBuffer cb_scaler_obj(cb_scaler);
    DataflowBuffer cb_output_obj(cb_output);
    DataflowBuffer cb_acc_obj(cb_acc);
    DataflowBuffer cb_ineg_obj(cb_ineg);

    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);
    cb_scaler_obj.wait_front(1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_input_obj.wait_front(onetile);

                tile_regs_acquire();
                copy_init(cb_input);
                copy_tile(cb_input, 0, reduce_dst_idx);
                negative_tile_init();
                negative_tile(reduce_dst_idx);
                tile_regs_commit();

                cb_input_obj.pop_front(onetile);

                cb_ineg_obj.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(reduce_dst_idx, cb_ineg);
                tile_regs_release();

                cb_ineg_obj.push_back(onetile);

                if (wt > 0 || ht > 0) {
                    cb_acc_obj.wait_front(onetile);
                }
                cb_ineg_obj.wait_front(onetile);

                tile_regs_acquire();
                if (wt > 0 || ht > 0) {
                    copy_init(cb_acc);
                    copy_tile(cb_acc, 0, reduce_dst_idx);
                }
                reduce_init<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, cb_acc);
                reduce_tile<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, 0, 0, reduce_dst_idx);
                reduce_uninit();
                tile_regs_commit();

                cb_ineg_obj.pop_front(onetile);
                if (wt > 0 || ht > 0) {
                    cb_acc_obj.pop_front(onetile);
                }

                cb_acc_obj.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(reduce_dst_idx, cb_acc);
                tile_regs_release();

                cb_acc_obj.push_back(onetile);
            }  // wt
        }  // ht

        cb_acc_obj.wait_front(onetile);

        tile_regs_acquire();
        copy_init(cb_acc);
        copy_tile(cb_acc, 0, reduce_dst_idx);
        negative_tile_init();
        negative_tile(reduce_dst_idx);
#ifdef REDUCE_POST_MUL
        // GMPOOL only respects the scaler's exponent for MAX/MIN, so the host requests reduction
        // with scaler=1.0 and then applies the user scalar via mul_unary_tile (SFPU) on each
        // output DEST register.
        binop_with_scalar_tile_init();
        mul_unary_tile(reduce_dst_idx, post_mul_scaler_bits);
#endif
        tile_regs_commit();

        cb_acc_obj.pop_front(onetile);

        cb_output_obj.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(reduce_dst_idx, cb_output);
        tile_regs_release();

        cb_output_obj.push_back(onetile);
    }  // nc
}
