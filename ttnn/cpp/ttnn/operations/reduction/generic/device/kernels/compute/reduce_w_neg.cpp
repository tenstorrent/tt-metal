// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reduce W with negation: per-tile negate -> reduce_tile -> accumulate -> negate.
// Ported to Metal 2.0 (named DFB bindings, named args).
//
// Host bindings expected:
//   compile_time_arg_bindings: { {"Ht", ...}, {"Wt", ...}, {"NC", ...},
//                                {"post_mul_scaler_bits", ...} (only if REDUCE_POST_MUL) }
//   dfb_bindings:
//     { INPUT (CONSUMER, name="input"),
//       SCALER (CONSUMER, name="scaler"),
//       OUTPUT (PRODUCER, name="output"),
//       ACC self-loop (PRODUCER + CONSUMER, name="acc"),
//       INEG self-loop (PRODUCER + CONSUMER, name="ineg") }

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
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t NC = get_arg(args::NC);
#ifdef REDUCE_POST_MUL
    // Packed fp32 user scalar applied via mul_unary_tile after the reduce+negate finishes.
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
        int dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_input_obj.wait_front(onetile);
                tile_regs_acquire();
                copy_tile_init(dfb::input);
                copy_tile(dfb::input, 0, dst_idx);
                negative_tile_init();
                negative_tile(dst_idx);
                tile_regs_wait();
                cb_input_obj.pop_front(onetile);
                cb_ineg_obj.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, dfb::ineg);
                tile_regs_release();
                cb_ineg_obj.push_back(onetile);

                tile_regs_acquire();
                if (wt > 0) {
                    cb_acc_obj.wait_front(onetile);
                    copy_tile_init(dfb::acc);
                    copy_tile(dfb::acc, 0, dst_idx);
                }

                cb_ineg_obj.wait_front(onetile);
                reduce_init<REDUCE_OP, REDUCE_DIM>(dfb::ineg, dfb::scaler, dfb::acc);
                reduce_tile<REDUCE_OP, REDUCE_DIM>(dfb::ineg, dfb::scaler, 0, 0, dst_idx);
                reduce_uninit();
                tile_regs_wait();
                cb_ineg_obj.pop_front(onetile);
                if (wt > 0) {
                    cb_acc_obj.pop_front(onetile);
                }
                cb_acc_obj.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, dfb::acc);
                tile_regs_release();
                cb_acc_obj.push_back(onetile);
            }  // wt

            cb_acc_obj.wait_front(onetile);
            tile_regs_acquire();
            copy_tile_init(dfb::acc);
            copy_tile(dfb::acc, 0, dst_idx);
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
            pack_tile(dst_idx, dfb::output);
            tile_regs_release();
            cb_output_obj.push_back(onetile);
        }  // ht
    }  // nc
}
