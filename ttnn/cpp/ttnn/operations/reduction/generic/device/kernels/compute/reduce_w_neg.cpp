// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
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
    uint32_t Ht = get_arg(args::Ht);
    uint32_t Wt = get_arg(args::Wt);
    uint32_t NC = get_arg(args::NC);
#ifdef REDUCE_POST_MUL
    // Packed fp32 user scalar applied via mul_unary_tile after the reduce+negate finishes.
    constexpr auto post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
#endif

    constexpr uint32_t onetile = 1;

// The FPU path below is the only user of the acc / ineg scratch buffers, and the host binds them
// only when it selects this path (REDUCE_FPU_NEGATE). An `if constexpr` could not gate the
// dfb::acc / dfb::ineg *names*: name lookup happens before the condition is evaluated, so the gate
// has to be a preprocessor one.
#ifdef REDUCE_FPU_NEGATE
    DataflowBuffer dfb_input(dfb::in0);
    DataflowBuffer dfb_scaler(dfb::scaler);
    DataflowBuffer dfb_output(dfb::out);
    DataflowBuffer dfb_acc(dfb::acc);
    DataflowBuffer dfb_ineg(dfb::ineg);

    compute_kernel_hw_startup(dfb::in0, dfb::scaler, dfb::out);

    dfb_scaler.wait_front(1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        int dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                dfb_input.wait_front(onetile);
                tile_regs_acquire();
                copy_init(dfb::in0);
                copy_tile(dfb::in0, 0, dst_idx);
                negative_tile_init();
                negative_tile(dst_idx);
                tile_regs_wait();
                dfb_input.pop_front(onetile);
                dfb_ineg.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, dfb::ineg);
                tile_regs_release();
                dfb_ineg.push_back(onetile);

                tile_regs_acquire();
                if (wt > 0) {
                    dfb_acc.wait_front(onetile);
                    copy_init(dfb::acc);
                    copy_tile(dfb::acc, 0, dst_idx);
                }

                dfb_ineg.wait_front(onetile);
                constexpr bool swap_operands = (REDUCE_DIM == ReduceDim::REDUCE_ROW) && (REDUCE_OP != PoolType::MAX);
                if constexpr (swap_operands) {
                    reconfig_data_format(dfb::scaler, dfb::ineg);
                }
                reduce_init<REDUCE_OP, REDUCE_DIM>(dfb::ineg, dfb::scaler, dfb::acc);
                reduce_tile<REDUCE_OP, REDUCE_DIM>(dfb::ineg, dfb::scaler, 0, 0, dst_idx);
                reduce_uninit();
                tile_regs_wait();
                dfb_ineg.pop_front(onetile);
                if (wt > 0) {
                    dfb_acc.pop_front(onetile);
                }
                dfb_acc.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, dfb::acc);
                tile_regs_release();
                dfb_acc.push_back(onetile);
            }  // wt

            dfb_acc.wait_front(onetile);
            tile_regs_acquire();
            copy_init(dfb::acc);
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
            dfb_acc.pop_front(onetile);
            dfb_output.reserve_back(onetile);
            tile_regs_commit();
            pack_tile(dst_idx, dfb::out);
            tile_regs_release();
            dfb_output.push_back(onetile);
        }  // ht
    }  // nc
    // The scaler tile is waited once and reused for the whole reduction; pop it at the
    // end so the buffer is left balanced.
    dfb_scaler.pop_front(onetile);
#endif  // REDUCE_FPU_NEGATE
}
