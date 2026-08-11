// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
#ifdef REDUCE_POST_MUL
    // Packed fp32 user scalar applied via mul_unary_tile after the reduce+negate finishes.
    constexpr uint32_t post_mul_scaler_bits = get_compile_time_arg_val(3);
#endif

    constexpr uint32_t dfb_input = tt::CBIndex::c_0;
    constexpr uint32_t dfb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t dfb_output = tt::CBIndex::c_3;
    constexpr uint32_t dfb_acc = tt::CBIndex::c_4;
    constexpr uint32_t dfb_ineg = tt::CBIndex::c_5;

    DataflowBuffer dfb_input_obj(dfb_input);
    DataflowBuffer dfb_scaler_obj(dfb_scaler);
    DataflowBuffer dfb_output_obj(dfb_output);
    DataflowBuffer dfb_acc_obj(dfb_acc);
    DataflowBuffer dfb_ineg_obj(dfb_ineg);

    compute_kernel_hw_startup(dfb_input, dfb_scaler, dfb_output);
    dfb_scaler_obj.wait_front(1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                dfb_input_obj.wait_front(onetile);

                tile_regs_acquire();
                copy_tile_init(dfb_input);
                copy_tile(dfb_input, 0, reduce_dst_idx);
                negative_tile_init();
                negative_tile(reduce_dst_idx);
                tile_regs_commit();

                dfb_input_obj.pop_front(onetile);

                dfb_ineg_obj.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(reduce_dst_idx, dfb_ineg);
                tile_regs_release();

                dfb_ineg_obj.push_back(onetile);

                if (wt > 0 || ht > 0) {
                    dfb_acc_obj.wait_front(onetile);
                }
                dfb_ineg_obj.wait_front(onetile);

                tile_regs_acquire();
                if (wt > 0 || ht > 0) {
                    copy_tile_init(dfb_acc);
                    copy_tile(dfb_acc, 0, reduce_dst_idx);
                }
                constexpr bool swap_operands = (REDUCE_DIM == ReduceDim::REDUCE_ROW) && (REDUCE_OP != PoolType::MAX);
                if constexpr (swap_operands) {
                    reconfig_data_format(dfb_scaler, dfb_ineg);
                }
                reduce_init<REDUCE_OP, REDUCE_DIM>(dfb_ineg, dfb_scaler, dfb_acc);
                reduce_tile<REDUCE_OP, REDUCE_DIM>(dfb_ineg, dfb_scaler, 0, 0, reduce_dst_idx);
                reduce_uninit();
                tile_regs_commit();

                dfb_ineg_obj.pop_front(onetile);
                if (wt > 0 || ht > 0) {
                    dfb_acc_obj.pop_front(onetile);
                }

                dfb_acc_obj.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(reduce_dst_idx, dfb_acc);
                tile_regs_release();

                dfb_acc_obj.push_back(onetile);
            }  // wt
        }  // ht

        dfb_acc_obj.wait_front(onetile);

        tile_regs_acquire();
        copy_tile_init(dfb_acc);
        copy_tile(dfb_acc, 0, reduce_dst_idx);
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

        dfb_acc_obj.pop_front(onetile);

        dfb_output_obj.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(reduce_dst_idx, dfb_output);
        tile_regs_release();

        dfb_output_obj.push_back(onetile);
    }  // nc
}
