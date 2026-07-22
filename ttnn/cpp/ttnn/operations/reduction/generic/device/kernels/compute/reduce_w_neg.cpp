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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#include "llk_math_eltwise_binary.h"

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

    // Circular buffers:
    constexpr uint32_t dfb_input = tt::CBIndex::c_0;
    constexpr uint32_t dfb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t dfb_output = tt::CBIndex::c_3;
    constexpr uint32_t onetile = 1;
    constexpr DataFormat reduce_format = static_cast<DataFormat>(unpack_src_format[dfb_input]);

    if constexpr (is_sfpu_reduce_path<REDUCE_OP, REDUCE_DIM, reduce_format>()) {
        constexpr uint32_t acc_dst = 0;
        constexpr uint32_t work_dst = 1;

        DataflowBuffer dfb_input_obj(dfb_input);
        DataflowBuffer dfb_scaler_obj(dfb_scaler);
        DataflowBuffer dfb_output_obj(dfb_output);

        init_sfpu(dfb_input, dfb_output);
        copy_tile_to_dst_init_short(dfb_input);
        dfb_scaler_obj.wait_front(onetile);
        PACK((llk_pack_reduce_mask_config<REDUCE_DIM, PackMode::Default>(dfb_output)));

        for (uint32_t nc = 0; nc < NC; ++nc) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                tile_regs_acquire();
                negative_tile_init();
                if (Wt > 1) {
                    compute_kernel_lib::detail::sfpu_reduce_max_fold_init<reduce_format>();
                }

                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    dfb_input_obj.wait_front(onetile);
                    if (wt == 0) {
                        copy_tile(dfb_input, 0, acc_dst);
                        if constexpr (reduce_format == DataFormat::Int32) {
                            negative_tile_int32(acc_dst);
                        } else {
                            negative_tile(acc_dst);
                        }
                    } else {
                        copy_tile(dfb_input, 0, work_dst);
                        if constexpr (reduce_format == DataFormat::Int32) {
                            negative_tile_int32(work_dst);
                        } else {
                            negative_tile(work_dst);
                        }
                        compute_kernel_lib::detail::sfpu_reduce_max_fold_tile<reduce_format>(
                            acc_dst, work_dst, acc_dst);
                    }
                    dfb_input_obj.pop_front(onetile);
                }

                sfpu_reduce_init<REDUCE_OP, reduce_format>();
                sfpu_reduce<REDUCE_OP, reduce_format, REDUCE_DIM>(acc_dst, /*ct_dim=*/1, /*rt_dim=*/1);
                if constexpr (reduce_format == DataFormat::Int32) {
                    negative_tile_int32(acc_dst);
                } else {
                    negative_tile(acc_dst);
                }
#ifdef REDUCE_POST_MUL
                compute_kernel_lib::detail::reduce_post_mul_tile<reduce_format>(acc_dst, post_mul_scaler_bits);
#endif

                tile_regs_commit();
                dfb_output_obj.reserve_back(onetile);
                tile_regs_wait();
                pack_tile(acc_dst, dfb_output);
                tile_regs_release();
                dfb_output_obj.push_back(onetile);
            }
        }

        PACK((llk_pack_reduce_mask_clear()));
        // The scaler tile is waited once and reused for the whole reduction; pop it at the
        // end so the CB is left balanced.
        dfb_scaler_obj.pop_front(onetile);
        return;
    }

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
        int dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                dfb_input_obj.wait_front(onetile);
                tile_regs_acquire();
                copy_tile_init(dfb_input);
                copy_tile(dfb_input, 0, dst_idx);
                negative_tile_init();
                negative_tile(dst_idx);
                tile_regs_wait();
                dfb_input_obj.pop_front(onetile);
                dfb_ineg_obj.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, dfb_ineg);
                tile_regs_release();
                dfb_ineg_obj.push_back(onetile);

                tile_regs_acquire();
                if (wt > 0) {
                    dfb_acc_obj.wait_front(onetile);
                    copy_tile_init(dfb_acc);
                    copy_tile(dfb_acc, 0, dst_idx);
                }

                dfb_ineg_obj.wait_front(onetile);
                constexpr bool swap_operands = (REDUCE_DIM == ReduceDim::REDUCE_ROW) && (REDUCE_OP != PoolType::MAX);
                if constexpr (swap_operands) {
                    reconfig_data_format(dfb_scaler, dfb_ineg);
                }
                reduce_init<REDUCE_OP, REDUCE_DIM>(dfb_ineg, dfb_scaler, dfb_acc);
                reduce_tile<REDUCE_OP, REDUCE_DIM>(dfb_ineg, dfb_scaler, 0, 0, dst_idx);
                reduce_uninit();
                tile_regs_wait();
                dfb_ineg_obj.pop_front(onetile);
                if (wt > 0) {
                    dfb_acc_obj.pop_front(onetile);
                }
                dfb_acc_obj.reserve_back(onetile);
                tile_regs_commit();
                pack_tile(dst_idx, dfb_acc);
                tile_regs_release();
                dfb_acc_obj.push_back(onetile);
            }  // wt

            dfb_acc_obj.wait_front(onetile);
            tile_regs_acquire();
            copy_tile_init(dfb_acc);
            copy_tile(dfb_acc, 0, dst_idx);
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
            dfb_acc_obj.pop_front(onetile);
            dfb_output_obj.reserve_back(onetile);
            tile_regs_commit();
            pack_tile(dst_idx, dfb_output);
            tile_regs_release();
            dfb_output_obj.push_back(onetile);
        }  // ht
    }  // nc
    // The scaler tile is waited once and reused for the whole reduction; pop it at the
    // end so the CB is left balanced.
    dfb_scaler_obj.pop_front(onetile);
}
