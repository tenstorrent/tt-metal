// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t origin_W = get_compile_time_arg_val(3);

    auto dfb_input_id = tt::CBIndex::c_0;
    constexpr auto dfb_scaler_id = tt::CBIndex::c_2;
    DataflowBuffer dfb_scaler_obj(dfb_scaler_id);
    constexpr auto dfb_mask_w_id = tt::CBIndex::c_3;
    DataflowBuffer dfb_mask_w_obj(dfb_mask_w_id);
    constexpr auto dfb_accum_dst_id = tt::CBIndex::c_24;
    DataflowBuffer dfb_accum_dst_obj(dfb_accum_dst_id);
    constexpr auto dfb_masked_input_id = tt::CBIndex::c_25;
    constexpr auto dfb_out_id = tt::CBIndex::c_16;
    DataflowBuffer dfb_out_obj(dfb_out_id);
    constexpr uint32_t TILE_W = 32;
    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;

    compute_kernel_hw_startup(dfb_input_id, dfb_scaler_id, dfb_out_id);

    dfb_scaler_obj.wait_front(1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    const uint32_t mask_dst_idx = reduce_dst_idx + 1;

    if (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);
    }

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            dfb_input_id = tt::CBIndex::c_0;
            bool is_w_single_tile = (Wt == 1);
            if (!is_w_single_tile) {
                tile_regs_acquire();
                for (uint32_t wt = 0; wt < Wt - 1; ++wt) {
                    DataflowBuffer(dfb_input_id).wait_front(onetile);
#if defined FP32_DEST_ACC_EN
                    reconfig_data_format(dfb_input_id, dfb_scaler_id);
#endif
                    matmul_init(dfb_input_id, dfb_scaler_id, false);
                    matmul_tiles(dfb_input_id, dfb_scaler_id, 0, 0, reduce_dst_idx);

                    DataflowBuffer(dfb_input_id).pop_front(onetile);
                }
                tile_regs_commit();
                dfb_accum_dst_obj.reserve_back(onetile);
                tile_regs_wait();
#if defined FP32_DEST_ACC_EN
                pack_reconfig_data_format(dfb_accum_dst_id);
#endif
                pack_tile(reduce_dst_idx, dfb_accum_dst_id);
                tile_regs_release();
                dfb_accum_dst_obj.push_back(onetile);
            }

            if (do_mask_w) {
                // CopyTile<input(c_0)> + CopyTile<input(dfb_mask_w_id), D1> + Mask + PackTile.
                // dfb_input_id is always c_0 here (reset at line 46 before this conditional).
                // Reconfig: chain Input+Output (fold elides no-op transitions); matches
                // the FP32_DEST_ACC_EN-guarded reconfigs in the original.
                compute_kernel_lib::eltwise_chain(
                    compute_kernel_lib::EltwiseShape::tiles(onetile),
                    compute_kernel_lib::CopyTile<compute_kernel_lib::input(tt::CBIndex::c_0)>{},
                    compute_kernel_lib::CopyTile<
                        compute_kernel_lib::input(
                            dfb_mask_w_id, compute_kernel_lib::WaitPolicy::None, compute_kernel_lib::PopPolicy::None),
                        compute_kernel_lib::Dst::D1>{},
                    compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::PackTile<compute_kernel_lib::output(dfb_masked_input_id)>{});
                dfb_input_id = dfb_masked_input_id;
            }

            tile_regs_acquire();
            DataflowBuffer(dfb_input_id).wait_front(onetile);
            if (!is_w_single_tile) {
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(dfb_accum_dst_id);
#endif
                dfb_accum_dst_obj.wait_front(onetile);
                copy_tile_to_dst_init_short(dfb_accum_dst_id);
                copy_tile(dfb_accum_dst_id, 0, reduce_dst_idx);
            }

#if defined FP32_DEST_ACC_EN
            reconfig_data_format(dfb_input_id, dfb_scaler_id);
#endif
            matmul_init(dfb_input_id, dfb_scaler_id, false);
            matmul_tiles(dfb_input_id, dfb_scaler_id, 0, 0, reduce_dst_idx);
            tile_regs_commit();

            dfb_out_obj.reserve_back(onetile);
            tile_regs_wait();
#if defined FP32_DEST_ACC_EN
            pack_reconfig_data_format(dfb_out_id);
#endif
            pack_tile(reduce_dst_idx, dfb_out_id);
            tile_regs_release();
            dfb_out_obj.push_back(onetile);

            DataflowBuffer(dfb_input_id).pop_front(onetile);
            if (!is_w_single_tile) {
                dfb_accum_dst_obj.pop_front(onetile);
            }
        }
    }

    if (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
    dfb_scaler_obj.pop_front(onetile);
}
