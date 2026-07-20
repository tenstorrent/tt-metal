// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t origin_W = get_compile_time_arg_val(3);

    auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    CircularBuffer cb_scaler_obj(cb_scaler);
    constexpr auto cb_mask_w = tt::CBIndex::c_3;
    CircularBuffer cb_mask_w_obj(cb_mask_w);
    constexpr auto cb_accum_dst = tt::CBIndex::c_24;
    CircularBuffer cb_accum_dst_obj(cb_accum_dst);
    constexpr auto cb_masked_input = tt::CBIndex::c_25;
    CircularBuffer cb_masked_input_obj(cb_masked_input);
    constexpr auto cb_out = tt::CBIndex::c_16;
    CircularBuffer cb_out_obj(cb_out);
    constexpr uint32_t TILE_W = 32;
    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;

    binary_op_init_common(cb_input, cb_scaler, cb_out);

    cb_scaler_obj.wait_front(1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    const uint32_t mask_dst_idx = reduce_dst_idx + 1;

    if (do_mask_w) {
        cb_mask_w_obj.wait_front(onetile);
    }

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            cb_input = tt::CBIndex::c_0;
            bool is_w_single_tile = (Wt == 1);
            if (!is_w_single_tile) {
                tile_regs_acquire();
                for (uint32_t wt = 0; wt < Wt - 1; ++wt) {
                    CircularBuffer(cb_input).wait_front(onetile);
#if defined FP32_DEST_ACC_EN
                    reconfig_data_format(cb_input, cb_scaler);
#endif
                    matmul_init(cb_input, cb_scaler, false);
                    matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx);

                    CircularBuffer(cb_input).pop_front(onetile);
                }
                tile_regs_commit();
                cb_accum_dst_obj.reserve_back(onetile);
                tile_regs_wait();
#if defined FP32_DEST_ACC_EN
                pack_reconfig_data_format(cb_accum_dst);
#endif
                pack_tile(reduce_dst_idx, cb_accum_dst);
                tile_regs_release();
                cb_accum_dst_obj.push_back(onetile);
            }

            if (do_mask_w) {
                // CopyTile<cb_input=c_0> + CopyTile<cb_mask_w, D1> + Mask + PackTile.
                // cb_input is always c_0 here (reset at line 46 before this conditional).
                // Reconfig: chain Input+Output (fold elides no-op transitions); matches
                // the FP32_DEST_ACC_EN-guarded reconfigs in the original.
                compute_kernel_lib::eltwise_chain(
                    compute_kernel_lib::EltwiseShape::tiles(onetile),
                    compute_kernel_lib::CopyTile<tt::CBIndex::c_0>{},
                    compute_kernel_lib::CopyTile<
                        cb_mask_w,
                        compute_kernel_lib::Dst::D1,
                        compute_kernel_lib::input(compute_kernel_lib::InputLifecycle::CallerManaged)>{},
                    compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::PackTile<cb_masked_input>{});
                cb_input = cb_masked_input;
            }

            tile_regs_acquire();
            CircularBuffer(cb_input).wait_front(onetile);
            if (!is_w_single_tile) {
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(cb_accum_dst);
#endif
                cb_accum_dst_obj.wait_front(onetile);
                copy_tile_to_dst_init_short(cb_accum_dst);
                copy_tile(cb_accum_dst, 0, reduce_dst_idx);
            }

#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_input, cb_scaler);
#endif
            matmul_init(cb_input, cb_scaler, false);
            matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx);
            tile_regs_commit();

            cb_out_obj.reserve_back(onetile);
            tile_regs_wait();
#if defined FP32_DEST_ACC_EN
            pack_reconfig_data_format(cb_out);
#endif
            pack_tile(reduce_dst_idx, cb_out);
            tile_regs_release();
            cb_out_obj.push_back(onetile);

            CircularBuffer(cb_input).pop_front(onetile);
            if (!is_w_single_tile) {
                cb_accum_dst_obj.pop_front(onetile);
            }
        }
    }

    if (do_mask_w) {
        cb_mask_w_obj.pop_front(onetile);
    }
    cb_scaler_obj.pop_front(onetile);
}
