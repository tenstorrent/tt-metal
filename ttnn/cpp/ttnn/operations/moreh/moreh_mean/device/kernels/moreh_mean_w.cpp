// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/mask.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t origin_W = get_compile_time_arg_val(3);

    auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    DataflowBuffer cb_scaler_obj(cb_scaler);
    constexpr auto cb_mask_w = tt::CBIndex::c_3;
    DataflowBuffer cb_mask_w_obj(cb_mask_w);
    constexpr auto cb_accum_dst = tt::CBIndex::c_24;
    DataflowBuffer cb_accum_dst_obj(cb_accum_dst);
    constexpr auto cb_masked_input = tt::CBIndex::c_25;
    constexpr auto cb_out = tt::CBIndex::c_16;
    DataflowBuffer cb_out_obj(cb_out);
    constexpr bool do_mask_w = (origin_W % TILE_WIDTH) != 0;
    constexpr bool is_w_single_tile = Wt == 1;

    binary_op_init_common(cb_input, cb_input, cb_out);

    cb_scaler_obj.wait_front(1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;

    if constexpr (do_mask_w) {
        cb_mask_w_obj.wait_front(onetile);
    }

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            cb_input = tt::CBIndex::c_0;
            if constexpr (!is_w_single_tile) {
                tile_regs_acquire();

                for (uint32_t wt = 0; wt < Wt - 1; ++wt) {
                    DataflowBuffer(cb_input).wait_front(onetile);
#if defined FP32_DEST_ACC_EN
                    reconfig_data_format(cb_input, cb_scaler);
#endif
                    matmul_init(cb_input, cb_scaler, false);
                    matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx);
                    DataflowBuffer(cb_input).pop_front(onetile);
                }
                tile_regs_commit();

                cb_accum_dst_obj.reserve_back(onetile);
                tile_regs_wait();
                pack_tile_with_dt(reduce_dst_idx, DataflowBuffer(cb_accum_dst));
                tile_regs_release();
                cb_accum_dst_obj.push_back(onetile);
            }

            if constexpr (do_mask_w) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::CopyTile<ckl::input(tt::CBIndex::c_0, ckl::InputLifecycle::Streaming, kDataFormatReconfig)>{},
                    ckl::CopyTile<
                        ckl::input(cb_mask_w, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                        ckl::Dst::D1>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                    ckl::PackTile<ckl::output(
                        cb_masked_input, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});
                cb_input = cb_masked_input;
            }

            tile_regs_acquire();
            DataflowBuffer(cb_input).wait_front(onetile);
            if constexpr (!is_w_single_tile) {
                cb_accum_dst_obj.wait_front(onetile);
                copy_tile_init_with_dt(DataflowBuffer(cb_accum_dst));
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
            pack_tile_with_dt(reduce_dst_idx, DataflowBuffer(cb_out));
            tile_regs_release();
            cb_out_obj.push_back(onetile);

            DataflowBuffer(cb_input).pop_front(onetile);
            if constexpr (!is_w_single_tile) {
                cb_accum_dst_obj.pop_front(onetile);
            }
        }
    }

    if constexpr (do_mask_w) {
        cb_mask_w_obj.pop_front(onetile);
    }
    cb_scaler_obj.pop_front(onetile);
}
