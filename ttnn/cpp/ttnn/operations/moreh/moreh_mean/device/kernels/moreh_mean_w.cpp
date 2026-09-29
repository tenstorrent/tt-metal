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
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::units_per_core);  // Per-core output-row count, not tile height.
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t NC = get_arg(args::NC);
    constexpr uint32_t origin_W = get_arg(args::origin_W);

    // This switches between two named DFBs below, so keep the resolved index rather than
    // the accessor type inferred by auto.
    uint32_t dfb_input_id = dfb::input;
    DataflowBuffer dfb_input_obj(dfb::input);
    DataflowBuffer dfb_scaler_obj(dfb::scaler);
    DataflowBuffer dfb_mask_w_obj(dfb::mask_w);
    DataflowBuffer dfb_accum_dst_obj(dfb::accum_dst);
    DataflowBuffer dfb_masked_input_obj(dfb::masked_input);
    DataflowBuffer dfb_out_obj(dfb::out);
    constexpr bool do_mask_w = (origin_W % TILE_WIDTH) != 0;
    constexpr bool is_w_single_tile = Wt == 1;
    DataflowBuffer& dfb_reduction_input_obj = do_mask_w ? dfb_masked_input_obj : dfb_input_obj;

    compute_kernel_hw_startup(dfb_input_id, dfb_input_id, dfb::out);

    dfb_scaler_obj.wait_front(1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;

    if constexpr (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);
    }

    for (uint32_t nc = 0; nc < NC; nc++) {
        // Input is W-contiguous; each output tile reduces one row of W tiles.
        // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
        // in this case we just sequentially add to accumulator all the W-tiles in a row
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            dfb_input_id = dfb::input;
            if constexpr (!is_w_single_tile) {
                tile_regs_acquire();

                for (uint32_t wt = 0; wt < Wt - 1; ++wt) {
                    dfb_input_obj.wait_front(onetile);
#if defined FP32_DEST_ACC_EN
                    reconfig_data_format(dfb_input_id, dfb::scaler);
#endif
                    matmul_init(dfb_input_id, dfb::scaler, false);
                    matmul_tiles(dfb_input_id, dfb::scaler, 0, 0, reduce_dst_idx);
                    dfb_input_obj.pop_front(onetile);
                }
                tile_regs_commit();

                dfb_accum_dst_obj.reserve_back(onetile);
                tile_regs_wait();
                pack_tile_with_dt(reduce_dst_idx, dfb_accum_dst_obj);
                tile_regs_release();
                dfb_accum_dst_obj.push_back(onetile);
            }

            if constexpr (do_mask_w) {
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(onetile),
                    ckl::CopyTile<ckl::input(
                        dfb::input, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
                    ckl::CopyTile<
                        ckl::input(dfb::mask_w, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                        ckl::Dst::D1>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                    ckl::PackTile<ckl::output(
                        dfb::masked_input,
                        ckl::ReservePolicy::PerTile,
                        ckl::PushPolicy::PerTile,
                        kDataFormatReconfig)>{});
                dfb_input_id = dfb::masked_input;
            }

            tile_regs_acquire();
            dfb_reduction_input_obj.wait_front(onetile);
            if constexpr (!is_w_single_tile) {
                dfb_accum_dst_obj.wait_front(onetile);
                copy_tile_init_with_dt(dfb_accum_dst_obj);
                copy_tile(dfb::accum_dst, 0, reduce_dst_idx);
            }

#if defined FP32_DEST_ACC_EN
            reconfig_data_format(dfb_input_id, dfb::scaler);
#endif
            matmul_init(dfb_input_id, dfb::scaler, false);
            matmul_tiles(dfb_input_id, dfb::scaler, 0, 0, reduce_dst_idx);
            tile_regs_commit();

            dfb_out_obj.reserve_back(onetile);
            tile_regs_wait();
            pack_tile_with_dt(reduce_dst_idx, dfb_out_obj);
            tile_regs_release();
            dfb_out_obj.push_back(onetile);

            dfb_reduction_input_obj.pop_front(onetile);
            if constexpr (!is_w_single_tile) {
                dfb_accum_dst_obj.pop_front(onetile);
            }
        }
    }

    if constexpr (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
    dfb_scaler_obj.pop_front(onetile);
}
