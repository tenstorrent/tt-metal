// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#ifdef DO_COL_MASK
#include "ttnn/kernel/dataflow/moreh_common.hpp"  // generate_mask_w<T>
#endif

void kernel_main() {
    constexpr bool is_all_to_all_worker = get_compile_time_arg_val(0) == 1;
    constexpr bool use_welford = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
    const uint32_t scalar_w_bits = get_arg_val<uint32_t>(1);
    float scalar_w_f = __builtin_bit_cast(float, scalar_w_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_in_2, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scalar_w_f);

#ifdef DO_COL_MASK
    // Generate this core's column mask on-device (block_w tiles, one per width-tile position), zeroing
    // the boundary tile's padding columns and any all-padding tiles. The core's width position comes
    // from gamma_tile_start_id (= width_index * block_w), so no extra runtime arg is needed.
    {
        constexpr uint32_t block_w = get_compile_time_arg_val(3);
        const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(5);
        constexpr uint32_t cb_col_mask = get_named_compile_time_arg_val("cb_col_mask");
        constexpr uint32_t logical_K = get_named_compile_time_arg_val("logical_K");
        constexpr uint32_t mask_fp32 = get_named_compile_time_arg_val("mask_fp32");
        constexpr uint32_t tile_w = 32;
        CircularBuffer cb_col_mask_obj(cb_col_mask);
        const uint32_t core_start_col = (gamma_tile_start_id / block_w) * block_w * tile_w;
        for (uint32_t wt = 0; wt < block_w; wt++) {
            const uint32_t tile_start_col = core_start_col + wt * tile_w;
            uint32_t mask_w;
            if (logical_K <= tile_start_col) {
                mask_w = 0;
            } else if (logical_K - tile_start_col >= tile_w) {
                mask_w = tile_w;
            } else {
                mask_w = logical_K - tile_start_col;
            }
            if constexpr (mask_fp32 == 1) {
                generate_mask_w<uint32_t>(cb_col_mask_obj, mask_w);
            } else {
                generate_mask_w<uint16_t>(cb_col_mask_obj, mask_w);
            }
        }
    }
#endif

    if constexpr (is_all_to_all_worker && !use_welford) {
        constexpr uint32_t cb_in_4 = tt::CBIndex::c_4;
        const uint32_t scalar_c_bits = get_arg_val<uint32_t>(0);
        float scalar_c_f = __builtin_bit_cast(float, scalar_c_bits);
        dataflow_kernel_lib::prepare_reduce_scaler<cb_in_4, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scalar_c_f);
    }
}
