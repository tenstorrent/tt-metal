// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/compile_time_args.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#ifdef DO_COL_MASK
#include "col_mask_dataflow.h"
#endif

void kernel_main() {
    constexpr bool is_all_to_all_worker = get_named_compile_time_arg_val("is_all_to_all_worker") == 1;
    constexpr bool use_welford = get_named_compile_time_arg_val("use_welford") == 1;
    constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
    const uint32_t scalar_w_bits = get_arg_val<uint32_t>(1);
    float scalar_w_f = __builtin_bit_cast(float, scalar_w_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_in_2, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scalar_w_f);

#ifdef DO_COL_MASK
    constexpr uint32_t cb_col_mask = get_named_compile_time_arg_val("cb_col_mask");
    constexpr uint32_t block_w = get_named_compile_time_arg_val("block_w");
    constexpr uint32_t logical_K = get_named_compile_time_arg_val("logical_K");
    // This core's first tile index along the width (the normalized dimension): width_index * block_w,
    // the start of this core's width shard.
    const uint32_t width_shard_tile_start_id = get_arg_val<uint32_t>(5);
    generate_col_mask(cb_col_mask, block_w, logical_K, width_shard_tile_start_id);
#endif

    if constexpr (is_all_to_all_worker && !use_welford) {
        constexpr uint32_t cb_in_4 = tt::CBIndex::c_4;
        const uint32_t scalar_c_bits = get_arg_val<uint32_t>(0);
        float scalar_c_f = __builtin_bit_cast(float, scalar_c_bits);
        dataflow_kernel_lib::prepare_reduce_scaler<cb_in_4, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scalar_c_f);
    }
}
