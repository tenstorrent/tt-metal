// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#ifdef DO_COL_MASK
#include "col_mask_dataflow.h"
#endif

void kernel_main() {
    constexpr bool is_all_to_all_worker = get_arg(args::is_all_to_all_worker) == 1;
    const uint32_t scalar_w_bits = get_arg(args::scalar_w);
    float scalar_w_f = __builtin_bit_cast(float, scalar_w_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scalar_w_f);

#ifdef DO_COL_MASK
    constexpr auto block_w = get_arg(args::block_w);
    constexpr auto logical_K = get_arg(args::logical_K);
    // This core's first tile index along the width (the normalized dimension): width_index * block_w,
    // the start of this core's width shard.
    const uint32_t width_shard_tile_start_id = get_arg(args::width_shard_tile_start_id);
    generate_col_mask(dfb::col_mask, block_w, logical_K, width_shard_tile_start_id);
#endif

#ifndef USE_WELFORD
    if constexpr (is_all_to_all_worker) {
        const uint32_t scalar_c_bits = get_arg(args::scalar_c);
        float scalar_c_f = __builtin_bit_cast(float, scalar_c_bits);
        dataflow_kernel_lib::
            prepare_reduce_scaler<dfb::scaler_global, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                scalar_c_f);
    }
#endif
}
