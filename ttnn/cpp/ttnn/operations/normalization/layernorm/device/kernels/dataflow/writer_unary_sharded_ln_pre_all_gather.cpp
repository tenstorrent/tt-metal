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
    using LocalArgs = ttnn::kernel_lib::ReduceAuxiliaryArgs<0>;
    using ScaledArgs = ttnn::kernel_lib::ReduceAuxiliaryArgs<LocalArgs::next_compile_time_args_offset()>;
    using IdentityArgs = ttnn::kernel_lib::ReduceAuxiliaryArgs<ScaledArgs::next_compile_time_args_offset()>;
    using LocalAuxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<LocalArgs, dfb::scaler>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<LocalAuxiliary>();

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
        // Packed BF16 identity marks cores that must not apply the global scale twice.
        if (get_arg(args::scalar_c) == 0x3f803f80U) {
            using Auxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<IdentityArgs, dfb::scaler_global>;
            dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();
        } else {
            using Auxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<ScaledArgs, dfb::scaler_global>;
            dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();
        }
    }
#endif
}
