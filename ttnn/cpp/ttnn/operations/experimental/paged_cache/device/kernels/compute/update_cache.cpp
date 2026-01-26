// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

// Helper constexpr function to compute num_blocks_per_col
constexpr uint32_t compute_num_blocks_per_col(uint32_t per_core_block_tile_cnt) {
    const uint32_t max_bct = DST_ACCUM_MODE ? 4 : 8;

    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (per_core_block_tile_cnt % bct == 0) {
            return per_core_block_tile_cnt / bct;
        }
    }

    return 1;
}

void kernel_main() {
    constexpr uint32_t cache_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache_cb = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache2_cb = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_in_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t num_heads = get_compile_time_arg_val(7);

    compute_kernel_hw_startup(in_cb, untilized_in_cb);

    // Config for untilizing new input tokens (skip uninit since we only do this once)
    using UntilizeNewToken =
        UntilizeConfig<WidthInTiles<Wt>, InputCB<in_cb>, OutputCB<untilized_in_cb>, UntilizeFlags::SKIP_UNINIT>;

    // Config for untilizing existing cache blocks
    using UntilizeCacheBlock = UntilizeConfig<WidthInTiles<Wt>, InputCB<cache_cb>, OutputCB<untilized_cache_cb>>;

    // Config for re-tilizing the updated cache (with data format reconfig)
    using RetilizeUpdatedCache =
        TilizeConfig<InputCB<untilized_cache2_cb>, OutputCB<out_cb>, TilizeFlags::DT_RECONFIG, PreviousCB<cache_cb>>;

    // Untilize the new input token
    compute_kernel_lib::untilize<UntilizeNewToken>(1);

    reconfig_data_format_srca(in_cb, cache_cb);
    pack_reconfig_data_format(untilized_in_cb, untilized_cache_cb);

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Untilize cache block to be updated
        compute_kernel_lib::untilize<UntilizeCacheBlock>(1);

        reconfig_data_format_srca(cache_cb, untilized_cache2_cb);
        pack_reconfig_data_format(untilized_cache_cb, out_cb);

        // Writer updates the untilized cache with new token. Re-tilize the result.
        compute_kernel_lib::tilize<RetilizeUpdatedCache>(Wt, 1);

        pack_reconfig_data_format(out_cb, untilized_cache_cb);
    }
}
