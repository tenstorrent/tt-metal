// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    uint32_t rt_args_idx = 0;
    const bool has_work = get_arg_val<uint32_t>(rt_args_idx++);
    if (!has_work) {
        return;
    }
    const bool is_input1 = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint32_t in1_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in2_cb = get_compile_time_arg_val(1);
    uint32_t in_cb = in1_cb;
    if (!is_input1) {
        in_cb = in2_cb;
    }

    constexpr uint32_t cache_cb = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache_cb = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_cache2_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t num_heads = get_compile_time_arg_val(7);

    compute_kernel_hw_startup(cache_cb, untilized_cache_cb);

    // Config for untilizing existing cache blocks
    using UntilizeCacheBlock = UntilizeConfig<WidthInTiles<Wt>, InputCB<cache_cb>, OutputCB<untilized_cache_cb>>;

    // Config for re-tilizing the updated cache (with data format reconfig)
    using RetilizeUpdatedCache =
        TilizeConfig<InputCB<untilized_cache2_cb>, OutputCB<out_cb>, TilizeFlags::DT_RECONFIG, PreviousCB<cache_cb>>;

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
