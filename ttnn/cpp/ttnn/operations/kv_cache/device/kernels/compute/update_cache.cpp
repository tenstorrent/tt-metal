// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cache_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache_cb = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache2_cb = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_in_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(5);
    constexpr uint32_t num_batched_heads = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t granularity = get_compile_time_arg_val(8);
    constexpr uint32_t u_count = get_compile_time_arg_val(9);

    compute_kernel_hw_startup(in_cb, untilized_in_cb);

    // Config for untilizing new input tokens - used for init/reinit between heads
    using UntilizeNewTokenSetup = UntilizeConfig<WidthInTiles<Wt>, InputCB<in_cb>, OutputCB<untilized_in_cb>>;

    // Config for untilizing new input tokens in the loop (init/uninit handled outside)
    using UntilizeNewTokenInLoop = UntilizeConfig<
        WidthInTiles<Wt>,
        InputCB<in_cb>,
        OutputCB<untilized_in_cb>,
        UntilizeFlags::SKIP_INIT | UntilizeFlags::SKIP_UNINIT>;

    // Config for untilizing existing cache blocks
    using UntilizeCacheBlock = UntilizeConfig<WidthInTiles<Wt>, InputCB<cache_cb>, OutputCB<untilized_cache_cb>>;

    // Config for re-tilizing the updated cache (with data format reconfig from previous cache read)
    using RetilizeUpdatedCache =
        TilizeConfig<InputCB<untilized_cache2_cb>, OutputCB<out_cb>, TilizeFlags::DT_RECONFIG, PreviousCB<cache_cb>>;

    compute_kernel_lib::untilize_init<UntilizeNewTokenSetup>();

    for (uint32_t h = 0; h < num_batched_heads; ++h) {
        // Untilize new input token for this head
        compute_kernel_lib::untilize<UntilizeNewTokenInLoop>(1);

        reconfig_data_format_srca(in_cb, cache_cb);
        for (uint32_t u = 0; u < u_count; ++u) {
            // Untilize cache block to be updated
            compute_kernel_lib::untilize<UntilizeCacheBlock>(granularity);

            reconfig_data_format_srca(cache_cb, untilized_cache2_cb);
            pack_reconfig_data_format(untilized_cache_cb, out_cb);

            // Writer updates the untilized cache with new token. Re-tilize the result.
            compute_kernel_lib::tilize<RetilizeUpdatedCache>(Wt, granularity);

            pack_reconfig_data_format(out_cb, untilized_cache_cb);
        }
        reconfig_data_format_srca(cache_cb, in_cb);

        // Re-initialize for next head
        compute_kernel_lib::untilize_init<UntilizeNewTokenSetup>();
    }
}
