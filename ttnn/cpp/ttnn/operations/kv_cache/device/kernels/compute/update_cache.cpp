// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr std::uint32_t cache = dfb::cache;
    constexpr std::uint32_t input = dfb::input;
    constexpr std::uint32_t untilized_cache = dfb::untilized_cache;
    constexpr std::uint32_t untilized_cache2 = dfb::untilized_cache2;
    constexpr std::uint32_t untilized_input = dfb::untilized_input;
    constexpr std::uint32_t output = dfb::output;
    constexpr std::uint32_t num_batched_heads = get_arg(args::num_batched_heads);
    constexpr std::uint32_t Wt = get_arg(args::Wt);
    constexpr std::uint32_t granularity = get_arg(args::granularity);
    constexpr std::uint32_t u_count = get_arg(args::u_count);

    compute_kernel_hw_startup(input, untilized_input);

    for (std::uint32_t h = 0; h < num_batched_heads; ++h) {
        // Untilize input (standalone operation)
        compute_kernel_lib::untilize<
            Wt,
            input,
            untilized_input,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);

        for (std::uint32_t u = 0; u < u_count; ++u) {
            compute_kernel_lib::untilize<Wt, cache, untilized_cache>(granularity);

            // Wait on writer to update block, then tilize back
            compute_kernel_lib::tilize<Wt, untilized_cache2, output>(granularity);
        }
        reconfig_data_format_srca(cache, input);
    }
}
