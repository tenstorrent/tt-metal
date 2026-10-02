// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    const auto has_work = get_arg(args::has_work);
    if (!has_work) {
        return;
    }

    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto num_heads = get_arg(args::num_heads);

    // Row-major input needs no untilize step, so this kernel never touches an input buffer; the
    // writer drains the resident shard directly.
    compute_kernel_hw_startup(dfb::cache, dfb::untilized_cache);

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Untilize a block from the cache with reconfiguration
        compute_kernel_lib::untilize<Wt, dfb::cache, dfb::untilized_cache>(1);

        // Wait on writer to update block. Tilize with reconfiguration
        compute_kernel_lib::tilize<Wt, dfb::untilized_cache2, dfb::out>(1);
    }
}
