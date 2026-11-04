// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-row variant of paged_fused_update_cache.cpp (rows_per_user > 1): untilize all T input tile
// rows once, then num_passes read-modify-write passes over the cache (one per distinct cache tile the
// rows can land in; the dataflow kernels skip the DRAM traffic of passes with no rows).

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
    constexpr auto rows_per_user = get_arg(args::rows_per_user);
    constexpr auto num_passes = get_arg(args::num_passes);

    compute_kernel_hw_startup(dfb::in, dfb::untilized_in);

    // Untilize the T input tile rows (init only - no uninit needed), as the single-row kernel does
    // for its one row.
    compute_kernel_lib::untilize<
        Wt,
        dfb::in,
        dfb::untilized_in,
        compute_kernel_lib::untilize_config::InitUninitMode::InitOnly,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(rows_per_user);

    for (uint32_t pass = 0; pass < num_passes; ++pass) {
        for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
            // Untilize a block from the cache with reconfiguration from previous iteration
            compute_kernel_lib::untilize<Wt, dfb::cache, dfb::untilized_cache>(1);

            // Wait on writer to update block. Tilize with reconfiguration
#ifdef ARCH_QUASAR
            // See paged_fused_update_cache.cpp: retarget the packer BFD at dfb::out before packing.
            pack_init(dfb::out);
#endif
            compute_kernel_lib::tilize<Wt, dfb::untilized_cache2, dfb::out>(1);
        }
    }
}
