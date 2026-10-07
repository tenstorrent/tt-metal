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

    // Which of the two inputs this instance serves is structural: the factory places one compute
    // kernel per input shard grid, and each binds only its own input buffer.
    compute_kernel_hw_startup(dfb::in, dfb::untilized_in);

    // Untilize input (single block, init only - no uninit needed)
    // Four-tile fast untilize chunks: this kernel measured slower with eight (tt-metal#58736).
    compute_kernel_lib::untilize<
        Wt,
        dfb::in,
        dfb::untilized_in,
        compute_kernel_lib::untilize_config::InitUninitMode::InitOnly,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        compute_kernel_lib::untilize_config::RemapMode::Configure,
        compute_kernel_lib::untilize_config::FastChunk::FourTiles>(1);

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Untilize a block from the cache with reconfiguration from previous iteration
        compute_kernel_lib::untilize<
            Wt,
            dfb::cache,
            dfb::untilized_cache,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
            compute_kernel_lib::untilize_config::RemapMode::Configure,
            compute_kernel_lib::untilize_config::FastChunk::FourTiles>(1);

        // Wait on writer to update block. Tilize with reconfiguration
        compute_kernel_lib::tilize<Wt, dfb::untilized_cache2, dfb::out>(1);
    }
}
