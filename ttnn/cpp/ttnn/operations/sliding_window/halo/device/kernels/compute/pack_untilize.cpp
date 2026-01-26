// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/untilize.h"
#include "api/compute/pack_untilize.h"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;
constexpr uint32_t NUM_RISCV_DATA_MOVEMENT_CORES = 2;
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id0 = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_id1 = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(3);  // number of tiles along width of shard
    constexpr uint32_t block_size = get_compile_time_arg_val(4);  // number of tiles along height that make up a block

    const uint32_t total_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(src_cb_id, out_cb_id0);

    // Config for init/uninit (only needs one, uses first output CB)
    using SetupConfig = UntilizeConfig<WidthInTiles<tiles_per_row>, InputCB<src_cb_id>, OutputCB<out_cb_id0>>;

    // Loop configs alternate between two output CBs, skip init/uninit since we handle those outside
    using EvenBlockConfig = UntilizeConfig<
        WidthInTiles<tiles_per_row>,
        InputCB<src_cb_id>,
        OutputCB<out_cb_id0>,
        UntilizeFlags::SKIP_INIT | UntilizeFlags::SKIP_UNINIT>;
    using OddBlockConfig = UntilizeConfig<
        WidthInTiles<tiles_per_row>,
        InputCB<src_cb_id>,
        OutputCB<out_cb_id1>,
        UntilizeFlags::SKIP_INIT | UntilizeFlags::SKIP_UNINIT>;

    compute_kernel_lib::untilize_init<SetupConfig>();

    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        if (block_idx % 2 == 0) {
            compute_kernel_lib::untilize<EvenBlockConfig>(1, block_size);
        } else {
            compute_kernel_lib::untilize<OddBlockConfig>(1, block_size);
        }
    }

    compute_kernel_lib::untilize_uninit<SetupConfig>();
}
