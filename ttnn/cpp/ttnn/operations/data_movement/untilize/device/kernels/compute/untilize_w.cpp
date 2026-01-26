// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/untilize.h"
#include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    uint32_t third_dim = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

    // tile_width=1 (single tile per row)
    compute_kernel_lib::untilize<
        UntilizeConfig<WidthInTiles<1>, InputCB<tt::CBIndex::c_0>, OutputCB<tt::CBIndex::c_16>>>(
        per_core_block_cnt * per_core_block_tile_cnt * third_dim);
}
