// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "experimental/kernel_args.h"
#include "pool_2d_compute_impl.hpp"

template <
    uint32_t in_ntiles_c,
    uint32_t window_size_hw,
    uint32_t max_out_sticks_per_core,
    uint32_t in_c,
    uint32_t in_nblocks_c,
    uint32_t max_sticks_for_reduction,
    uint32_t one_scalar_per_core,
    uint32_t force_max_tiles_per_reduction_4>
TT_KERNEL void pool_2d_bilinear_split() {
    static_assert(max_out_sticks_per_core > 0, "Bilinear kernels require a compile-time output-stick count");
    // pre_tilize and fast_tilize alias the row-major output DFB; both tiled-output branches are disabled. A distinct
    // wrapper keeps both reader consumers explicit instead of relying on an unused alias and implicit-sync behavior.
    pool_2d_compute_impl<
        in_ntiles_c,
        window_size_hw,
        1,
        max_out_sticks_per_core,
        in_c,
        in_nblocks_c,
        max_sticks_for_reduction,
        true,
        dfb::input0,
        dfb::input1,
        dfb::scalar0,
        dfb::scalar1,
        dfb::output,
        one_scalar_per_core,
        dfb::output,
        false,
        false,
        force_max_tiles_per_reduction_4,
        dfb::output>(0);
}
