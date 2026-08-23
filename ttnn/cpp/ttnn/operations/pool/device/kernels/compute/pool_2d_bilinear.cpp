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
TT_KERNEL void pool_2d_bilinear() {
    static_assert(max_out_sticks_per_core > 0, "Bilinear kernels require a compile-time output-stick count");
    // Rotate and non-split GridSample use one reader and row-major output. The duplicate input/scalar handles are
    // valid aliases; split-reader selection stays false, so only reader 0 is consumed. Keep a separate wrapper from
    // the split-reader variant so an unused second consumer binding cannot affect implicit synchronization.
    pool_2d_compute_impl<
        in_ntiles_c,
        window_size_hw,
        0,
        max_out_sticks_per_core,
        in_c,
        in_nblocks_c,
        max_sticks_for_reduction,
        true,
        dfb::input,
        dfb::input,
        dfb::scalar,
        dfb::scalar,
        dfb::output,
        one_scalar_per_core,
        dfb::output,
        false,
        false,
        force_max_tiles_per_reduction_4,
        dfb::output>(0);
}
