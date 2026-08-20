// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/pool_2d_compute_impl.hpp"

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
    // Rotate and Grid Sample use a single reader, row-major output, and no pre-tilize path.
    // Bind inactive branches to valid DFB tokens; if constexpr removes every use.
    pool_2d_compute_impl<
        in_ntiles_c,
        window_size_hw,
        0,
        max_out_sticks_per_core,
        in_c,
        in_nblocks_c,
        max_sticks_for_reduction,
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
