// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/pool/device/kernels/compute/pool_2d_compute_impl.hpp"

template <
    uint32_t in_ntiles_c,
    uint32_t window_size_hw,
    uint32_t split_reader,
    uint32_t max_out_sticks_per_core,
    uint32_t in_c,
    uint32_t in_nblocks_c,
    uint32_t max_sticks_for_reduction,
    uint32_t is_avg_pool,
    uint32_t one_scalar_per_core,
    uint32_t is_output_tiled,
    uint32_t is_output_block_format>
TT_KERNEL void compute_pool_2d(uint32_t out_nhw_this_core) {
    constexpr auto in_cb_1 = dfb::in_cb_1;
    constexpr auto in_scalar_cb_1 = dfb::in_scalar_cb_1;
    constexpr auto pre_tilize_cb = dfb::pre_tilize_cb;
    constexpr auto fast_tilize_cb = dfb::fast_tilize_cb;

    pool_2d_compute_impl<
        in_ntiles_c,
        window_size_hw,
        split_reader,
        max_out_sticks_per_core,
        in_c,
        in_nblocks_c,
        max_sticks_for_reduction,
        is_avg_pool,
        dfb::in_cb_0,
        in_cb_1,
        dfb::in_scalar_cb_0,
        in_scalar_cb_1,
        dfb::out_cb,
        one_scalar_per_core,
        pre_tilize_cb,
        is_output_tiled,
        is_output_block_format,
        false,
        fast_tilize_cb>(out_nhw_this_core);
}
