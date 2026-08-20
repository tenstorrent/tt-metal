// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "experimental/kernel_args.h"
#include "grid_sample_nearest_impl.hpp"

template <
    uint32_t input_stick_nbytes,
    uint32_t grid_stick_nbytes,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t grid_batching_factor,
    uint32_t grid_dtype,
    uint32_t grid_hw,
    uint32_t use_precomputed_grid,
    uint32_t align_corners,
    uint32_t split_reader,
    uint32_t reader_id,
    uint32_t grid_nsticks_per_core,
    uint32_t batch_size>
TT_KERNEL void writer_grid_sample_nearest_interleaved(uint32_t start_page_id) {
    const auto input = TensorAccessor(tensor::input);
    const auto grid = TensorAccessor(tensor::grid);
    grid_sample_nearest_impl<
        grid_dtype,
        false,
        use_precomputed_grid,
        align_corners,
        input_height,
        input_width,
        input_stick_nbytes,
        grid_stick_nbytes,
        grid_batching_factor,
        grid_hw,
        split_reader,
        reader_id,
        grid_nsticks_per_core,
        batch_size,
        dfb::grid,
        dfb::output,
        dfb::fill>(input, grid, start_page_id, start_page_id);
}
