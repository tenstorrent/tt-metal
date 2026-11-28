// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdint.h>
#include "api/compile_time_args.h"
#include <api/dataflow/dataflow_api.h>
#include "ttnn/operations/pool/device/kernels/pool_kernels_common.hpp"
#include "../grid_sample_reader_common.hpp"
#include "experimental/kernel_args.h"

#define PRINT_AND_PROFILE 0
#if PRINT_AND_PROFILE
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#endif

void kernel_main() {
    uint32_t num_pages = get_arg(args::num_pages);
    uint32_t start_page_id = get_arg(args::start_page_id);

    constexpr uint32_t input_stick_nbytes = get_arg(args::input_stick_nbytes);
    constexpr uint32_t grid_stick_nbytes = get_arg(args::grid_stick_nbytes);

    constexpr uint32_t input_height = get_arg(args::input_height);
    constexpr uint32_t input_width = get_arg(args::input_width);
    constexpr uint32_t grid_batches = get_arg(args::grid_batches);
    constexpr uint32_t grid_dtype = get_arg(args::grid_dtype);
    constexpr uint32_t output_hw_size = get_arg(args::output_hw_size);
    constexpr bool use_precomputed_grid = get_arg(args::use_precomputed_grid);
    constexpr bool align_corners = get_arg(args::align_corners);
    constexpr uint32_t in_nblocks_c = get_arg(args::in_nblocks_c);
    constexpr uint32_t input_chunk_nbytes = get_arg(args::input_chunk_nbytes);
    constexpr bool last_chunk_partial = get_arg(args::last_chunk_partial);

    const auto grid_tensor_accessor = TensorAccessor(tensor::grid);
    const auto input_tensor_accessor = TensorAccessor(tensor::input);

    DataflowBuffer grid_dfb(dfb::grid);
    Noc noc;

    const uint32_t end_id = start_page_id + num_pages;

    /*
    In the case of grid sampling, we need to account for the fact that the grid coordinates may fall outside the bounds
    of the input image. Since the padding mode is zero, we would simply set the weights for the appropriate sticks to
    zero in the for loop, and simply do not read from DRAM. In that case the stick we send to reduction would be the
    last pixel that we read for the appropriate location (SW, SE, NW, NE), but since weights are 0 this is not a
    problem.

    However, if there was no previous read for the appropriate stick, the memory in that location is invalid, and could
    include NaN and Inf values. For that reason we zero out the input_dfb at the start.
    */
    DataflowBuffer input_dfb(dfb::in);
    DataflowBuffer scalar_dfb(dfb::scalar);
    zero_out_tiles<dfb::in>(noc, input_dfb);

    // Calculate starting batch from starting spatial position (avoid division in loop)
    uint32_t curr_batch = start_page_id / output_hw_size;
    uint32_t spatial_points_processed = start_page_id % output_hw_size;
    uint32_t batch_offset = curr_batch * input_height * input_width;

    // Outer loop: iterate over spatial positions (output sticks)
    for (uint32_t spatial_pos = start_page_id; spatial_pos < end_id; ++spatial_pos) {
        // Read the grid stick for this spatial position (contains grid_batches sets of coordinates)
        noc.async_read(grid_tensor_accessor, grid_dfb, grid_stick_nbytes, {.page_id = spatial_pos}, {});
        noc.async_read_barrier();

        // Cast to appropriate pointer type for grid data access
        volatile tt_l1_ptr uint16_t* grid_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(grid_dfb.get_write_ptr());

        // Inner loop: process grid_batches coordinate sets within this spatial position
        for (uint32_t grid_idx = 0; grid_idx < grid_batches; ++grid_idx) {
            // Direct template dispatch - no branching needed
            process_grid_point<
                grid_dtype,
                use_precomputed_grid,
                align_corners,
                input_height,
                input_width,
                input_stick_nbytes,
                in_nblocks_c,
                input_chunk_nbytes,
                last_chunk_partial,
                dfb::in,
                dfb::scalar>(noc, input_dfb, scalar_dfb, grid_ptr, grid_idx, input_tensor_accessor, batch_offset);
        }

        // Update batch tracking (avoid division in loop)
        ++spatial_points_processed;
        if (spatial_points_processed == output_hw_size) {
            spatial_points_processed = 0;
            ++curr_batch;
            batch_offset = curr_batch * input_height * input_width;
        }
    }
}
