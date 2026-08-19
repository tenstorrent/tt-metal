// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * TTNN Slice Operation - Multi-Core Writer Kernel (4D Support)
 *
 * This kernel handles the output data writing phase of the slice operation for multi-core
 * execution, writing sliced tensor data from circular buffer to output tensor memory.
 * Supports 1D, 2D, 3D, and 4D tensors with work distribution across cores.
 *
 * Metal 2.0: named kernel arguments + named tensor binding (tensor::out).
 *
 * Compatible with: TTNN framework, ROW_MAJOR_LAYOUT tensors, 1D-4D dimensions, multi-core execution
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Per-core runtime arguments (named).
    uint32_t tensor_rank = get_arg(args::tensor_rank);
    uint32_t output_w = get_arg(args::output_w);
    uint32_t output_h = get_arg(args::output_h);
    uint32_t output_d = get_arg(args::output_d);
    uint32_t output_n = get_arg(args::output_n);
    uint32_t element_size = get_arg(args::element_size);
    uint32_t num_rows_for_this_core = get_arg(args::num_rows_for_this_core);
    uint32_t start_row_for_this_core = get_arg(args::start_row_for_this_core);

    // Compile-time arguments
    constexpr uint32_t compile_time_element_size = get_arg(args::compile_time_element_size);

    // Calculate sizes - working with rows, not tiles
    uint32_t output_bytes_per_row = output_w * element_size;  // Dynamic element size

    // Set up TensorAccessor for output data - use row size as page size
    const auto s0 = TensorAccessor(tensor::out);

    Noc noc;
    // Create DataflowBuffer for Device 2.0 API
    DataflowBuffer cb_in(dfb::cb_in);

    // Multi-core work distribution: this core writes rows starting from start_row_for_this_core
    // Write each row from circular buffer to output tensor at the correct logical position
    for (uint32_t local_row = 0; local_row < num_rows_for_this_core; ++local_row) {
        cb_in.wait_front(1);
        uint32_t l1_read_addr = cb_in.get_read_ptr();

        // Calculate global output row index for this local row
        uint32_t global_output_row = start_row_for_this_core + local_row;

        // noc_async_write_sharded splits the write across shards for B/W-sharded outputs;
        // falls through to a single noc_async_write for interleaved / HEIGHT-sharded.
        tt::data_movement::common::noc_async_write_sharded(
            noc, l1_read_addr, s0, global_output_row, /*offset=*/0, /*size=*/output_bytes_per_row);
        noc.async_writes_flushed();

        cb_in.pop_front(1);
    }
    noc.async_write_barrier();
}
