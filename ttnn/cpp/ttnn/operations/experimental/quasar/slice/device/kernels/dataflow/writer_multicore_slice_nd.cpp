// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * TTNN Slice Operation - Multi-Core Writer Kernel (N-Dimensional Support)
 *
 * This kernel handles the output data writing phase of the slice operation for multi-core
 * execution, writing sliced tensor data from circular buffer to output tensor memory.
 * Supports 1D, 2D, 3D, 4D, 5D, etc. tensors with work distribution across cores.
 *
 * Metal 2.0: named tensor binding (tensor::out); the per-program output_dims array
 * (identical for every core) arrives as a common runtime vararg; only the per-core work
 * split (num_rows / start_row) is a per-core RTA.
 *
 * Common-vararg layout: [0, tensor_rank) output_dims.
 *
 * Compatible with: TTNN framework, ROW_MAJOR_LAYOUT tensors, 1D-ND dimensions, multi-core execution
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t tensor_rank = get_arg(args::tensor_rank);
    constexpr uint32_t element_size = get_arg(args::element_size);

    // Per-core runtime arguments (named).
    uint32_t num_rows_for_this_core = get_arg(args::num_rows_for_this_core);
    uint32_t start_row_for_this_core = get_arg(args::start_row_for_this_core);

    // Output dimensions from common runtime varargs (identical for all cores).
    uint32_t output_dims[tensor_rank];
    for (uint32_t i = 0; i < tensor_rank; ++i) {
        output_dims[i] = get_common_vararg(i);
    }

    // Calculate sizes - working with rows, not tiles
    uint32_t output_bytes_per_row = output_dims[tensor_rank - 1] * element_size;

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
