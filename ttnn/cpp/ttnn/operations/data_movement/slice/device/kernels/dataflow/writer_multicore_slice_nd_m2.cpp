// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Metal 2.0 (ProgramSpec) port of writer_multicore_slice_nd.cpp. Used only by
 * SliceRmStrideSpecProgramFactory; the legacy file is still consumed unchanged by
 * SliceRmStrideProgramFactory::create_descriptor (reused by ccl/mesh_partition).
 * Logic, loop bounds and numeric paths are UNCHANGED; only the access mechanism moves to
 * named bindings:
 *   dst address (clean Buffer* RTA) -> ta::dst (TensorAccessor, Case-1 binding)
 *   CB id                           -> dfb::cb_in_out
 *   leading scalar RTAs / CTAs      -> named args (get_arg(args::...))
 *   the output_dims[tensor_rank] array -> runtime varargs (only output_dims[tensor_rank-1] is
 *     read; copied to a local stack array to preserve the legacy index, identical arithmetic).
 *
 * TTNN Slice Operation - Multi-Core Writer Kernel (N-Dimensional Support)
 *
 * This kernel handles the output data writing phase of the slice operation for multi-core
 * execution, writing sliced tensor data from circular buffer to output tensor memory.
 * Supports 1D, 2D, 3D, 4D, 5D, etc. tensors with work distribution across cores.
 *
 * Key Responsibilities:
 * - Read sliced data from circular buffer (produced by reader kernel)
 * - Write assigned portion of output data to DRAM using TensorAccessor
 * - Handle different tensor dimensions with proper address calculations
 * - Support different data types with proper element size handling
 * - Process assigned rows for this core based on work distribution
 *
 * Architecture:
 * - Uses TensorAccessor for efficient DRAM address generation
 * - Processes data row-by-row to match reader kernel output
 * - Simple sequential write pattern for optimal memory controller utilization
 * - Multi-core work distribution: each core writes a subset of output rows
 *
 * Memory Management:
 * - DRAM alignment: 32-byte boundaries for memory controller optimization
 * - L1 alignment: 16-byte boundaries for L1 cache efficiency
 * - Circular buffer: Double buffering synchronized with reader kernel
 *
 * Data Type Support:
 * - Element size determined at compile time for performance
 * - Dynamic element size passed as runtime argument for flexibility
 *
 * Performance Optimizations:
 * - Minimal synchronization overhead with reader kernel
 * - Efficient memory write operations using NOC async transfers
 * - Sequential access patterns optimized for DRAM controllers
 * - Parallel processing with load balancing across multiple cores
 *
 * N-Dimensional Processing:
 * - Row concept: for rank R, rows = product(dims[0:R-1]), width = dims[R-1]
 * - Each core writes a contiguous range of logical output rows
 * - Simple linear mapping from logical row index to physical address
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
    // Runtime arguments - first get basic parameters
    uint32_t tensor_rank = get_arg(args::tensor_rank);
    uint32_t element_size = get_arg(args::element_size);
    uint32_t num_rows_for_this_core = get_arg(args::num_rows_for_this_core);
    uint32_t start_row_for_this_core = get_arg(args::start_row_for_this_core);

    // Compile-time arguments
    constexpr uint32_t cb_id_in = dfb::cb_in_out;
    constexpr uint32_t compile_time_element_size = get_arg(args::compile_time_element_size);

    // Get dimension array from runtime varargs.
    // Layout: output_dims[rank] (copied into a local stack array; legacy caps rank at 16).
    uint32_t output_dims[16];
    for (uint32_t i = 0; i < tensor_rank; ++i) {
        output_dims[i] = get_vararg(i);
    }

    // Calculate sizes - working with rows, not tiles
    uint32_t output_bytes_per_row = output_dims[tensor_rank - 1] * element_size;

    // Set up TensorAccessor for output data - use row size as page size
    const auto s0 = TensorAccessor(ta::dst);

    Noc noc;
    // Create CircularBuffer for Device 2.0 API
    CircularBuffer cb_in(cb_id_in);

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
            l1_read_addr, s0, global_output_row, /*offset=*/0, /*size=*/output_bytes_per_row);
        noc.async_writes_flushed();

        cb_in.pop_front(1);
    }
    noc.async_write_barrier();
}
