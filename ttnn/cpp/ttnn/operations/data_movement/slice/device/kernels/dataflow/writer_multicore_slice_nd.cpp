// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
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
#include "dataflow_api.h"

void kernel_main() {
    // Runtime arguments - first get basic parameters
    uint32_t rt_args_idx = 0;
    uint32_t dst_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t tensor_rank = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t element_size = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_rows_for_this_core = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t start_row_for_this_core = get_arg_val<uint32_t>(rt_args_idx++);

    // Compile-time arguments
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t compile_time_element_size = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    // Get dimension arrays from runtime arguments
    // Layout: output_dims[rank]

    // Read output dimensions
    volatile tt_l1_ptr uint32_t* output_dims = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(rt_args_idx++));

    // Calculate sizes - working with rows, not tiles
    uint32_t output_bytes_per_row = output_dims[tensor_rank - 1] * element_size;

    // Set up TensorAccessor for output data - use row size as page size
    const auto s0 = TensorAccessor(dst_args, dst_addr, output_bytes_per_row);

    // Multi-core work distribution: this core writes rows starting from start_row_for_this_core
    // Write each row from circular buffer to output tensor at the correct logical position
    for (uint32_t local_row = 0; local_row < num_rows_for_this_core; ++local_row) {
        cb_wait_front(cb_id_in, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_in);

        // Calculate global output row index for this local row
        uint32_t global_output_row = start_row_for_this_core + local_row;

        // Calculate output address for this row
        uint64_t output_row_noc_addr = get_noc_addr(global_output_row, s0);

        // Write the complete row to output tensor
        noc_async_write(l1_read_addr, output_row_noc_addr, output_bytes_per_row);
        noc_async_writes_flushed();

        cb_pop_front(cb_id_in, 1);
    }
    noc_async_write_barrier();
}
