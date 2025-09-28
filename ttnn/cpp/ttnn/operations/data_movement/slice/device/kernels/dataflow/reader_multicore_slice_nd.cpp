// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * TTNN Slice Operation - Multi-Core Reader Kernel (N-Dimensional Support)
 *
 * This kernel handles the input data reading phase of the slice operation for multi-core
 * execution, implementing slice logic for N-dimensional tensors (1D, 2D, 3D, 4D, 5D, etc.)
 * with work distribution across multiple cores for improved performance.
 *
 * Key Responsibilities:
 * - Read assigned portion of input tensor data from DRAM using TensorAccessor
 * - Apply slice logic (start, end, step) for all dimensions dynamically
 * - Handle different data types with proper element size calculations
 * - Process assigned rows for this core based on work distribution
 * - Output sliced rows to circular buffer for writer kernel consumption
 *
 * Architecture:
 * - Uses TensorAccessor for efficient DRAM address generation
 * - Processes data row-by-row for optimal memory access patterns
 * - Supports slicing with configurable start, end, and step parameters for all dimensions
 * - Handles 1D, 2D, 3D, 4D, 5D+ tensors with dynamic nested loops
 * - Multi-core work distribution: each core processes a subset of output rows
 *
 * Memory Management:
 * - DRAM alignment: 32-byte boundaries for memory controller optimization
 * - L1 alignment: 16-byte boundaries for L1 cache efficiency
 * - Circular buffer: Double buffering for continuous data flow
 *
 * Data Type Support:
 * - Element size determined at compile time for performance
 * - Dynamic element size passed as runtime argument for flexibility
 *
 * Performance Optimizations:
 * - Minimal branching in inner loops for consistent execution
 * - Efficient memory copy operations using NOC async transfers
 * - Cache-friendly access patterns aligned to memory hierarchy
 * - Parallel processing with load balancing across multiple cores
 *
 * N-Dimensional Processing:
 * - Dynamic nested loops based on tensor rank
 * - Generic address calculation using stride-based indexing
 * - Row concept: for rank R, rows = product(dims[0:R-1]), width = dims[R-1]
 *
 * Compatible with: TTNN framework, ROW_MAJOR_LAYOUT tensors, 1D-ND dimensions, multi-core execution
 */

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Runtime arguments - first get basic parameters
    uint32_t rt_args_idx = 0;
    uint32_t src_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t tensor_rank = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t element_size = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_rows_for_this_core = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t start_row_for_this_core = get_arg_val<uint32_t>(rt_args_idx++);

    // Compile-time arguments
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t compile_time_element_size = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    // Get dimension arrays from runtime arguments
    // Layout: input_dims[rank], output_dims[rank], slice_starts[rank], slice_ends[rank], slice_steps[rank]
    // Read input dimensions
    volatile tt_l1_ptr uint32_t* input_dims = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(rt_args_idx));
    rt_args_idx += tensor_rank;

    // Read output dimensions
    volatile tt_l1_ptr uint32_t* output_dims = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(rt_args_idx));
    rt_args_idx += tensor_rank;

    // Read slice parameters
    volatile tt_l1_ptr uint32_t* slice_starts = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(rt_args_idx));
    rt_args_idx += tensor_rank;

    volatile tt_l1_ptr uint32_t* slice_ends = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(rt_args_idx));
    rt_args_idx += tensor_rank;

    volatile tt_l1_ptr uint32_t* slice_steps = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(rt_args_idx));

    // Calculate sizes - working with rows, not tiles
    uint32_t input_bytes_per_row = input_dims[tensor_rank - 1] * element_size;
    uint32_t output_bytes_per_row = output_dims[tensor_rank - 1] * element_size;

    // Set up TensorAccessor for input data - use row size as page size
    const auto s0 = TensorAccessor(src_args, src_addr, input_bytes_per_row);

    // Multi-core work distribution using iterative approach with explicit coordinate tracking
    // Track current position in N-dimensional space
    uint32_t coords[16];  // Support up to 16 dimensions (reasonable limit)
    for (uint32_t i = 0; i < tensor_rank; ++i) {
        coords[i] = slice_starts[i];
    }

    uint32_t rows_processed = 0;
    uint32_t current_logical_row = 0;
    bool found_start = false;
    bool more_iterations = true;

    while (more_iterations && rows_processed < num_rows_for_this_core) {
        // Check if this logical row should be processed by this core
        bool should_process_row = false;
        if (!found_start) {
            if (current_logical_row == start_row_for_this_core) {
                found_start = true;
                should_process_row = true;
            }
        } else {
            should_process_row = true;
        }

        if (should_process_row && rows_processed < num_rows_for_this_core) {
            // Calculate input row index from coordinates (exclude last dimension)
            uint32_t input_row_idx = 0;
            if (tensor_rank > 1) {
                for (int32_t dim = 0; dim < (int32_t)(tensor_rank - 1); ++dim) {
                    uint32_t stride = 1;
                    for (int32_t d = dim + 1; d < (int32_t)(tensor_rank - 1); ++d) {
                        stride *= input_dims[d];
                    }
                    input_row_idx += coords[dim] * stride;
                }
            }

            cb_reserve_back(cb_id_out, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_id_out);

            // Read the full input row first
            uint64_t input_row_noc_addr = get_noc_addr(input_row_idx, s0);
            noc_async_read(input_row_noc_addr, l1_write_addr, input_bytes_per_row);
            noc_async_read_barrier();

            // Now slice the row according to width slice parameters (last dimension)
            uint32_t last_dim = tensor_rank - 1;
            if (slice_starts[last_dim] != 0 || slice_steps[last_dim] != 1 ||
                slice_ends[last_dim] != input_dims[last_dim]) {
                // Need to reorganize the data in the buffer
                volatile tt_l1_ptr uint8_t* src_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write_addr);
                volatile tt_l1_ptr uint8_t* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write_addr);

                uint32_t out_col = 0;
                for (uint32_t input_col = slice_starts[last_dim];
                     input_col < slice_ends[last_dim] && out_col < output_dims[last_dim];
                     input_col += slice_steps[last_dim]) {
                    // Copy element by element for the slice
                    for (uint32_t byte_idx = 0; byte_idx < element_size; ++byte_idx) {
                        dst_ptr[out_col * element_size + byte_idx] = src_ptr[input_col * element_size + byte_idx];
                    }
                    out_col++;
                }
            }

            cb_push_back(cb_id_out, 1);
            rows_processed++;
        }

        current_logical_row++;

        // Advance to next coordinate combination (skip last dimension)
        bool carry = true;
        for (int32_t dim = (int32_t)(tensor_rank - 2); dim >= 0 && carry; --dim) {
            coords[dim] += slice_steps[dim];
            if (coords[dim] < slice_ends[dim]) {
                carry = false;
            } else {
                coords[dim] = slice_starts[dim];
            }
        }
        if (carry) {
            more_iterations = false;
        }

        // Early exit for 1D tensors
        if (tensor_rank == 1) {
            break;
        }
    }
}
