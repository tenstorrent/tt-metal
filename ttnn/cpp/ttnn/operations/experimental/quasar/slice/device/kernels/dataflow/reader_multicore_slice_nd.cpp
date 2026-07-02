// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * TTNN Slice Operation - Multi-Core Reader Kernel (N-Dimensional Support)
 *
 * This kernel handles the input data reading phase of the slice operation for multi-core
 * execution, implementing slice logic for N-dimensional tensors (1D, 2D, 3D, 4D, 5D, etc.)
 * with work distribution across multiple cores for improved performance.
 *
 * Metal 2.0: named tensor binding (tensor::in); the per-program dim/slice arrays
 * (input_dims, output_dims, slice_starts/ends/steps — identical for every core) arrive as
 * common runtime varargs; only the per-core work split (num_rows / start_row) is a per-core RTA.
 *
 * Common-vararg layout (each block tensor_rank long):
 *   [0*R, 1*R)  input_dims
 *   [1*R, 2*R)  output_dims
 *   [2*R, 3*R)  slice_starts
 *   [3*R, 4*R)  slice_ends
 *   [4*R, 5*R)  slice_steps
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

    // Dimension / slice arrays from common runtime varargs (identical for all cores).
    uint32_t input_dims[tensor_rank];
    uint32_t output_dims[tensor_rank];
    uint32_t slice_starts[tensor_rank];
    uint32_t slice_ends[tensor_rank];
    uint32_t slice_steps[tensor_rank];
    for (uint32_t i = 0; i < tensor_rank; ++i) {
        input_dims[i] = get_common_vararg(0 * tensor_rank + i);
        output_dims[i] = get_common_vararg(1 * tensor_rank + i);
        slice_starts[i] = get_common_vararg(2 * tensor_rank + i);
        slice_ends[i] = get_common_vararg(3 * tensor_rank + i);
        slice_steps[i] = get_common_vararg(4 * tensor_rank + i);
    }

    // Calculate sizes - working with rows, not tiles
    uint32_t input_bytes_per_row = input_dims[tensor_rank - 1] * element_size;
    uint32_t output_bytes_per_row = output_dims[tensor_rank - 1] * element_size;

    // Set up TensorAccessor for input data - use row size as page size
    const auto s0 = TensorAccessor(tensor::in);

    Noc noc;
    // Create DataflowBuffer for Device 2.0 API
    DataflowBuffer cb_out(dfb::cb_out);

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

            cb_out.reserve_back(1);
            uint32_t l1_write_addr = cb_out.get_write_ptr();

            // noc_async_read_sharded splits the read across shards for B/W-sharded inputs;
            // falls through to a single noc_async_read for interleaved / HEIGHT-sharded.
            tt::data_movement::common::noc_async_read_sharded(
                noc, l1_write_addr, s0, input_row_idx, /*offset=*/0, /*size=*/input_bytes_per_row);
            noc.async_read_barrier();

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

            cb_out.push_back(1);
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
