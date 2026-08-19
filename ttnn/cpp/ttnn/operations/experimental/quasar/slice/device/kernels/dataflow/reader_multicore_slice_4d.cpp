// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * TTNN Slice Operation - Multi-Core Reader Kernel (4D Support)
 *
 * This kernel handles the input data reading phase of the slice operation for multi-core
 * execution, implementing slice logic for 1D, 2D, 3D, and 4D tensors with work distribution
 * across multiple cores for improved performance.
 *
 * Metal 2.0: named kernel arguments + named tensor binding (tensor::in).
 *
 * Key Responsibilities:
 * - Read assigned portion of input tensor data from DRAM using TensorAccessor
 * - Apply slice logic (start, end, step) for all dimensions (N, D, H, W)
 * - Handle different data types with proper element size calculations
 * - Process assigned rows for this core based on work distribution
 * - Output sliced rows to circular buffer for writer kernel consumption
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
    uint32_t input_w = get_arg(args::input_w);
    uint32_t input_h = get_arg(args::input_h);
    uint32_t input_d = get_arg(args::input_d);
    uint32_t input_n = get_arg(args::input_n);
    uint32_t output_w = get_arg(args::output_w);
    uint32_t output_h = get_arg(args::output_h);
    uint32_t output_d = get_arg(args::output_d);
    uint32_t output_n = get_arg(args::output_n);
    uint32_t slice_start_w = get_arg(args::slice_start_w);
    uint32_t slice_end_w = get_arg(args::slice_end_w);
    uint32_t slice_step_w = get_arg(args::slice_step_w);
    uint32_t slice_start_h = get_arg(args::slice_start_h);
    uint32_t slice_end_h = get_arg(args::slice_end_h);
    uint32_t slice_step_h = get_arg(args::slice_step_h);
    uint32_t slice_start_d = get_arg(args::slice_start_d);
    uint32_t slice_end_d = get_arg(args::slice_end_d);
    uint32_t slice_step_d = get_arg(args::slice_step_d);
    uint32_t slice_start_n = get_arg(args::slice_start_n);
    uint32_t slice_end_n = get_arg(args::slice_end_n);
    uint32_t slice_step_n = get_arg(args::slice_step_n);
    uint32_t element_size = get_arg(args::element_size);
    uint32_t num_rows_for_this_core = get_arg(args::num_rows_for_this_core);
    uint32_t start_row_for_this_core = get_arg(args::start_row_for_this_core);

    // Compile-time arguments
    constexpr uint32_t compile_time_element_size = get_arg(args::compile_time_element_size);

    // Calculate sizes - working with rows, not tiles
    uint32_t input_bytes_per_row = input_w * element_size;  // Dynamic element size
    uint32_t output_bytes_per_row = output_w * element_size;

    // Set up TensorAccessor for input data - use row size as page size
    const auto s0 = TensorAccessor(tensor::in);

    Noc noc;
    // Create DataflowBuffer for Device 2.0 API
    DataflowBuffer cb_out(dfb::cb_out);

    // Multi-core work distribution: this core processes rows [start_row_for_this_core, start_row_for_this_core +
    // num_rows_for_this_core) We need to map these logical output row indices back to the corresponding (n,d,h)
    // coordinates

    uint32_t rows_processed = 0;
    uint32_t current_logical_row = 0;
    bool found_start = false;

    // Set loop bounds based on tensor rank
    uint32_t n_start = (tensor_rank >= 4) ? slice_start_n : 0;
    uint32_t n_end = (tensor_rank >= 4) ? slice_end_n : 1;
    uint32_t n_step = (tensor_rank >= 4) ? slice_step_n : 1;

    uint32_t d_start = (tensor_rank >= 3) ? slice_start_d : 0;
    uint32_t d_end = (tensor_rank >= 3) ? slice_end_d : 1;
    uint32_t d_step = (tensor_rank >= 3) ? slice_step_d : 1;

    uint32_t h_start = (tensor_rank >= 2) ? slice_start_h : 0;
    uint32_t h_end = (tensor_rank >= 2) ? slice_end_h : 1;
    uint32_t h_step = (tensor_rank >= 2) ? slice_step_h : 1;

    // Multi-dimensional slicing with nested loops
    // 4D: Batch dimension loop
    for (uint32_t n = n_start; n < n_end && rows_processed < num_rows_for_this_core; n += n_step) {
        // 3D: Depth dimension loop
        for (uint32_t d = d_start; d < d_end && rows_processed < num_rows_for_this_core; d += d_step) {
            // 2D: Height dimension loop
            for (uint32_t h = h_start; h < h_end && rows_processed < num_rows_for_this_core; h += h_step) {
                // Check if this logical row should be processed by this core
                if (!found_start) {
                    if (current_logical_row == start_row_for_this_core) {
                        found_start = true;
                    } else {
                        current_logical_row++;
                        if (tensor_rank == 1) {
                            break;  // For 1D, exit after checking once
                        }
                        continue;
                    }
                }

                if (rows_processed >= num_rows_for_this_core) {
                    break;
                }

                // Calculate input row index based on tensor rank
                uint32_t input_row_idx;
                if (tensor_rank == 1) {
                    input_row_idx = 0;  // Single row for 1D
                } else if (tensor_rank == 2) {
                    input_row_idx = h;
                } else if (tensor_rank == 3) {
                    input_row_idx = d * input_h + h;
                } else {  // 4D
                    input_row_idx = n * input_d * input_h + d * input_h + h;
                }

                cb_out.reserve_back(1);
                uint32_t l1_write_addr = cb_out.get_write_ptr();

                // noc_async_read_sharded splits the read across shards for B/W-sharded inputs;
                // falls through to a single noc_async_read for interleaved / HEIGHT-sharded.
                tt::data_movement::common::noc_async_read_sharded(
                    noc, l1_write_addr, s0, input_row_idx, /*offset=*/0, /*size=*/input_bytes_per_row);
                noc.async_read_barrier();

                // Now slice the row according to width slice parameters
                // Copy sliced elements to the beginning of the buffer
                if (slice_start_w != 0 || slice_step_w != 1 || slice_end_w != input_w) {
                    // Need to reorganize the data in the buffer
                    volatile tt_l1_ptr uint8_t* src_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write_addr);
                    volatile tt_l1_ptr uint8_t* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write_addr);

                    uint32_t out_col = 0;
                    for (uint32_t input_col = slice_start_w; input_col < slice_end_w && out_col < output_w;
                         input_col += slice_step_w) {
                        // Copy element by element for the slice
                        for (uint32_t byte_idx = 0; byte_idx < element_size; ++byte_idx) {
                            dst_ptr[out_col * element_size + byte_idx] = src_ptr[input_col * element_size + byte_idx];
                        }
                        out_col++;
                    }
                }

                cb_out.push_back(1);
                rows_processed++;
                current_logical_row++;

                // Early exit for 1D tensors
                if (tensor_rank == 1) {
                    break;
                }
            }
            // Early exit for 2D tensors
            if (tensor_rank <= 2) {
                break;
            }
        }
        // Early exit for 3D tensors
        if (tensor_rank <= 3) {
            break;
        }
    }
}
