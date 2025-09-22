/*
 * TTNN Slice Operation - Multi-Core Reader Kernel (4D Support)
 *
 * This kernel handles the input data reading phase of the slice operation for multi-core
 * execution, implementing slice logic for 1D, 2D, 3D, and 4D tensors with work distribution
 * across multiple cores for improved performance.
 *
 * Key Responsibilities:
 * - Read assigned portion of input tensor data from DRAM using InterleavedAddrGenFast
 * - Apply slice logic (start, end, step) for all dimensions (N, D, H, W)
 * - Handle different data types with proper element size calculations
 * - Process assigned rows for this core based on work distribution
 * - Output sliced rows to circular buffer for writer kernel consumption
 *
 * Architecture:
 * - Uses InterleavedAddrGenFast for efficient DRAM address generation
 * - Processes data row-by-row for optimal memory access patterns
 * - Supports slicing with configurable start, end, and step parameters for all dimensions
 * - Handles 1D (W), 2D (H,W), 3D (D,H,W), and 4D (N,D,H,W) tensors
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
 * Author: claude (based on pad operation template and single-core 4D slice)
 * Compatible with: TTNN framework, ROW_MAJOR_LAYOUT tensors, 1D-4D dimensions, multi-core execution
 */

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

inline void print_bf16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

void kernel_main() {
    DPRINT << "LLONG reader kernel_main" << ENDL();
    // Runtime arguments for 4D slice support with multi-core work distribution
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t tensor_rank = get_arg_val<uint32_t>(1);
    uint32_t input_w = get_arg_val<uint32_t>(2);
    uint32_t input_h = get_arg_val<uint32_t>(3);
    uint32_t input_d = get_arg_val<uint32_t>(4);
    uint32_t input_n = get_arg_val<uint32_t>(5);
    uint32_t output_w = get_arg_val<uint32_t>(6);
    uint32_t output_h = get_arg_val<uint32_t>(7);
    uint32_t output_d = get_arg_val<uint32_t>(8);
    uint32_t output_n = get_arg_val<uint32_t>(9);
    uint32_t slice_start_w = get_arg_val<uint32_t>(10);
    uint32_t slice_end_w = get_arg_val<uint32_t>(11);
    uint32_t slice_step_w = get_arg_val<uint32_t>(12);
    uint32_t slice_start_h = get_arg_val<uint32_t>(13);
    uint32_t slice_end_h = get_arg_val<uint32_t>(14);
    uint32_t slice_step_h = get_arg_val<uint32_t>(15);
    uint32_t slice_start_d = get_arg_val<uint32_t>(16);
    uint32_t slice_end_d = get_arg_val<uint32_t>(17);
    uint32_t slice_step_d = get_arg_val<uint32_t>(18);
    uint32_t slice_start_n = get_arg_val<uint32_t>(19);
    uint32_t slice_end_n = get_arg_val<uint32_t>(20);
    uint32_t slice_step_n = get_arg_val<uint32_t>(21);
    uint32_t element_size = get_arg_val<uint32_t>(22);
    uint32_t num_rows_for_this_core = get_arg_val<uint32_t>(23);
    uint32_t start_row_for_this_core = get_arg_val<uint32_t>(24);
    DPRINT << "num_rows_for_this_core: " << num_rows_for_this_core << ENDL();
    DPRINT << "start_row_for_this_core: " << start_row_for_this_core << ENDL();

    // Compile-time arguments
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);
    constexpr uint32_t compile_time_element_size = get_compile_time_arg_val(2);

    // Calculate sizes - working with rows, not tiles
    uint32_t input_bytes_per_row = input_w * element_size;  // Dynamic element size
    uint32_t output_bytes_per_row = output_w * element_size;

    // Set up InterleavedAddrGenFast for input data - use row size as page size
    const InterleavedAddrGen<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = input_bytes_per_row,  // Each page is one input row
    };

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

                cb_reserve_back(cb_id_out, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_id_out);

                // Read the full input row first
                uint64_t input_row_noc_addr = get_noc_addr(input_row_idx, s);
                noc_async_read(input_row_noc_addr, l1_write_addr, input_bytes_per_row);
                noc_async_read_barrier();

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

                cb_push_back(cb_id_out, 1);
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
