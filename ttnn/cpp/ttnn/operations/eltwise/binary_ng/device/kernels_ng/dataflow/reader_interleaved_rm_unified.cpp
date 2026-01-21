// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    uint32_t index = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(index++);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(index++);
    const uint32_t aD = get_arg_val<uint32_t>(index++);
    const uint32_t aN = get_arg_val<uint32_t>(index++);
    const uint32_t aC = get_arg_val<uint32_t>(index++);
    const uint32_t aHt = get_arg_val<uint32_t>(index++);  // A Height (Elements)
    const uint32_t aWt = get_arg_val<uint32_t>(index++);  // A Width (Elements)

    const uint32_t src_addr_b = get_arg_val<uint32_t>(index++);
    const uint32_t bD = get_arg_val<uint32_t>(index++);
    const uint32_t bN = get_arg_val<uint32_t>(index++);
    const uint32_t bC = get_arg_val<uint32_t>(index++);
    const uint32_t bHt = get_arg_val<uint32_t>(index++);  // B Height (Elements)
    const uint32_t bWt = get_arg_val<uint32_t>(index++);  // B Width (Elements)

    const uint32_t current_row_start = get_arg_val<uint32_t>(index++);
    const uint32_t num_rows = get_arg_val<uint32_t>(index++);       // Rows to process in this tile (e.g. 32)
    const uint32_t page_size_arg = get_arg_val<uint32_t>(index++);  // Output Row Width (Bytes)

    constexpr auto cb_id_src = tt::CBIndex::c_0;
    constexpr auto cb_id_src_b = tt::CBIndex::c_1;

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto src_b_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    // --- Geometry & Size Calculations ---
    constexpr uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const uint32_t tile_hw = get_tile_hw(cb_id_src);
    constexpr uint32_t element_size = src_tile_bytes / tile_hw;
    const uint32_t element_size_aligned = ((element_size + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;
    const uint32_t tile_bytes = tile_hw * element_size;

    const uint32_t full_page_size = ((page_size_arg + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

    const uint32_t row_width_elements = full_page_size / element_size;

    // Output Height (Logical Block Size for Broadcast)
    // If we broadcast 1 -> H, the pattern repeats every max(H_a, H_b).
    const uint32_t outHt = (aHt > bHt) ? aHt : bHt;

    // --- Broadcast Mode Logic ---
    // Enable Row Broadcast if H=1 and the Target Height > 1
    // This handles both (1 -> N) and (1 -> 1 if N=1) correctly via the filling loop logic.
    bool is_a_row_bcast = (aHt == 1 && outHt > 1);
    bool is_a_col_bcast = (aWt == 1 && row_width_elements > 1);

    bool is_b_row_bcast = (bHt == 1 && outHt > 1);
    bool is_b_col_bcast = (bWt == 1 && row_width_elements > 1);

    const uint32_t page_size_a = is_a_col_bcast ? element_size : full_page_size;
    const uint32_t page_size_b = is_b_col_bcast ? element_size : full_page_size;

    const auto src = TensorAccessor(src_args, src_addr, page_size_a);
    const auto src_b = TensorAccessor(src_b_args, src_addr_b, page_size_b);

    // --- Tiling/Chunking Logic ---
    const uint32_t div = (row_width_elements + tile_hw - 1) / tile_hw;
    const uint32_t num_batches = dst_num_tiles / div;

    // Max bytes per horizontal chunk 't'
    uint32_t stride_size_bytes = full_page_size > tile_bytes ? tile_bytes : full_page_size;

    uint32_t current_row_offset = 0;

    for (uint32_t b = 0; b < num_batches; ++b) {
        uint32_t bytes_left = full_page_size;

        for (uint32_t t = 0; t < div; t++) {
            cb_reserve_back(cb_id_src, 1);
            cb_reserve_back(cb_id_src_b, 1);

            uint32_t l1_write_addr_src = get_write_ptr(cb_id_src);
            uint32_t l1_write_addr_src_b = get_write_ptr(cb_id_src_b);

            uint32_t current_chunk_bytes = stride_size_bytes < bytes_left ? stride_size_bytes : bytes_left;
            uint32_t current_chunk_elements = current_chunk_bytes / element_size;

            uint32_t a_read_bytes =
                is_a_col_bcast ? element_size_aligned
                               : ((current_chunk_bytes + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;
            uint32_t b_read_bytes =
                is_b_col_bcast ? element_size_aligned
                               : ((current_chunk_bytes + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

            // =========================================================
            // PHASE 1: ROW BROADCAST HANDLING
            // =========================================================

            // --- Input A Row Broadcast ---
            if (is_a_row_bcast) {
                uint32_t current_row_global = current_row_start + current_row_offset;
                uint32_t rows_remaining = num_rows;

                while (rows_remaining > 0) {
                    // Calculate Index: Map current output row to A's dimension.
                    // If multi-batch, A index advances every 'outHt' rows.
                    uint32_t a_index = 0;
                    if ((aD * aN * aC) > 1) {
                        a_index = (current_row_global / outHt) * aHt;
                    }

                    // Boundary Check: Do not cross a Batch/Channel boundary in this smear op
                    uint32_t rows_in_current_block = outHt - (current_row_global % outHt);
                    uint32_t rows_to_fill =
                        (rows_remaining < rows_in_current_block) ? rows_remaining : rows_in_current_block;

                    uint32_t t_offset = is_a_col_bcast ? 0 : (stride_size_bytes * t);
                    uint64_t addr = get_noc_addr(a_index, src) + t_offset;

                    noc_async_read(addr, l1_write_addr_src, a_read_bytes);
                    noc_async_read_barrier();

                    if (is_a_col_bcast) {  // Scalar Case
                        FILL_TILE_WITH_FIRST_COLUMN_RM(l1_write_addr_src, row_width_elements);
                    }

                    FILL_TILE_WITH_FIRST_ROW_RM(l1_write_addr_src, row_width_elements, rows_to_fill);
                    noc_async_read_barrier();

                    l1_write_addr_src += rows_to_fill * row_width_elements * element_size;
                    current_row_global += rows_to_fill;
                    rows_remaining -= rows_to_fill;
                }
            }

            // --- Input B Row Broadcast ---
            if (is_b_row_bcast) {
                uint32_t current_row_global = current_row_start + current_row_offset;
                uint32_t rows_remaining = num_rows;

                while (rows_remaining > 0) {
                    uint32_t b_index = 0;
                    if ((bD * bN * bC) > 1) {
                        b_index = (current_row_global / outHt) * bHt;
                    }

                    // Boundary Check: Do not cross a Batch/Channel boundary
                    uint32_t rows_in_current_block = outHt - (current_row_global % outHt);
                    uint32_t rows_to_fill =
                        (rows_remaining < rows_in_current_block) ? rows_remaining : rows_in_current_block;

                    uint32_t t_offset = is_b_col_bcast ? 0 : (stride_size_bytes * t);
                    uint64_t addr = get_noc_addr(b_index, src_b) + t_offset;

                    noc_async_read(addr, l1_write_addr_src_b, b_read_bytes);
                    noc_async_read_barrier();

                    if (is_b_col_bcast) {  // Scalar case
                        FILL_TILE_WITH_FIRST_COLUMN_RM(l1_write_addr_src_b, row_width_elements);
                    }

                    FILL_TILE_WITH_FIRST_ROW_RM(l1_write_addr_src_b, row_width_elements, rows_to_fill);
                    noc_async_read_barrier();

                    l1_write_addr_src_b += rows_to_fill * row_width_elements * element_size;
                    current_row_global += rows_to_fill;
                    rows_remaining -= rows_to_fill;
                }
            }

            // =========================================================
            // PHASE 2: PER-ROW PROCESSING (Col Bcast & Dense)
            // =========================================================

            uint32_t ptr_a = get_write_ptr(cb_id_src);
            uint32_t ptr_b = get_write_ptr(cb_id_src_b);

            if (!is_a_row_bcast || !is_b_row_bcast) {
                for (uint32_t i = 0; i < num_rows; i++) {
                    uint32_t current_row_global = current_row_start + current_row_offset + i;

                    // --- Input A ---
                    if (!is_a_row_bcast) {
                        if (is_a_col_bcast) {
                            uint32_t total = aN * aC * aD * aHt;
                            uint32_t a_idx = ((aD * aN * aC) > 1) ? (current_row_global % total) : current_row_global;

                            uint64_t addr = get_noc_addr(a_idx, src);
                            noc_async_read(addr, ptr_a, element_size_aligned);
                            noc_async_read_barrier();
                            FILL_TILE_WITH_FIRST_COLUMN_RM(ptr_a, current_chunk_elements);
                        } else {
                            uint64_t addr = get_noc_addr(current_row_global, src) + stride_size_bytes * t;
                            noc_async_read(addr, ptr_a, a_read_bytes);
                            noc_async_read_barrier();
                        }
                        ptr_a += current_chunk_bytes;
                    }

                    // --- Input B ---
                    if (!is_b_row_bcast) {
                        if (is_b_col_bcast) {
                            uint32_t total = bN * bC * bD * bHt;
                            uint32_t b_idx = ((bD * bN * bC) > 1) ? (current_row_global % total) : current_row_global;

                            uint64_t addr = get_noc_addr(b_idx, src_b);
                            noc_async_read(addr, ptr_b, element_size_aligned);
                            noc_async_read_barrier();
                            FILL_TILE_WITH_FIRST_COLUMN_RM(ptr_b, current_chunk_elements);
                        } else {
                            uint64_t addr = get_noc_addr(current_row_global, src_b) + stride_size_bytes * t;
                            noc_async_read(addr, ptr_b, b_read_bytes);
                            noc_async_read_barrier();
                        }
                        ptr_b += current_chunk_bytes;
                    }
                }
            }
            noc_async_read_barrier();

            bytes_left -= current_chunk_bytes;
            cb_push_back(cb_id_src, 1);
            cb_push_back(cb_id_src_b, 1);
        }
        current_row_offset += num_rows;
    }
}
