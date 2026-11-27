// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// page_copy reader (NCRISC, NOC0). Each output page of this core's range is one input page,
// fetched into cb_output_pages a block of block_pages at a time.
//
// No kernel_lib helper covers this step: the helpers that copy tiles are compute-side and round-trip
// values through DEST, which this op must not do (op_design.md, "Helpers considered and rejected").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_concat_tile_rows.hpp"

namespace {

// Common runtime args; layout mirrors PAGE_READER_COMMON_ARGS in g6_concat_program_descriptor.py.
constexpr uint32_t CRTA_BLOCK_PAGES = 0;
constexpr uint32_t CRTA_OUT_WT = 1;
constexpr uint32_t CRTA_OUT_HT = 2;
constexpr uint32_t CRTA_LAST_ROW_VALID_ROWS = 3;
constexpr uint32_t CRTA_NUM_INPUTS = 4;
constexpr uint32_t CRTA_FIRST_INPUT = 5;
constexpr uint32_t CRTA_WORDS_PER_INPUT = 2;  // base address, pages of the input per outer slice

FORCE_INLINE uint32_t input_base_address(uint32_t input_idx) {
    return get_common_arg_val<uint32_t>(CRTA_FIRST_INPUT + input_idx * CRTA_WORDS_PER_INPUT);
}

FORCE_INLINE uint32_t input_pages_per_slice(uint32_t input_idx) {
    return get_common_arg_val<uint32_t>(CRTA_FIRST_INPUT + input_idx * CRTA_WORDS_PER_INPUT + 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_pages = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t trim_rows_max = get_compile_time_arg_val(3);
    constexpr auto input_args = TensorAccessorArgs<4>();

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    // Walk cursor (o, k, c') of start_page: output page o * C + Coff_k + c' is input k's page o * C_k + c'.
    uint32_t slice = get_arg_val<uint32_t>(2);
    uint32_t input_idx = get_arg_val<uint32_t>(3);
    uint32_t page_in_slice = get_arg_val<uint32_t>(4);

    const uint32_t block_pages = get_common_arg_val<uint32_t>(CRTA_BLOCK_PAGES);
    const uint32_t num_inputs = get_common_arg_val<uint32_t>(CRTA_NUM_INPUTS);
    g6_concat::OutputRowCursor rows(
        start_page,
        get_common_arg_val<uint32_t>(CRTA_OUT_WT),
        get_common_arg_val<uint32_t>(CRTA_OUT_HT),
        get_common_arg_val<uint32_t>(CRTA_LAST_ROW_VALID_ROWS));

    uint32_t base_address = input_base_address(input_idx);
    uint32_t pages_per_slice = input_pages_per_slice(input_idx);
    uint32_t input_page = slice * pages_per_slice + page_in_slice;

    const uint32_t num_blocks_this_core = (core_pages + block_pages - 1) / block_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const bool last_block = block_idx + 1 == num_blocks_this_core;
        const uint32_t block_extent = last_block ? core_pages - block_idx * block_pages : block_pages;

        {
            DeviceZoneScopedN("G6C-RESERVE");
            cb_reserve_back(cb_output_pages, block_pages);
        }
        {
            DeviceZoneScopedN("G6C-FETCH");
            // page_fetch_block
            uint32_t slot_addr = get_write_ptr(cb_output_pages);
            for (uint32_t page = 0; page < block_extent; ++page) {
                const auto input = TensorAccessor(input_args, base_address, page_bytes);
                const uint32_t valid_rows = rows.valid_rows();
                if (g6_concat::is_trimmed(valid_rows, trim_rows_max)) {
                    // Source and destination offsets are identical, so the read stays alignment-legal.
                    g6_concat::for_each_valid_row_piece<face_row_bytes>(
                        valid_rows, [&](uint32_t offset_bytes, uint32_t size_bytes) {
                            noc_async_read(
                                input.get_noc_addr(input_page, offset_bytes), slot_addr + offset_bytes, size_bytes);
                        });
                } else {
                    noc_async_read_page(input_page, input, slot_addr);
                }
                slot_addr += page_bytes;
                rows.advance();

                ++input_page;
                if (++page_in_slice == pages_per_slice) {
                    page_in_slice = 0;
                    if (++input_idx == num_inputs) {
                        input_idx = 0;
                        ++slice;
                    }
                    base_address = input_base_address(input_idx);
                    pages_per_slice = input_pages_per_slice(input_idx);
                    input_page = slice * pages_per_slice;
                }
            }
            noc_async_read_barrier();
        }
        // Nominal count on the ragged last block keeps the ring cycle closed.
        cb_push_back(cb_output_pages, block_pages);
    }
}
