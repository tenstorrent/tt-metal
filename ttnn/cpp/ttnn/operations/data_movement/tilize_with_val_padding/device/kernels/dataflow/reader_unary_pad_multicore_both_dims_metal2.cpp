// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of reader_unary_pad_multicore_both_dims.cpp.
//
// Reads padded blocks of a row-major interleaved input, splitting along both
// height and width into sub-blocks. Used by tilize multi_core_block factory.
//
// Bindings:
//   dfb::input                          — DFB endpoint (PRODUCER)
//   ta::input                           — TensorAccessor (input)
//   args::total_num_rows                — CTA
//   args::third_dim                     — CTA
//   args::tile_height                   — CTA
//   args::element_size                  — CTA
//   args::unpadded_X_size               — CTA
//   args::pad_value                     — RTA
//   args::width_size                    — RTA
//   args::start_row_id                  — RTA
//   args::start_column_id               — RTA
//   args::single_block_size_row         — RTA
//   args::single_block_size_col         — RTA
//   args::sub_block_width_size          — RTA
//   args::single_sub_block_size_row     — RTA

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Alignment-aware fill: writes 4 bytes at a time for the aligned middle,
// and uses element-sized writes for unaligned start/end to avoid rv32 unaligned faults.
// Assumption: if val_size < 4, multiple vals are packed into a single uint32_t val.
template <uint32_t val_size>
FORCE_INLINE void fill_with_val(uint32_t start_addr, uint32_t n_bytes, uint32_t val) {
    static_assert(val_size == sizeof(uint16_t) || val_size == sizeof(uint32_t), "Unsupported val_size");
    using IntType = std::conditional_t<(val_size == sizeof(uint16_t)), uint16_t, uint32_t>;

    const uint32_t end_addr = start_addr + n_bytes;
    const uint32_t start_addr_4B = (start_addr + 0x3) & 0xFFFFFFFC;
    const uint32_t end_addr_4B = end_addr & 0xFFFFFFFC;

    {
        auto* start_ptr_4B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_addr_4B);
        auto* end_ptr_4B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(end_addr_4B);
        for (auto* ptr = start_ptr_4B; ptr < end_ptr_4B; ++ptr) {
            *ptr = val;
        }
    }

    if constexpr (val_size < sizeof(uint32_t)) {
        auto* start_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(start_addr);
        auto* end_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(end_addr);
        auto* start_ptr_4B = reinterpret_cast<volatile tt_l1_ptr IntType*>(start_addr_4B);
        auto* end_ptr_4B = reinterpret_cast<volatile tt_l1_ptr IntType*>(end_addr_4B);
        const IntType val_ = static_cast<IntType>(val);

        for (auto* ptr = start_ptr; ptr < start_ptr_4B; ++ptr) {
            *ptr = val_;
        }
        for (auto* ptr = end_ptr_4B; ptr < end_ptr; ++ptr) {
            *ptr = val_;
        }
    }
}

void kernel_main() {
    constexpr auto total_num_rows = get_arg(args::total_num_rows);
    constexpr auto third_dim = get_arg(args::third_dim);
    constexpr auto tile_height = get_arg(args::tile_height);
    constexpr auto element_size = get_arg(args::element_size);
    constexpr auto unpadded_X_size = get_arg(args::unpadded_X_size);

    auto pad_value = get_arg(args::pad_value);
    auto width_size = get_arg(args::width_size);

    const auto s = TensorAccessor(ta::input);
    Noc noc;
    DataflowBuffer cb_in0(dfb::input);

    auto read_block = [&](uint32_t num_rows,
                          uint32_t start_row_id,
                          uint32_t start_column_id,
                          uint32_t block_width,
                          uint32_t size_2d,
                          uint32_t single_block_size) {
        uint32_t padding_rows = num_rows == 32 ? 0 : 32 - num_rows;
        bool has_rows = (num_rows + padding_rows) > 0;

        cb_in0.reserve_back(single_block_size * has_rows);
        uint32_t l1_write_addr = cb_in0.get_write_ptr();

        for (uint32_t k = start_row_id; k < start_row_id + num_rows; k++) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(
                s, dst, block_width, {.page_id = size_2d + k, .offset_bytes = start_column_id}, {.offset_bytes = 0});

            uint32_t prev_size = start_column_id;
            uint32_t this_block_size = unpadded_X_size - prev_size;
            if (this_block_size < block_width) {
                uint32_t to_pad = block_width - this_block_size;
                fill_with_val<element_size>(l1_write_addr + this_block_size, to_pad, pad_value);
            }

            noc.async_read_barrier();
            l1_write_addr += block_width;
        }

        for (uint32_t pad_row = 0; pad_row < padding_rows; pad_row++) {
            fill_with_val<element_size>(l1_write_addr, block_width, pad_value);
            l1_write_addr += block_width;
        }

        cb_in0.push_back(single_block_size * has_rows);
    };

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        uint32_t start_row_id = get_arg(args::start_row_id);
        uint32_t start_column_id = get_arg(args::start_column_id);
        uint32_t single_block_size_row_arg = get_arg(args::single_block_size_row);
        uint32_t single_block_size_col_arg = get_arg(args::single_block_size_col);
        uint32_t sub_block_width_size = get_arg(args::sub_block_width_size);
        uint32_t single_sub_block_size_row_arg = get_arg(args::single_sub_block_size_row);

        for (uint32_t b = 0; b < single_block_size_col_arg; b++) {
            uint32_t this_block_num_rows = tile_height;
            if (start_row_id + tile_height > total_num_rows) {
                this_block_num_rows = total_num_rows - start_row_id;
            }
            if (this_block_num_rows > 0) {
                for (uint32_t m = 0; m < width_size; m += sub_block_width_size) {
                    uint32_t start_column_id_u = start_column_id + m;
                    read_block(
                        this_block_num_rows,
                        start_row_id,
                        start_column_id_u,
                        sub_block_width_size,
                        size_2d,
                        single_sub_block_size_row_arg);
                }
            }
            start_row_id += tile_height;
        }
        size_2d += total_num_rows;
    }
}
