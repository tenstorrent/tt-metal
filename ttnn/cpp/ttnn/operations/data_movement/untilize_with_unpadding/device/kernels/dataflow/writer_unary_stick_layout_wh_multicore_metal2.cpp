// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of writer_unary_stick_layout_wh_multicore.cpp, which lives beside
// it. Ops ported to Metal 2.0 bind this file; the original serves the consumers still on the legacy
// API — the file sits in untilize_with_unpadding's directory but is also bound by
// data_movement/untilize's block factory, so it could not be converted in place. Until the last
// consumer migrates and the original is retired, changes here likely belong there too.
//
// The binding names below (dfb::out, tensor::dst) and the named argument set are this fork's
// interface: every later consumer inherits them, so they are taken from the kernel's own vocabulary
// rather than any one op's locals, and are not renamed once a consumer exists.

#include <stdint.h>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto total_num_rows = get_arg(args::total_num_rows);
    constexpr auto third_dim = get_arg(args::third_dim);
    constexpr auto tile_height = get_arg(args::tile_height);
    constexpr auto unpadded_X_size = get_arg(args::unpadded_X_size);

    const auto s = TensorAccessor(tensor::dst);
    Noc noc;
    // The block factories emit one writer instance per buffer set, so which buffer this instance
    // drains is decided by the host binding rather than by a fixed index.
    DataflowBuffer dfb_out0(dfb::out);

    auto write_block = [&](std::uint32_t num_rows,
                           std::uint32_t start_row_id,
                           std::uint32_t start_column_id,
                           std::uint32_t width_size,
                           std::uint32_t size_2d,
                           std::uint32_t single_block_size) {
        bool has_rows = (num_rows) > 0;

        dfb_out0.wait_front(single_block_size * has_rows);
        std::uint32_t l1_read_addr = dfb_out0.get_read_ptr();

        for (std::uint32_t k = start_row_id; k < start_row_id + num_rows; k++) {
            std::uint32_t total_size = start_column_id + width_size;
            std::uint32_t write_size = width_size;

            if (total_size > unpadded_X_size) {
                std::uint32_t padded_size = total_size - unpadded_X_size;
                write_size -= padded_size;
            }

            CoreLocalMem<std::uint32_t> src(l1_read_addr);
            noc.async_write(
                src, s, write_size, {.offset_bytes = 0}, {.page_id = size_2d + k, .offset_bytes = start_column_id});

            noc.async_write_barrier();

            l1_read_addr += width_size;
        }

        dfb_out0.pop_front(single_block_size * has_rows);
    };

    const auto width_size = get_arg(args::width_size);

    std::uint32_t size_2d = 0;
    for (std::uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        std::uint32_t start_row_id = get_arg(args::start_row_id);
        std::uint32_t start_column_id = get_arg(args::start_column_id);
        std::uint32_t single_block_size_row_arg = get_arg(args::single_block_size_row_arg);
        std::uint32_t single_block_size_col_arg = get_arg(args::single_block_size_col_arg);
        std::uint32_t sub_block_width_size = get_arg(args::sub_block_width_size);
        std::uint32_t single_sub_block_size_row_arg = get_arg(args::single_sub_block_size_row_arg);

        for (std::uint32_t b = 0; b < single_block_size_col_arg; b++) {
            std::uint32_t this_block_num_rows = tile_height;
            if (start_row_id + tile_height > total_num_rows) {
                this_block_num_rows = total_num_rows - start_row_id;
            }
            for (std::uint32_t m = 0; m < width_size; m += sub_block_width_size) {
                std::uint32_t start_column_id_u = start_column_id + m;
                if (this_block_num_rows > 0) {
                    write_block(
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
