// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "ckernel_defs.h"
#include "experimental/kernel_args.h"
#include "tt-metalium/constants.hpp"
#include "ttnn/operations/pool/device/kernels/experimental_device_api.hpp"

constexpr uint32_t MAX_RANK = 8;

FORCE_INLINE uint32_t round_down(uint32_t value, uint32_t multiple) {
    if (value % multiple != 0) {
        value -= value % multiple;
    }
    return value;
}

template <uint32_t is_non_aligned, uint32_t num_dims, uint32_t output_elem_size, uint32_t output_row_size_bytes>
TT_KERNEL void writer(
    uint32_t total_num_tiles,
    uint32_t num_tiles_per_read,
    uint32_t num_sticks_this_core,
    uint32_t padded_channels_elems,
    uint32_t misalignment) {
    uint32_t output_coord[MAX_RANK];
    uint32_t output_start_in_input[MAX_RANK];
    uint32_t output_end[MAX_RANK];
    for (uint32_t j = 0; j < num_dims; ++j) {
        output_coord[j] = get_vararg(j);
        output_start_in_input[j] = get_vararg(num_dims + j);
        output_end[j] = get_vararg(2 * num_dims + j);
    }

    constexpr uint32_t tile_size = get_tile_size(dfb::output);
    const uint32_t read_size = tile_size * num_tiles_per_read;

    DataflowBuffer untilized(dfb::untilized);
    DataflowBuffer output(dfb::output);
    Scratchpad<uint32_t> padding(scratch::padding);
    Noc noc;

    uint32_t write_offset = 0;
    uint32_t rows_remaining = num_sticks_this_core;
    uint32_t tiles_read = 0;
    const uint32_t block_row_size = read_size / tt::constants::TILE_HEIGHT;
    const uint32_t pad_addr = padding.get_base_address();
    constexpr uint32_t output_row_size_elems = output_row_size_bytes / output_elem_size;
    const uint32_t padded_channels_bytes = padded_channels_elems * output_elem_size;

    if (padded_channels_elems > 0) {
        if constexpr (output_elem_size == 4) {
            volatile tt_l1_ptr uint32_t* pad_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pad_addr);
            for (uint32_t i = 0; i < output_row_size_elems; ++i) {
                pad_ptr[i] = 0;
            }
        } else if constexpr (output_elem_size == 2) {
            volatile tt_l1_ptr uint16_t* pad_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pad_addr);
            for (uint32_t i = 0; i < output_row_size_elems; ++i) {
                pad_ptr[i] = 0;
            }
        }
    }

    const uint32_t pad_src_l1_addr = pad_addr + output_row_size_bytes - padded_channels_bytes;
    const uint32_t output_end_width_in_input = output_end[1] + output_start_in_input[1];
    UnicastEndpoint self_ep;
    while (tiles_read < total_num_tiles && rows_remaining > 0) {
        const uint32_t width_start_in_input = output_start_in_input[1] + output_coord[1];
        const uint32_t width_tile_start_in_input = round_down(width_start_in_input, ckernel::TILE_HEIGHT);
        const uint32_t width_tile_end_in_input = width_tile_start_in_input + ckernel::TILE_HEIGHT;

        const uint32_t read_start_offset = width_start_in_input - width_tile_start_in_input;
        uint32_t read_rows_size = ckernel::TILE_HEIGHT - read_start_offset;
        if (width_tile_end_in_input > output_end_width_in_input) {
            read_rows_size -= width_tile_end_in_input - output_end_width_in_input;
        }
        read_rows_size = std::min(read_rows_size, rows_remaining);
        rows_remaining -= read_rows_size;

        untilized.wait_front(num_tiles_per_read);
        const uint32_t noc_read_src_base = untilized.get_read_ptr() + read_start_offset * block_row_size;

        if constexpr (is_non_aligned) {
            uint32_t current_src_l1 = noc_read_src_base;
            uint32_t current_write_offset = write_offset;
            for (uint32_t row = 0; row < read_rows_size; row++) {
                noc.async_read(
                    self_ep,
                    output,
                    output_row_size_bytes,
                    experimental::local_addr(current_src_l1 + misalignment, noc.get_noc_id()),
                    {.offset_bytes = current_write_offset});
                current_src_l1 += block_row_size;
                current_write_offset += output_row_size_bytes;
            }
        } else {
            noc.async_read(
                self_ep,
                output,
                read_rows_size * block_row_size,
                experimental::local_addr(noc_read_src_base, noc.get_noc_id()),
                {.offset_bytes = write_offset});
        }

        uint32_t pad_write_offset = write_offset + output_row_size_bytes - padded_channels_bytes;
        if (padded_channels_elems > 0) {
            const auto pad_src = experimental::local_addr(pad_src_l1_addr, noc.get_noc_id());
            for (uint32_t row_index = 0; row_index < read_rows_size; row_index++) {
                noc.async_read(self_ep, output, padded_channels_bytes, pad_src, {.offset_bytes = pad_write_offset});
                pad_write_offset += output_row_size_bytes;
            }
        }
        noc.async_read_barrier();

        write_offset += read_rows_size * output_row_size_bytes;
        untilized.pop_front(num_tiles_per_read);
        tiles_read += num_tiles_per_read;

        // Advance the coordinate by the output rows emitted from this tile block.
        output_coord[1] += read_rows_size;
        for (uint32_t index = 1; index < num_dims - 1; index++) {
            if (output_coord[index] >= output_end[index]) {
                output_coord[index] = 0;
                // Carry a completed dimension into the next outer dimension.
                output_coord[index + 1] += 1;
            }
        }
    }
    noc.async_read_barrier();
}
