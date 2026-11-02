// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "internal/dataflow/dataflow_api_addrgen.h"
#include "api/debug/dprint.h"
#include "ckernel_defs.h"
#include "api/tile_constants.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

uint32_t round_down(uint32_t value, uint32_t multiple) {
    if (value % multiple != 0) {
        value -= (value % multiple);
    }
    return value;
}
void kernel_main() {
    constexpr uint32_t is_non_aligned = get_arg(args::is_non_aligned);
    constexpr uint32_t num_dims = get_arg(args::num_dims);
    constexpr uint32_t output_elem_size = get_arg(args::output_elem_size);
    constexpr uint32_t output_row_size_bytes = get_arg(args::output_row_size_bytes);

    const uint32_t total_num_tiles = get_arg(args::total_num_tiles);
    const uint32_t num_tiles_per_read = get_arg(args::num_tiles_per_read);
    const uint32_t num_sticks_this_core = get_arg(args::num_sticks_this_core);
    const uint32_t padded_channels_elems = get_arg(args::padded_channels_elems);
    const uint32_t misalignment = get_arg(args::misalignment);
    // Vararg blocks, num_dims each: the output_coord start, output_start_in_input, output_end.
    constexpr uint32_t output_coord_seed_base = 0;
    constexpr uint32_t output_start_in_input_base = num_dims;
    constexpr uint32_t output_end_base = 2 * num_dims;

    constexpr uint32_t tile_size = get_tile_size(dfb::out);
    const uint32_t read_size = tile_size * num_tiles_per_read;

    // Per-dimension output position. The host passes its starting values as varargs, which are read-only,
    // so the kernel copies them into this scratchpad and advances them there.
    Scratchpad<uint32_t> output_coord(scratch::output_coord);
    for (uint32_t j = 0; j < num_dims; ++j) {
        output_coord[j] = get_vararg(output_coord_seed_base + j);
    }

    Noc noc;
    DataflowBuffer dfb_untilized(dfb::untilized);
    DataflowBuffer dfb_out(dfb::out);
    DataflowBuffer dfb_padding(dfb::padding);

    uint32_t write_offset = 0;
    uint32_t rows_remaining = num_sticks_this_core;
    uint32_t tiles_read = 0;
    uint32_t block_row_size = read_size / tt::constants::TILE_HEIGHT;
    const uint32_t pad_addr = dfb_padding.get_read_ptr();
    const uint32_t output_row_size_elems = output_row_size_bytes / output_elem_size;

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
        } else if constexpr (output_elem_size == 1) {
            volatile tt_l1_ptr uint8_t* pad_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(pad_addr);
            for (uint32_t i = 0; i < output_row_size_elems; ++i) {
                pad_ptr[i] = 0;
            }
        }
    }

    const uint32_t pad_src_l1_addr = pad_addr + output_row_size_bytes - padded_channels_bytes;

#ifdef DEBUG
    DPRINT(
        "total_num_tiles: {}, num_tiles_per_read: {}, tile_size: {}, read_size: {}, block_row_size: {}\n",
        total_num_tiles,
        num_tiles_per_read,
        tile_size,
        read_size,
        block_row_size);
    DPRINT(
        "pad_addr : {} pad_src_l1_addr : {} padded_channels_elems: {}\n",
        pad_addr,
        pad_src_l1_addr,
        padded_channels_elems);
    DPRINT("Unaligned {}\n", is_non_aligned);
    DPRINT(
        "Output Row Size Elems {} Bytes: {} Output Elem Size: {}\n",
        output_row_size_elems,
        output_row_size_bytes,
        output_elem_size);
#endif

    const uint32_t output_end_width_in_input =
        get_vararg(output_end_base + 1) + get_vararg(output_start_in_input_base + 1);
    uint32_t row_count = 0;
    UnicastEndpoint self_ep;
    while (tiles_read < total_num_tiles && rows_remaining > 0) {
        uint32_t width_start_in_input = get_vararg(output_start_in_input_base + 1) + output_coord[1];
        uint32_t width_tile_start_in_input = round_down(width_start_in_input, ckernel::TILE_HEIGHT);
        uint32_t width_tile_end_in_input = width_tile_start_in_input + ckernel::TILE_HEIGHT;

        uint32_t read_start_offset = width_start_in_input - width_tile_start_in_input;
        uint32_t read_rows_size = ckernel::TILE_HEIGHT - read_start_offset;
        if (width_tile_end_in_input > output_end_width_in_input) {
            read_rows_size -= (width_tile_end_in_input - output_end_width_in_input);
        }
        read_rows_size = std::min(read_rows_size, rows_remaining);
        rows_remaining -= read_rows_size;

#ifdef DEBUG
        DPRINT(
            "Width Start in Input: {} Width Tile Start in Input: {} Read Start Offset: {} Read Rows Size: {} Remaining "
            "{}\n",
            width_start_in_input,
            width_tile_start_in_input,
            read_start_offset,
            read_rows_size,
            rows_remaining);

        DPRINT(
            "Tiles Read {} Output Coord: {} {} {} {}\n",
            tiles_read,
            output_coord[0],
            output_coord[1],
            output_coord[2],
            output_coord[3]);
        DPRINT("Write Offset {}\n", write_offset);

#endif

        dfb_untilized.wait_front(num_tiles_per_read);
        const uint32_t noc_read_src_base = dfb_untilized.get_read_ptr() + read_start_offset * block_row_size;

        if constexpr (is_non_aligned) {
            uint32_t current_src_l1 = noc_read_src_base;
            uint32_t current_write_offset = write_offset;
            for (uint32_t row = 0; row < read_rows_size; row++) {
                noc.async_read(
                    self_ep,
                    dfb_out,
                    output_row_size_bytes,
                    experimental::local_addr(current_src_l1 + misalignment, noc.get_noc_id()),
                    {.offset_bytes = current_write_offset});
                current_src_l1 += block_row_size;
                current_write_offset += output_row_size_bytes;
            }
        } else {
            noc.async_read(
                self_ep,
                dfb_out,
                read_rows_size * block_row_size,
                experimental::local_addr(noc_read_src_base, noc.get_noc_id()),
                {.offset_bytes = write_offset});
        }
        uint32_t pad_write_offset = write_offset + output_row_size_bytes - padded_channels_bytes;
        if (padded_channels_elems > 0) {
#ifdef DEBUG
            volatile tt_l1_ptr uint16_t* pad_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                pad_addr + output_row_size_bytes - padded_channels_bytes);
            DPRINT("Pad Data = ");
            for (uint32_t i = 0; i < padded_channels_elems; ++i) {
                DPRINT("{} ", pad_ptr[i]);
            }
            DPRINT("\n");
            DPRINT("Pad Write Offset : {}\n", pad_write_offset);
#endif
            const auto pad_src = experimental::local_addr(pad_src_l1_addr, noc.get_noc_id());
            for (uint32_t row_index = 0; row_index < read_rows_size; row_index++) {
                noc.async_read(
                    self_ep,
                    dfb_out,
                    padded_channels_elems * output_elem_size,
                    pad_src,
                    {.offset_bytes = pad_write_offset});
                pad_write_offset += output_row_size_bytes;
            }
        }
        noc.async_read_barrier();

        write_offset += read_rows_size * output_row_size_bytes;
        dfb_untilized.pop_front(num_tiles_per_read);
        tiles_read += num_tiles_per_read;

        // output_coord keeps track of the current position in the output tensor
        // Increment the output coordinate for the next read by the size of the current read.
        output_coord[1] += read_rows_size;

        // If output_coord goes beyond output_end, reset it to 0 and increment the next dimension.
        for (uint32_t index = 1; index < num_dims - 1; index++) {
            if (output_coord[index] >= get_vararg(output_end_base + index)) {
                output_coord[index] = 0;
                output_coord[index + 1] += 1;
            }
        }
    }
    noc.async_read_barrier();
}
