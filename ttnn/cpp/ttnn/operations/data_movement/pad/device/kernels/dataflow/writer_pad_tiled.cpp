// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// This kernel keeps track of which page (tile) we are on from a logical tensor perspective, and fills the output with
// either the input or padding respectively
// For example, if we are padding (2, 2, 32, 32) -> (4, 4, 64, 64), then we condense the inner dims to tiles:
// (2, 2, 1, 1) -> (4, 4, 2, 2) and as incrementing through writing the output, [0:2, 0:2, 0:1, 0:1] will be
// tiles read from input, and the rest will be padding. So for this writer kernel, if we are within
// [0:2, 0:2, 0:1, 0:1] we wait for the reader to send us the correct tile, and then write it, otherwise we
// write padding.
void kernel_main() {
    constexpr auto page_size = get_arg(args::page_size);
    constexpr auto num_dims = get_arg(args::num_dims);
    constexpr auto pad_value = get_arg(args::pad_value);
    constexpr auto element_size = get_arg(args::element_size);
    constexpr uint32_t num_elements = page_size / element_size;

    const auto num_pages_to_write = get_arg(args::num_pages_to_write);
    const auto start_offset = get_arg(args::start_offset);

    // Four num_dims-long runtime vararg blocks, in host push order. The two id_per_dim blocks are
    // advanced as this kernel walks the output, so they are copied into locals: get_vararg() reads
    // a vararg but cannot write one back.
    uint32_t input_page_shape[num_dims];
    uint32_t output_page_shape[num_dims];
    uint32_t input_id_per_dim[num_dims];
    uint32_t output_id_per_dim[num_dims];
    for (uint32_t d = 0; d < num_dims; d++) {
        input_page_shape[d] = get_vararg(d);
        output_page_shape[d] = get_vararg(num_dims + d);
        input_id_per_dim[d] = get_vararg(2 * num_dims + d);
        output_id_per_dim[d] = get_vararg(3 * num_dims + d);
    }

    const auto s0 = TensorAccessor(tensor::dst);
    Noc noc;
    DataflowBuffer dfb_input(dfb::in0);
    DataflowBuffer dfb_pad_val(dfb::pad);

    // Reserve and push the pad value into the circular buffer, generalized for any contiguous dtype
    dfb_pad_val.reserve_back(1);
    uint32_t l1_write_addr = dfb_pad_val.get_write_ptr();
    volatile tt_l1_ptr uint8_t* pad_val_page = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write_addr);
    const volatile tt_l1_ptr uint8_t* pad_val = reinterpret_cast<const volatile tt_l1_ptr uint8_t*>(&pad_value);
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t b = 0; b < element_size; b++) {
            pad_val_page[i * element_size + b] = pad_val[b];
        }
    }
    dfb_pad_val.push_back(1);
    // Our scratchpad DFB is now a tile full of padding.

    bool within_input_region;
    uint32_t output_page_offset = start_offset;

    // Loop over all output pages to write
    for (uint32_t out_pages_written = 0; out_pages_written < num_pages_to_write; out_pages_written++) {
        within_input_region = true;
        for (uint32_t d = 0; d < num_dims; d++) {
            if (input_id_per_dim[d] < output_id_per_dim[d]) {
                within_input_region = false;
                break;
            }
        }

        // We have two cases, if we are within the input region, we wait for the reader to send us the correct tile
        // Otherwise we simply write the padding tile we have in our circular buffer
        if (within_input_region) {
            dfb_input.wait_front(1);
            noc.async_write(
                dfb_input, s0, page_size, {.offset_bytes = 0}, {.page_id = output_page_offset, .offset_bytes = 0});
            noc.async_write_barrier();
            advance_tensor_index(input_id_per_dim, input_page_shape, num_dims);
            dfb_input.pop_front(1);
        } else {
            CoreLocalMem<uint32_t> pad_src(l1_write_addr);
            noc.async_write(
                pad_src, s0, page_size, {.offset_bytes = 0}, {.page_id = output_page_offset, .offset_bytes = 0});
            noc.async_write_barrier();
        }
        advance_tensor_index(output_id_per_dim, output_page_shape, num_dims);
        output_page_offset++;
    }
}
