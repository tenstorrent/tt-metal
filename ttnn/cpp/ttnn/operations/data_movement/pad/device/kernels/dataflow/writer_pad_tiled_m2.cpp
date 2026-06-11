// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-private copy of writer_pad_tiled.cpp.
// Binding changes only (data-movement logic is unchanged):
//   - the input circular buffer id (shared with the reader) comes from the DFB binding token
//     (dfb::cb_input); the pad-value scratchpad CB from dfb::cb_pad_val
//   - the output tensor base address comes from the TensorAccessor binding (ta::dst_args)
//   - the structural scalars (page_size, num_dims, pad_value, element_size) come from named
//     compile-time args (args::)
//   - the per-core work descriptors (num_pages_to_write, start_offset) are named runtime args (args::)
//   - the four per-dim shape/index arrays are positional runtime varargs (get_arg_addr), which live
//     right after the named runtime args.

#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// This kernel keeps track of which page (tile) we are on from a logical tensor perspective, and fills the output with
// either the input or padding respectively
// For example, if we are padding (2, 2, 32, 32) -> (4, 4, 64, 64), then we condense the inner dims to tiles:
// (2, 2, 1, 1) -> (4, 4, 2, 2) and as incrementing through writing the output, [0:2, 0:2, 0:1, 0:1] will be
// tiles read from input, and the rest will be padding. So for this writer kernel, if we are within
// [0:2, 0:2, 0:1, 0:1] we wait for the reader to send us the correct tile, and then write it, otherwise we
// write padding.
void kernel_main() {
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t num_dims = get_arg(args::num_dims);
    constexpr uint32_t pad_value = get_arg(args::pad_value);
    constexpr uint32_t element_size = get_arg(args::element_size);
    constexpr uint32_t num_elements = page_size / element_size;

    const uint32_t num_pages_to_write = get_arg(args::num_pages_to_write);
    const uint32_t start_offset = get_arg(args::start_offset);

    // Varargs follow the two named runtime args (positions 0..1), so they begin at position 2.
    volatile tt_l1_ptr uint32_t* input_page_shape = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
    volatile tt_l1_ptr uint32_t* output_page_shape = input_page_shape + num_dims;
    volatile tt_l1_ptr uint32_t* input_id_per_dim = output_page_shape + num_dims;
    volatile tt_l1_ptr uint32_t* output_id_per_dim = input_id_per_dim + num_dims;

    constexpr uint32_t cb_id_input = dfb::cb_input;
    constexpr uint32_t cb_id_pad_val = dfb::cb_pad_val;

    const auto s0 = TensorAccessor(ta::dst_args);
    Noc noc;
    CircularBuffer cb_input(cb_id_input);
    CircularBuffer cb_pad_val(cb_id_pad_val);

    // Reserve and push the pad value into the circular buffer, generalized for any contiguous dtype
    cb_pad_val.reserve_back(1);
    uint32_t l1_write_addr = cb_pad_val.get_write_ptr();
    volatile tt_l1_ptr uint8_t* pad_val_page = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write_addr);
    const volatile tt_l1_ptr uint8_t* pad_val = reinterpret_cast<const volatile tt_l1_ptr uint8_t*>(&pad_value);
    for (uint32_t i = 0; i < num_elements; i++) {
        for (uint32_t b = 0; b < element_size; b++) {
            pad_val_page[i * element_size + b] = pad_val[b];
        }
    }
    cb_pad_val.push_back(1);
    // Our scratchpad cb is now a tile full of padding.

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
            cb_input.wait_front(1);
            noc.async_write(
                cb_input, s0, page_size, {.offset_bytes = 0}, {.page_id = output_page_offset, .offset_bytes = 0});
            noc.async_write_barrier();
            advance_tensor_index(input_id_per_dim, input_page_shape, num_dims);
            cb_input.pop_front(1);
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
