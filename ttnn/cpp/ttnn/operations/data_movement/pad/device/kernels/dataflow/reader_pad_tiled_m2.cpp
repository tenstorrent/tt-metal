// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-private copy of reader_pad_tiled.cpp.
// Binding changes only (data-movement logic is unchanged):
//   - the input circular buffer id comes from the DFB binding token (dfb::cb_input)
//   - the input tensor base address comes from the TensorAccessor binding (ta::src_args)
//   - the structural scalars (page_size, num_dims) come from named compile-time args (args::)
//   - the per-core work descriptors (num_pages_to_write, start_offset) are named runtime args (args::)
//   - the four per-dim shape/index arrays are positional runtime varargs (get_arg_addr), which live
//     right after the named runtime args.

#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t num_dims = get_arg(args::num_dims);

    const uint32_t num_pages_to_write = get_arg(args::num_pages_to_write);
    const uint32_t start_offset = get_arg(args::start_offset);

    // Varargs follow the two named runtime args (positions 0..1), so they begin at position 2.
    volatile tt_l1_ptr uint32_t* input_page_shape = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
    volatile tt_l1_ptr uint32_t* output_page_shape = input_page_shape + num_dims;
    volatile tt_l1_ptr uint32_t* input_id_per_dim = output_page_shape + num_dims;
    volatile tt_l1_ptr uint32_t* output_id_per_dim = input_id_per_dim + num_dims;

    constexpr uint32_t cb_id_input = dfb::cb_input;

    const auto s0 = TensorAccessor(ta::src_args);
    Noc noc;
    CircularBuffer cb_input(cb_id_input);

    bool within_input_region;
    uint32_t input_page_offset = start_offset;

    // This kernel keeps track of which page (tile) we are on from a logical tensor perspective
    // and reads from the input tensor only when we are within the input region
    // The writer will be waiting for the correct page to be available in the input circular buffer
    // For example, if we are padding (2, 2, 32, 32) -> (4, 4, 64, 64), then we condense the inner dims to tiles:
    // (2, 2, 1, 1) -> (4, 4, 2, 2) and as incrementing through writing the output, [0:2, 0:2, 0:1, 0:1] will be
    // tiles read from input, and the rest will be padding. So for this reader kernel, we will only read when
    // [0:2, 0:2, 0:1, 0:1] is reached, and skip reads otherwise.

    for (uint32_t out_pages_written = 0; out_pages_written < num_pages_to_write; out_pages_written++) {
        within_input_region = true;
        for (uint32_t d = 0; d < num_dims; d++) {
            if (input_id_per_dim[d] < output_id_per_dim[d]) {
                within_input_region = false;
                break;
            }
        }

        if (within_input_region) {
            cb_input.reserve_back(1);
            noc.async_read(s0, cb_input, page_size, {.page_id = input_page_offset}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_input.push_back(1);
            input_page_offset++;
            advance_tensor_index(input_id_per_dim, input_page_shape, num_dims);
        }
        advance_tensor_index(output_id_per_dim, output_page_shape, num_dims);
    }
}
