// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto page_size = get_arg(args::page_size);
    constexpr auto num_dims = get_arg(args::num_dims);

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

    const auto s0 = TensorAccessor(tensor::src);
    Noc noc;
    DataflowBuffer dfb_input(dfb::in0);

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
            dfb_input.reserve_back(1);
            noc.async_read(s0, dfb_input, page_size, {.page_id = input_page_offset}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_input.push_back(1);
            input_page_offset++;
            advance_tensor_index(input_id_per_dim, input_page_shape, num_dims);
        }
        advance_tensor_index(output_id_per_dim, output_page_shape, num_dims);
    }
}
