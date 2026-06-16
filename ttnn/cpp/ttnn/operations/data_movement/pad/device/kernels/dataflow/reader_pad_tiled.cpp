// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port. Used only by the PadTileMulticore factory, so ported in place.
// Logic, loop bounds and numeric paths are UNCHANGED; only the access mechanism moves to
// named bindings:
//   input address              -> ta::src (TensorAccessor)
//   CB id                      -> dfb::in0
//   page_size/num_dims CTAs    -> get_arg(args::...)
//   num_pages_to_write/start_offset RTAs -> get_arg(args::...)
//   per-dim arrays (input_page_shape, output_page_shape, input_id_per_dim,
//                   output_id_per_dim) -> runtime varargs (get_vararg)
//
// Vararg note: the legacy kernel read the per-dim arrays via in-place tt_l1_ptr pointers and
// mutated input_id_per_dim / output_id_per_dim through advance_tensor_index. The m2 vararg API
// exposes value getters only (no writable pointer into the vararg region), so the per-dim
// arrays are copied into local stack arrays and mutated there. This is purely an access-mechanism
// change; the index arithmetic is identical.

#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t num_dims = get_arg(args::num_dims);

    const uint32_t num_pages_to_write = get_arg(args::num_pages_to_write);
    const uint32_t start_offset = get_arg(args::start_offset);

    // Per-dim arrays as runtime varargs (positional):
    //   [input_page_shape[0..num_dims-1], output_page_shape[0..num_dims-1],
    //    input_id_per_dim[0..num_dims-1], output_id_per_dim[0..num_dims-1]]
    // input_id_per_dim / output_id_per_dim are mutated below, so copy all into local stack arrays.
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

    const auto s0 = TensorAccessor(ta::src);
    Noc noc;
    DataflowBuffer cb_input(dfb::in0);

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
