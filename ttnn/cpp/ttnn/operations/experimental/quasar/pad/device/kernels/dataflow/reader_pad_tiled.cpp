// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of pad's tiled multicore reader (private to PadTileMulticoreProgramFactory).
// Device-side NoC + TensorAccessor logic is unchanged; resource access moves to the Metal 2.0 named
// handles (dfb::/tensor::/args::). The four per-dim arrays (input/output page shape, input/output
// id_per_dim) are read in loops with a runtime dim index gated on the CTA-bound rank, which is the
// canonical RTA-vararg case; they are seeded from uniform per-rank varargs into local scratch (the
// id_per_dim arrays are also mutated as the kernel iterates, so they must be local & mutable).
#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_dims = get_arg(args::num_dims);
    constexpr uint32_t page_size = get_arg(args::page_size);

    const uint32_t num_pages_to_write = get_arg(args::num_pages_to_write);
    const uint32_t start_offset = get_arg(args::start_offset);

    // Vararg layout (per core): [input_page_shape | output_page_shape | input_id_per_dim | output_id_per_dim],
    // each `num_dims` long.
    uint32_t input_page_shape[MAX_NUM_DIMS];
    uint32_t output_page_shape[MAX_NUM_DIMS];
    uint32_t input_id_per_dim[MAX_NUM_DIMS];
    uint32_t output_id_per_dim[MAX_NUM_DIMS];
    for (uint32_t d = 0; d < num_dims; ++d) {
        input_page_shape[d] = get_vararg(d);
        output_page_shape[d] = get_vararg(num_dims + d);
        input_id_per_dim[d] = get_vararg(2 * num_dims + d);
        output_id_per_dim[d] = get_vararg(3 * num_dims + d);
    }

    const auto s0 = TensorAccessor(tensor::src);
    Noc noc;
    DataflowBuffer cb_input(dfb::cb_input);

    bool within_input_region;
    uint32_t input_page_offset = start_offset;

    // This kernel keeps track of which page (tile) we are on from a logical tensor perspective
    // and reads from the input tensor only when we are within the input region. The writer waits
    // for the correct page in cb_input.
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
