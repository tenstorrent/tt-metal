// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of pad's tiled multicore writer (private to PadTileMulticoreProgramFactory).
// Device-side NoC + TensorAccessor logic is unchanged; resource access moves to the Metal 2.0 named
// handles (dfb::/tensor::/args::):
//   - c_0 input stream -> dfb::cb_input (CONSUMER of the reader's producer)
//   - c_2 pad scratch   -> dfb::cb_pad_val (PRODUCER+CONSUMER self-loop)
//   - legacy c_1 (output CB) was dead (the writer streams straight to the output tensor) and is dropped.
// The four per-dim arrays are seeded from uniform per-rank varargs into local scratch (see reader).
#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// This kernel keeps track of which page (tile) we are on from a logical tensor perspective, and fills
// the output with either the input or padding respectively.
void kernel_main() {
    constexpr uint32_t num_dims = get_arg(args::num_dims);
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t pad_value = get_arg(args::pad_value);
    constexpr uint32_t element_size = get_arg(args::element_size);
    constexpr uint32_t num_elements = page_size / element_size;

    const uint32_t num_pages_to_write = get_arg(args::num_pages_to_write);
    const uint32_t start_offset = get_arg(args::start_offset);

    // Vararg layout (per core): [input_page_shape | output_page_shape | input_id_per_dim | output_id_per_dim].
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

    const auto s0 = TensorAccessor(tensor::dst);
    Noc noc;
    DataflowBuffer cb_input(dfb::cb_input);
    DataflowBuffer cb_pad_val(dfb::cb_pad_val);

    // Reserve and fill the pad-value scratchpad, generalized for any contiguous dtype.
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

    bool within_input_region;
    uint32_t output_page_offset = start_offset;

    for (uint32_t out_pages_written = 0; out_pages_written < num_pages_to_write; out_pages_written++) {
        within_input_region = true;
        for (uint32_t d = 0; d < num_dims; d++) {
            if (input_id_per_dim[d] < output_id_per_dim[d]) {
                within_input_region = false;
                break;
            }
        }

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
