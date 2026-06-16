// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port. Used only by the PadTileMulticore factory, so ported in place.
// Logic, loop bounds and numeric paths are UNCHANGED; only the access mechanism moves to
// named bindings:
//   output address                       -> ta::dst (TensorAccessor)
//   input CB id                          -> dfb::in0
//   pad-val CB id                        -> dfb::pad
//   page_size/num_dims/pad_value/element_size CTAs -> get_arg(args::...)
//   num_pages_to_write/start_offset RTAs -> get_arg(args::...)
//   per-dim arrays (input_page_shape, output_page_shape, input_id_per_dim,
//                   output_id_per_dim)   -> runtime varargs (get_vararg)
//
// (The legacy output CB id is now carried by the dfb::out self-loop binding on the host and is
// not referenced here — the kernel never used it as a FIFO.)
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
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t num_dims = get_arg(args::num_dims);
    constexpr uint32_t pad_value = get_arg(args::pad_value);
    constexpr uint32_t element_size = get_arg(args::element_size);
    constexpr uint32_t num_elements = page_size / element_size;

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

    const auto s0 = TensorAccessor(ta::dst);
    Noc noc;
    DataflowBuffer cb_input(dfb::in0);
    DataflowBuffer cb_pad_val(dfb::pad);

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
