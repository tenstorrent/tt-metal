// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 (ProgramSpec) port of reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp.
// Used only by SliceTileTensorArgsSpecProgramFactory, so it is a local fork of the legacy reader
// (the legacy file remains in place for the legacy descriptor path). Logic, loop bounds and
// numeric paths are UNCHANGED; only the access mechanism moves to named bindings:
//   src / start / end addresses -> ta::src / ta::start / ta::end (TensorAccessor)
//   CB ids                      -> dfb::cb_in / dfb::cb_tensor
//   num_dims/tile_width/tile_height CTAs -> get_arg(args::...)
//   start_id/num_tiles named per-core RTAs -> get_arg(args::...)
//   per-dim id_per_dim                   -> runtime varargs (get_vararg)
//   num_unpadded/num_padded/input_shape common section -> common varargs (get_common_vararg)
//
// Vararg note: the legacy kernel mutated the id_per_dim slots in the RTA buffer in place.
// The m2 vararg API exposes value getters only (no writable pointer into the vararg region),
// so the per-dim running counters are copied into a local stack array and mutated there. This
// is purely an access-mechanism change; the counter arithmetic is identical.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = dfb::cb_in;
    constexpr uint32_t cb_id_tensor = dfb::cb_tensor;
    constexpr uint32_t num_dims = get_arg(args::num_dims);
    const uint32_t tile_width = get_arg(args::tile_width);
    const uint32_t tile_height = get_arg(args::tile_height);

    // Common varargs layout (after the src/start/end TensorBinding sections):
    //   [num_unpadded_tiles_per_dim[0..num_dims-1],
    //    num_padded_tiles_per_dim[0..num_dims-1],
    //    input_shape[0..num_dims-1]]
    // num_unpadded starts at common vararg index 0, num_padded at num_dims, input_shape at 2*num_dims.

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    const auto s0 = TensorAccessor(ta::src);

    // Create objects for Device 2.0 API
    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_tensor(cb_id_tensor);
    Noc noc;

    // Get tile size from CB interface
    const uint32_t tile_size = cb_in0.get_tile_size();

    // Create TensorAccessors for start and end tensors
    const auto start_tensor_accessor = TensorAccessor(ta::start);
    const auto end_tensor_accessor = TensorAccessor(ta::end);

    // Read start and end indices from tensors using TensorAccessor
    uint32_t start_indices[num_dims];
    uint32_t end_indices[num_dims];

    // Read start tensor data using separate circular buffer
    cb_tensor.reserve_back(1);
    uint32_t start_buffer_l1_addr = cb_tensor.get_write_ptr();
    noc.async_read(start_tensor_accessor, cb_tensor, tile_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* start_data = (volatile tt_l1_ptr uint32_t*)start_buffer_l1_addr;

    for (uint32_t i = 0; i < num_dims; i++) {
        start_indices[i] = start_data[i];
    }
    cb_tensor.pop_front(1);

    // Read end tensor data using separate circular buffer
    cb_tensor.reserve_back(1);
    uint32_t end_buffer_l1_addr = cb_tensor.get_write_ptr();
    noc.async_read(end_tensor_accessor, cb_tensor, tile_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* end_data = (volatile tt_l1_ptr uint32_t*)end_buffer_l1_addr;

    for (uint32_t i = 0; i < num_dims; i++) {
        end_indices[i] = end_data[i];
    }
    cb_tensor.pop_front(1);

    uint32_t start_offset = 0;

    if (num_dims >= 2) {
        uint32_t start_h_tiles = start_indices[num_dims - 2] / tile_height;
        uint32_t start_w_tiles = start_indices[num_dims - 1] / tile_width;

        // input_shape lives in the common varargs starting at index 2 * num_dims.
        uint32_t input_width = get_common_vararg(2 * num_dims + (num_dims - 1));
        uint32_t input_height = get_common_vararg(2 * num_dims + (num_dims - 2));
        uint32_t num_pages_width = input_width / tile_width;

        start_offset += start_h_tiles * num_pages_width + start_w_tiles;

        if (num_dims > 2) {
            uint32_t upper_dims_offset = 0;
            uint32_t multiplier = (input_height / tile_height) * num_pages_width;

            for (int32_t i = num_dims - 3; i >= 0; i--) {
                upper_dims_offset = upper_dims_offset * get_common_vararg(2 * num_dims + i) + start_indices[i];
            }
            start_offset += upper_dims_offset * multiplier;
        }
    }

    // Add the calculated offset to the base start_id
    uint32_t src_tile_id = start_id + start_offset;

    // Local copy of the per-dim running indices (runtime varargs 0..num_dims-1).
    uint32_t id_per_dim[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(j);
    }

    for (uint32_t i = 0; i < num_tiles; ++i) {
        uint32_t old_src_tile_id = src_tile_id;

        cb_in0.reserve_back(1);
        noc.async_read(s0, cb_in0, tile_size, {.page_id = src_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(1);

        src_tile_id++;
        for (uint32_t j = 0; j < num_dims; ++j) {
            id_per_dim[j]++;
            if (id_per_dim[j] == get_common_vararg(j)) {
                id_per_dim[j] = 0;
                src_tile_id += get_common_vararg(num_dims + j);

            } else {
                break;
            }
        }
    }
}
