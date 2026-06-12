// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 slice tile reader (slice start/end come from device TENSORS). Logic
// identical to reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp (that
// one stays for the legacy/descriptor consumers); only the bindings are Metal 2.0:
//   - src CB index        -> dfb::cb_in
//   - staging CB index    -> dfb::cb_tensor (used to read the start/end tensor pages)
//   - num_dims/tile_w/h   -> named CTAs (get_named_compile_time_arg_val)
//   - input accessor      -> ta::src    (address implicit; no src_addr CRTA)
//   - start accessor      -> ta::starts (address implicit; no start_addr CRTA)
//   - end accessor        -> ta::ends   (address implicit; no end_addr CRTA)
//   - start_id/num_tiles  -> named RTAs (get_arg(args::...))
//   - per-dim shape arrays -> common varargs (read-only):
//       [num_unpadded[num_dims], num_padded[num_dims], input_shape[num_dims]]
//   - per-dim running index -> per-core varargs, copied into a local mutable array
//
// NOTE: the start/end tensors are read with a standard TensorAccessor (one page,
// page_id = 0) into the staging CB, exactly as the legacy kernel does — so the
// Metal 2.0 TensorAccessor binding fits cleanly. The `end` tensor is read but the
// end indices are unused in the offset math (matching the legacy kernel, which only
// uses the start indices); the binding is kept to preserve identical behavior.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = dfb::cb_in;
    constexpr uint32_t cb_id_tensor = dfb::cb_tensor;
    constexpr uint32_t num_dims = get_named_compile_time_arg_val("num_dims");
    const uint32_t tile_width = get_named_compile_time_arg_val("tile_width");
    const uint32_t tile_height = get_named_compile_time_arg_val("tile_height");

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    // Read-only per-dim shape (common varargs):
    //   [num_unpadded[num_dims], num_padded[num_dims], input_shape[num_dims]].
    // Mutable running index (per-core varargs): id_per_dim[num_dims] -> local copy.
    uint32_t num_unpadded_tiles[num_dims];
    uint32_t num_padded_tiles[num_dims];
    uint32_t input_shape_args[num_dims];
    uint32_t id_per_dim[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        num_unpadded_tiles[j] = get_common_vararg(j);
        num_padded_tiles[j] = get_common_vararg(num_dims + j);
        input_shape_args[j] = get_common_vararg(2 * num_dims + j);
        id_per_dim[j] = get_vararg(j);
    }

    const auto s0 = TensorAccessor(ta::src);
    const auto start_tensor_accessor = TensorAccessor(ta::starts);
    const auto end_tensor_accessor = TensorAccessor(ta::ends);

    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_tensor(cb_id_tensor);
    Noc noc;

    const uint32_t tile_size = cb_in0.get_tile_size();

    // Read start and end indices from the start/end tensors using a staging CB.
    uint32_t start_indices[num_dims];
    uint32_t end_indices[num_dims];

    cb_tensor.reserve_back(1);
    uint32_t start_buffer_l1_addr = cb_tensor.get_write_ptr();
    noc.async_read(start_tensor_accessor, cb_tensor, tile_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* start_data = (volatile tt_l1_ptr uint32_t*)start_buffer_l1_addr;
    for (uint32_t i = 0; i < num_dims; i++) {
        start_indices[i] = start_data[i];
    }
    cb_tensor.pop_front(1);

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

        uint32_t input_width = input_shape_args[num_dims - 1];
        uint32_t input_height = input_shape_args[num_dims - 2];
        uint32_t num_pages_width = input_width / tile_width;

        start_offset += start_h_tiles * num_pages_width + start_w_tiles;

        if (num_dims > 2) {
            uint32_t upper_dims_offset = 0;
            uint32_t multiplier = (input_height / tile_height) * num_pages_width;

            for (int32_t i = num_dims - 3; i >= 0; i--) {
                upper_dims_offset = upper_dims_offset * input_shape_args[i] + start_indices[i];
            }
            start_offset += upper_dims_offset * multiplier;
        }
    }

    // Add the calculated offset to the base start_id
    uint32_t src_tile_id = start_id + start_offset;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_in0.reserve_back(1);
        noc.async_read(s0, cb_in0, tile_size, {.page_id = src_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(1);

        src_tile_id++;
        for (uint32_t j = 0; j < num_dims; ++j) {
            id_per_dim[j]++;
            if (id_per_dim[j] == num_unpadded_tiles[j]) {
                id_per_dim[j] = 0;
                src_tile_id += num_padded_tiles[j];
            } else {
                break;
            }
        }
    }
}
