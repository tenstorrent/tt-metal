// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 slice tile reader. Logic identical to
// reader_unary_unpad_dims_interleaved_start_id.cpp (that one stays for the
// legacy/descriptor consumers); only the bindings are Metal 2.0:
//   - CB index            -> dfb::cb_in
//   - num_dims            -> named CTA   (get_named_compile_time_arg_val)
//   - input accessor      -> ta::src     (address implicit; no src_addr CRTA)
//   - start_id/num_tiles  -> named RTAs  (get_arg(args::...))
//   - per-dim shape arrays -> common varargs (read-only)
//   - per-dim running index -> per-core varargs, copied into a local mutable array

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = dfb::cb_in;
    constexpr uint32_t num_dims = get_named_compile_time_arg_val("num_dims");

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    // Read-only per-dim shape (common varargs): [num_unpadded[num_dims], num_padded[num_dims]].
    // Mutable running index (per-core varargs): id_per_dim[num_dims] -> local copy.
    uint32_t num_unpadded_tiles[num_dims];
    uint32_t num_padded_tiles[num_dims];
    uint32_t id_per_dim[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        num_unpadded_tiles[j] = get_common_vararg(j);
        num_padded_tiles[j] = get_common_vararg(num_dims + j);
        id_per_dim[j] = get_vararg(j);
    }

    const auto s0 = TensorAccessor(ta::src);

    CircularBuffer cb_in0(cb_id_in0);
    Noc noc;

    const uint32_t tile_size = cb_in0.get_tile_size();

    uint32_t src_tile_id = start_id;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        // Copy Input
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
