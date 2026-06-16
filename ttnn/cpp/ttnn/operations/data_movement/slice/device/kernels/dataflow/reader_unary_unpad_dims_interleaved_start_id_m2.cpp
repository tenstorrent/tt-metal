// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 (ProgramSpec) port of reader_unary_unpad_dims_interleaved_start_id.cpp.
// Used only by SliceTileSpecProgramFactory, so it is a local fork of the legacy reader
// (the legacy file is still consumed unchanged by SliceTileProgramFactory::create_descriptor,
// which ccl/mesh_partition reuses). Logic, loop bounds and numeric paths are UNCHANGED;
// only the access mechanism moves to named bindings:
//   src address    -> ta::src (TensorAccessor)
//   CB id          -> dfb::cb_in
//   start_id/num_tiles named RTAs -> get_arg(args::...)
//   per-dim id_per_dim          -> runtime varargs (get_vararg)
//   src_addr/num_unpadded/num_padded common section -> common varargs (get_common_vararg)
//
// Vararg note: the legacy kernel mutated the id_per_dim slots in the RTA buffer in place.
// The m2 vararg API exposes value getters only (no writable pointer into the vararg
// region), so the per-dim running counters are copied into a local stack array and mutated
// there. This is purely an access-mechanism change; the counter arithmetic is identical.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = dfb::cb_in;
    constexpr uint32_t num_dims = get_arg(args::num_dims);

    // Common varargs layout (after the single TensorBinding section for ta::src):
    //   [num_unpadded_tiles_per_dim[0..num_dims-1], num_padded_tiles_per_dim[0..num_dims-1]]
    // num_unpadded starts at common vararg index 0, num_padded at index num_dims.

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    // Local copy of the per-dim running indices (runtime varargs 0..num_dims-1).
    uint32_t id_per_dim[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(j);
    }

    // In and out are assumed to be same dataformat
    const auto s0 = TensorAccessor(ta::src);

    // Create objects for Device 2.0 API
    CircularBuffer cb_in0(cb_id_in0);
    Noc noc;

    // Get tile size from CB interface
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
            if (id_per_dim[j] == get_common_vararg(j)) {
                id_per_dim[j] = 0;
                src_tile_id += get_common_vararg(num_dims + j);
            } else {
                break;
            }
        }
    }
}
