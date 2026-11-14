// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_dims = get_arg(args::num_dims);
    const uint32_t tile_width = get_arg(args::tile_width);
    const uint32_t tile_height = get_arg(args::tile_height);

    // Three num_dims-long common vararg blocks, in the order the host appends them: the per-dimension
    // unpadded and padded tile counts, then the input's padded shape.
    constexpr uint32_t num_unpadded_tiles_base = 0;
    constexpr uint32_t num_padded_tiles_base = num_dims;
    constexpr uint32_t input_shape_base = 2 * num_dims;

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    const auto s0 = TensorAccessor(tensor::src);

    // Create objects for Device 2.0 API
    DataflowBuffer dfb_in0(dfb::in);
    Noc noc;

    // Get tile size from the DFB entry size
    const uint32_t tile_size = dfb_in0.get_entry_size();

    // Create TensorAccessors for start and end tensors
    const auto start_tensor_accessor = TensorAccessor(tensor::start);
    const auto end_tensor_accessor = TensorAccessor(tensor::end);

    // Read start and end indices from tensors using TensorAccessor
    uint32_t start_indices[num_dims];
    [[maybe_unused]] uint32_t end_indices[num_dims];

    // Staging region the start and end tensors are read into, one after the other.
    Scratchpad<volatile uint32_t> staging(scratch::tensor);

    // Read start tensor data into the staging region
    noc.async_read(start_tensor_accessor, staging, tile_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    {
        // Read the indices under a lock, which discards any cached copy, so the reads see what the NoC just wrote.
        auto staging_lock = staging.scoped_lock(0, num_dims);
        for (uint32_t i = 0; i < num_dims; i++) {
            start_indices[i] = staging[i];
        }
    }

    // Read end tensor data into the staging region
    noc.async_read(end_tensor_accessor, staging, tile_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    {
        auto staging_lock = staging.scoped_lock(0, num_dims);
        for (uint32_t i = 0; i < num_dims; i++) {
            end_indices[i] = staging[i];
        }
    }

    uint32_t start_offset = 0;

    if (num_dims >= 2) {
        uint32_t start_h_tiles = start_indices[num_dims - 2] / tile_height;
        uint32_t start_w_tiles = start_indices[num_dims - 1] / tile_width;

        uint32_t input_width = get_common_vararg(input_shape_base + num_dims - 1);
        uint32_t input_height = get_common_vararg(input_shape_base + num_dims - 2);
        uint32_t num_pages_width = input_width / tile_width;

        start_offset += start_h_tiles * num_pages_width + start_w_tiles;

        if (num_dims > 2) {
            uint32_t upper_dims_offset = 0;
            uint32_t multiplier = (input_height / tile_height) * num_pages_width;

            // Row-major Horner; must match get_upper_start_offset() in slice_device_operation.cpp
            for (uint32_t i = 0; i + 2 < num_dims; ++i) {
                upper_dims_offset = upper_dims_offset * get_common_vararg(input_shape_base + i) + start_indices[i];
            }
            start_offset += upper_dims_offset * multiplier;
        }
    }

    // The per-dimension index odometer this kernel advances as it walks the source. It is seeded from
    // the host's starting position for this core (the per-core vararg block) and then mutated in
    // place, so it needs writable storage of its own; vararg values are read-only.
    Scratchpad<uint32_t> id_per_dim(scratch::id_per_dim);
    // Keep every CPU access to the odometer inside a scoped_lock. On Quasar DM the scratchpad is reached
    // through the DM caches: the lock drops any old lines on entry and writes the counters back on exit, so
    // no dirty line outlives this program (a later program's flush or an eviction would otherwise write it
    // over whatever buffer then sits at this address). No-op on WH/BH.
    const auto id_per_dim_lock = id_per_dim.scoped_lock(0, num_dims);
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(j);
    }

    // Add the calculated offset to the base start_id
    uint32_t src_tile_id = start_id + start_offset;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        uint32_t old_src_tile_id = src_tile_id;

        dfb_in0.reserve_back(1);
        noc.async_read(s0, dfb_in0, tile_size, {.page_id = src_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0.push_back(1);

        src_tile_id++;
        for (uint32_t j = 0; j < num_dims; ++j) {
            id_per_dim[j]++;
            if (id_per_dim[j] == get_common_vararg(num_unpadded_tiles_base + j)) {
                id_per_dim[j] = 0;
                src_tile_id += get_common_vararg(num_padded_tiles_base + j);

            } else {
                break;
            }
        }
    }
}
