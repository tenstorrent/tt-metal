// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax writer kernel (BRISC/RISCV_0).
//
// Per slab (one (N,C) pair):
//   - Reads Ht×Wt output tiles from cb_output_tiles and writes to DRAM/L1
//   - Tiles are in row-major tile order within each slab

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
// CB indices — must match program descriptor
constexpr uint32_t cb_output_tiles = 16;
}  // namespace

void kernel_main() {
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t num_slabs = get_arg_val<uint32_t>(2);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // CT args: 2 scalar, then TensorAccessorArgs
    constexpr auto dst_args = TensorAccessorArgs<2>();
    const auto dst_accessor = TensorAccessor(dst_args, output_buffer_address);

    CircularBuffer output_cb(cb_output_tiles);
    Noc noc;
    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);

    uint32_t tiles_per_slab = Ht * Wt;
    uint32_t tile_id = start_tile_id;

    for (uint32_t slab = 0; slab < num_slabs; ++slab) {
        for (uint32_t i = 0; i < tiles_per_slab; ++i) {
            output_cb.wait_front(1);
            noc.async_write(output_cb, dst_accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
            noc.async_write_barrier();
            output_cb.pop_front(1);
            tile_id++;
        }
    }
}
