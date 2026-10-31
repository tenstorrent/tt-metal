// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    constexpr uint32_t num_dims = get_arg(args::num_dims);

    // Two num_dims-long common vararg blocks, in the order the host appends them: the per-dimension
    // unpadded and padded tile counts.
    constexpr uint32_t num_unpadded_tiles_base = 0;
    constexpr uint32_t num_padded_tiles_base = num_dims;

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    // In and out are assumed to be same dataformat
    const auto s0 = TensorAccessor(tensor::src);

    // Create objects for Device 2.0 API
    DataflowBuffer dfb_in0(dfb::in);
    Noc noc;

    // Get tile size from the DFB entry size
    const uint32_t tile_size = dfb_in0.get_entry_size();

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

    uint32_t src_tile_id = start_id;

    // Begin row inside a tile: output row r is input row r + row_shift of the tile the walk is on, or of the tile below
    // it once that passes row 31. Both tiles are staged in tile_pair. Each output face (16 rows of one half) then comes
    // from at most two runs of consecutive face rows, which are contiguous in a face, so a tile is at most eight NoC
    // copies; rows past the input are zeroed.
    constexpr uint32_t row_shift = get_arg(args::row_shift);
    constexpr uint32_t tile_rows_left = get_arg(args::tile_rows_left);
    Scratchpad<uint8_t> tile_pair(scratch::tile_pair);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_in0.reserve_back(1);
        if constexpr (row_shift == 0) {
            noc.async_read(s0, dfb_in0, tile_size, {.page_id = src_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
        } else {
            const uint32_t face_bytes = tile_size / 4;
            const uint32_t face_row_bytes = face_bytes / 16;
            const uint32_t total_tiles_x =
                get_common_vararg(num_unpadded_tiles_base) + get_common_vararg(num_padded_tiles_base);
            const uint32_t tile_row = id_per_dim[1];
            const bool tile_valid[2] = {tile_row < tile_rows_left, tile_row + 1 < tile_rows_left};
            for (uint32_t t = 0; t < 2; ++t) {
                if (tile_valid[t]) {
                    noc.async_read(
                        s0,
                        tile_pair,
                        tile_size,
                        {.page_id = src_tile_id + t * total_tiles_x},
                        {.offset_bytes = t * tile_size});
                }
            }
            noc.async_read_barrier();
            const uint32_t dst_addr = dfb_in0.get_write_ptr();
            bool zeroed = false;
            for (uint32_t out_face = 0; out_face < 4; ++out_face) {
                const uint32_t half = out_face & 1;
                uint32_t out_row = (out_face >> 1) * 16;
                uint32_t dst_offset = out_face * face_bytes;
                for (uint32_t rows_left = 16; rows_left > 0;) {
                    const uint32_t src_row = out_row + row_shift;  // 0..63 over the staged pair
                    const uint32_t src_tile = src_row >> 5;
                    const uint32_t in_row = src_row & 31;
                    const uint32_t face_rows_left = 16 - (in_row & 15);
                    const uint32_t run_rows = rows_left < face_rows_left ? rows_left : face_rows_left;
                    const uint32_t run_bytes = run_rows * face_row_bytes;
                    if (tile_valid[src_tile]) {
                        const uint32_t src_offset = src_tile * tile_size + ((in_row >> 4) * 2 + half) * face_bytes +
                                                    (in_row & 15) * face_row_bytes;
                        tt::data_movement::common::tt_memmove<true, true, true, 0>(
                            noc,
                            CoreLocalMem<uint8_t>(dst_addr + dst_offset),
                            tile_pair.local_mem() + src_offset,
                            run_bytes);
                    } else {
                        noc.async_write_zeros(dfb_in0, run_bytes, {.offset_bytes = dst_offset});
                        zeroed = true;
                    }
                    out_row += run_rows;
                    dst_offset += run_bytes;
                    rows_left -= run_rows;
                }
            }
            noc.async_read_barrier();
            if (zeroed) {
                noc.write_zeros_l1_barrier();
            }
        }
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
