// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Batched-width-sharded matmul output writer.
//
// Each core owns a distinct (batch-block, N-block) and produces a [Bc, M, Nc] output block in
// out_cb (compute -> writer). The output tensor is DRAM INTERLEAVED with logical shape
// [d0, d1, M, N] (= the torch reference), i.e. a tile grid of [batch*M_tiles, N_tiles]. This
// kernel scatters each local tile to its global tile (page) index:
//
//   out_row = b*M_tiles + mt   (b = b_idx*Bc + bc_i)
//   out_col = n_idx*Nc_tiles + nc
//   page_id = out_row * N_tiles + out_col
//
// out_cb is packed by compute in (bc_i, mt, nc) order, so the local tile offset advances linearly.
void kernel_main() {
    constexpr uint32_t out_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t Bc = get_compile_time_arg_val(1);
    constexpr uint32_t M_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t Nc_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_idx = get_arg_val<uint32_t>(1);  // batch-block index owned by this core
    const uint32_t n_idx = get_arg_val<uint32_t>(2);  // N-block index owned by this core

    constexpr uint32_t out_num_tiles = Bc * M_tiles * Nc_tiles;

    Noc noc;
    CircularBuffer out_cb(out_cb_index);
    const auto out = TensorAccessor(out_args, out_addr, out_cb.get_tile_size());

    const uint32_t tile_size = out_cb.get_tile_size();
    out_cb.wait_front(out_num_tiles);

    uint32_t local = 0;
    for (uint32_t bc_i = 0; bc_i < Bc; ++bc_i) {
        const uint32_t b = b_idx * Bc + bc_i;
        for (uint32_t mt = 0; mt < M_tiles; ++mt) {
            const uint32_t out_row = b * M_tiles + mt;
            for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
                const uint32_t out_col = n_idx * Nc_tiles + nc;
                const uint32_t page_id = out_row * N_tiles + out_col;
                noc.async_write(out_cb, out, tile_size, {.offset_bytes = local * tile_size}, {.page_id = page_id});
                ++local;
            }
        }
    }
    noc.async_write_barrier();
    out_cb.pop_front(out_num_tiles);
}
