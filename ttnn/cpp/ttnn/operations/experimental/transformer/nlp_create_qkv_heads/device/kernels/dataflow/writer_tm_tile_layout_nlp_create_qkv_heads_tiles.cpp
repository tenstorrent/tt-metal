// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tile-split writer for the interleaved nlp_create_qkv_heads (no transpose_k_heads); the counterpart of
// reader_tm_tile_layout_nlp_create_qkv_heads_tiles.cpp: walks the same flattened (tile-row, row-tile) range and
// scatters each tile to its [B, heads, S, head_dim] position, CHUNK writes behind one barrier.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;
    uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_tile = get_arg(args::start_tile);

    constexpr uint32_t q_out_h_tiles = get_arg(args::q_out_h_tiles);
    constexpr uint32_t q_out_w_tiles = get_arg(args::q_out_w_tiles);  // tiles along head_dim
    constexpr uint32_t q_out_HtWt = get_arg(args::q_out_HtWt);
    constexpr uint32_t q_out_c = get_arg(args::q_out_c);    // q heads
    constexpr uint32_t kv_out_c = get_arg(args::kv_out_c);  // kv heads
    constexpr uint32_t chunk = get_arg(args::chunk);
    constexpr uint32_t q_num_tiles = q_out_c * q_out_w_tiles;
    constexpr uint32_t kv_num_tiles = kv_out_c * q_out_w_tiles;
    constexpr uint32_t row_tiles = q_num_tiles + 2 * kv_num_tiles;
    constexpr uint32_t q_out_CHtWt = q_out_c * q_out_HtWt;
    constexpr uint32_t kv_out_CHtWt = kv_out_c * q_out_HtWt;

    const auto sq = TensorAccessor(tensor::q);
    const auto sk = TensorAccessor(tensor::k);
    const auto sv = TensorAccessor(tensor::v);
    DataflowBuffer dfb_qv(dfb::qv);
    const uint32_t tile_bytes = dfb_qv.get_tile_size();

    const uint32_t b = start_tile / row_tiles;  // tile row
    uint32_t r = start_tile - b * row_tiles;    // tile inside the row: [q | k | v]
    uint32_t batch = b / q_out_h_tiles;
    uint32_t h = b - batch * q_out_h_tiles;  // tile row inside the batch
    while (num_tiles > 0) {
        const uint32_t n = num_tiles < chunk ? num_tiles : chunk;
        dfb_qv.wait_front(n);
        uint32_t l1_read_addr = dfb_qv.get_read_ptr();
        for (uint32_t i = 0; i < n; i++) {
            uint32_t j = r;
            const uint32_t hw = h * q_out_w_tiles;
            if (j < q_num_tiles) {
                const uint32_t c = j / q_out_w_tiles;
                const uint32_t w = j - c * q_out_w_tiles;
                noc.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr),
                    sq,
                    tile_bytes,
                    {},
                    {.page_id = batch * q_out_CHtWt + c * q_out_HtWt + hw + w});
            } else {
                j -= q_num_tiles;
                const bool is_v = j >= kv_num_tiles;
                if (is_v) {
                    j -= kv_num_tiles;
                }
                const uint32_t c = j / q_out_w_tiles;
                const uint32_t w = j - c * q_out_w_tiles;
                const uint32_t page = batch * kv_out_CHtWt + c * q_out_HtWt + hw + w;
                if (is_v) {
                    noc.async_write(CoreLocalMem<uint32_t>(l1_read_addr), sv, tile_bytes, {}, {.page_id = page});
                } else {
                    noc.async_write(CoreLocalMem<uint32_t>(l1_read_addr), sk, tile_bytes, {}, {.page_id = page});
                }
            }
            l1_read_addr += tile_bytes;
            if (++r == row_tiles) {
                r = 0;
                if (++h == q_out_h_tiles) {
                    h = 0;
                    batch++;
                }
            }
        }
        noc.async_write_barrier();
        dfb_qv.pop_front(n);
        num_tiles -= n;
    }
}
