// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tile-split reader for the interleaved nlp_create_qkv_heads (no transpose_k_heads): every core owns a contiguous
// range of the flattened (tile-row, row-tile) index space instead of whole tile rows, so small-M shapes (SDXL:
// 32 tile rows x 120 tiles) spread over the full grid, and reads are issued CHUNK at a time behind one barrier.
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

    constexpr uint32_t q_num_tiles = get_arg(args::q_num_tiles);
    constexpr uint32_t kv_num_tiles = get_arg(args::kv_num_tiles);
    constexpr uint32_t in0_w_tiles = get_arg(args::in0_w_tiles);
    constexpr uint32_t chunk = get_arg(args::chunk);
    constexpr uint32_t row_tiles = q_num_tiles + 2 * kv_num_tiles;

    const auto s0 = TensorAccessor(tensor::input_q);
#ifdef READ_FROM_INPUT_TENSOR_KV
    constexpr uint32_t in1_w_tiles = get_arg(args::in1_w_tiles);
    const auto s1 = TensorAccessor(tensor::input_kv);
#endif
    DataflowBuffer dfb_qv(dfb::qv);
    const uint32_t tile_bytes = dfb_qv.get_tile_size();

    uint32_t b = start_tile / row_tiles;      // tile row
    uint32_t r = start_tile - b * row_tiles;  // tile inside the row: [q | k | v]
    while (num_tiles > 0) {
        const uint32_t n = num_tiles < chunk ? num_tiles : chunk;
        dfb_qv.reserve_back(n);
        uint32_t l1_write_addr = dfb_qv.get_write_ptr();
        for (uint32_t i = 0; i < n; i++) {
            if (r < q_num_tiles) {
                noc.async_read(
                    s0, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes, {.page_id = b * in0_w_tiles + r}, {});
            } else {
                uint32_t j = r - q_num_tiles;  // [0, 2 kv): k then v
#ifdef KV_TIED
                if (j >= kv_num_tiles) {
                    j -= kv_num_tiles;  // v is read from the k tiles
                }
#endif
#ifdef READ_FROM_INPUT_TENSOR_KV
                noc.async_read(
                    s1, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes, {.page_id = b * in1_w_tiles + j}, {});
#else
                noc.async_read(
                    s0,
                    CoreLocalMem<uint32_t>(l1_write_addr),
                    tile_bytes,
                    {.page_id = b * in0_w_tiles + q_num_tiles + j},
                    {});
#endif
            }
            l1_write_addr += tile_bytes;
            if (++r == row_tiles) {
                r = 0;
                b++;
            }
        }
        noc.async_read_barrier();
        dfb_qv.push_back(n);
        num_tiles -= n;
    }
}
