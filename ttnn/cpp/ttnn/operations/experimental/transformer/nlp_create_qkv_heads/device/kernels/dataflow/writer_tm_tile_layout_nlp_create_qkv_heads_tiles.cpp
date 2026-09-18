// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tile-split writer for the interleaved nlp_create_qkv_heads (no transpose_k_heads); the counterpart of
// reader_tm_tile_layout_nlp_create_qkv_heads_tiles.cpp: walks the same flattened (tile-row, row-tile) range and
// scatters each tile to its [B, heads, S, head_dim] position, CHUNK writes behind one barrier.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;
    uint32_t q_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t k_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t v_tensor_addr = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t start_tile = get_arg_val<uint32_t>(4);

    constexpr uint32_t q_out_h_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t q_out_w_tiles = get_compile_time_arg_val(1);  // tiles along head_dim
    constexpr uint32_t q_out_HtWt = get_compile_time_arg_val(2);
    constexpr uint32_t q_out_c = get_compile_time_arg_val(3);   // q heads
    constexpr uint32_t kv_out_c = get_compile_time_arg_val(4);  // kv heads
    constexpr uint32_t chunk = get_compile_time_arg_val(5);
    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr uint32_t q_num_tiles = q_out_c * q_out_w_tiles;
    constexpr uint32_t kv_num_tiles = kv_out_c * q_out_w_tiles;
    constexpr uint32_t row_tiles = q_num_tiles + 2 * kv_num_tiles;
    constexpr uint32_t q_out_CHtWt = q_out_c * q_out_HtWt;
    constexpr uint32_t kv_out_CHtWt = kv_out_c * q_out_HtWt;
    constexpr uint32_t cb_id = 1;

    const auto sq = TensorAccessor(q_args, q_tensor_addr);
    const auto sk = TensorAccessor(k_args, k_tensor_addr);
    const auto sv = TensorAccessor(v_args, v_tensor_addr);
    CircularBuffer cb(cb_id);
    const uint32_t tile_bytes = get_tile_size(cb_id);

    uint32_t b = start_tile / row_tiles;  // tile row
    uint32_t r = start_tile - b * row_tiles;
    uint32_t batch = b / q_out_h_tiles;
    uint32_t h = b - batch * q_out_h_tiles;  // tile row inside the image
    while (num_tiles > 0) {
        const uint32_t n = num_tiles < chunk ? num_tiles : chunk;
        cb.wait_front(n);
        uint32_t l1_read_addr = cb.get_read_ptr();
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
                b++;
                if (++h == q_out_h_tiles) {
                    h = 0;
                    batch++;
                }
            }
        }
        noc.async_write_barrier();
        cb.pop_front(n);
        num_tiles -= n;
    }
}
