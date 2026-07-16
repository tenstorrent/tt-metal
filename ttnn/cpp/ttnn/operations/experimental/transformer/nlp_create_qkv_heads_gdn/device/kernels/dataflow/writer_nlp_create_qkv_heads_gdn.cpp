// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GDN create-heads writer: consumes one seq-tile-row ("block") of cb1 tiles in q→k→v order and
// scatters them head-major into Q [B, q_out_c, S, head_dim], K [B, k_out_c, S, head_dim],
// V [B, v_out_c, S, head_dim]. Each head is a contiguous HtWt block in its output tensor (per-head
// stride out_HtWt); within a head the out_w_tiles (head_dim) tiles are contiguous. No transpose.
// Q/K/V share out_h_tiles/out_w_tiles (shared head_dim); only the head COUNT differs. The whole
// block's qkv_tiles scattered writes are issued behind a SINGLE async_write_barrier so the NoC
// pipelines them (vs one barrier per tile); the CB handshake (wait_front/pop_front) is per block.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    // RUNTIME ARGS
    uint32_t q_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t k_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t v_tensor_addr = get_arg_val<uint32_t>(2);
    uint32_t num_blocks = get_arg_val<uint32_t>(3);
    uint32_t out_h_dim = get_arg_val<uint32_t>(4);
    uint32_t q_out_tensor_tile_id = get_arg_val<uint32_t>(5);
    uint32_t k_out_tensor_tile_id = get_arg_val<uint32_t>(6);
    uint32_t v_out_tensor_tile_id = get_arg_val<uint32_t>(7);

    // COMPILE TIME ARGS
    constexpr uint32_t out_h_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t out_w_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t out_HtWt = get_compile_time_arg_val(2);
    constexpr uint32_t q_out_c = get_compile_time_arg_val(3);
    constexpr uint32_t k_out_c = get_compile_time_arg_val(4);
    constexpr uint32_t v_out_c = get_compile_time_arg_val(5);
    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id = 1;
    constexpr uint32_t qkv_tiles = (q_out_c + k_out_c + v_out_c) * out_w_tiles;

    const auto sq = TensorAccessor(q_args, q_tensor_addr);
    const auto sk = TensorAccessor(k_args, k_tensor_addr);
    const auto sv = TensorAccessor(v_args, v_tensor_addr);

    CircularBuffer cb(cb_id);
    const uint32_t tile_bytes = get_tile_size(cb_id);

    // Scatter one output tensor's heads for this block: `heads` heads, each an HtWt block, base
    // page-id `out_tile_id`, per-head stride out_HtWt, per-w-tile stride 1. Reads each tile from
    // `read_base + idx * tile_bytes` (idx runs across the whole block's q→k→v tiles, matching the
    // reader's read order) and issues the async_write WITHOUT its own barrier/pop — the caller
    // barriers + pops the whole block once. Returns the last current tile-id (for batch rollover).
    auto scatter =
        [&](const auto& s, uint32_t heads, uint32_t out_tile_id, uint32_t read_base, uint32_t& idx) -> uint32_t {
        uint32_t along_c = out_tile_id;
        uint32_t cur = out_tile_id;
        for (uint32_t c_dim = 0; c_dim < heads; c_dim++) {
            cur = along_c;
            for (uint32_t w_dim = 0; w_dim < out_w_tiles; w_dim++) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(read_base + idx * tile_bytes), s, tile_bytes, {}, {.page_id = cur});
                idx++;
                cur++;
            }
            along_c += out_HtWt;
        }
        return cur;
    };

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Consume the whole block at once. The CB is sized to a multiple of qkv_tiles (program
        // factory), so the qkv_tiles tiles are contiguous in L1 from read_base; idx threads the
        // q→k→v read order the reader produced.
        cb.wait_front(qkv_tiles);
        uint32_t read_base = cb.get_read_ptr();
        uint32_t idx = 0;
        uint32_t q_cur = scatter(sq, q_out_c, q_out_tensor_tile_id, read_base, idx);
        uint32_t k_cur = scatter(sk, k_out_c, k_out_tensor_tile_id, read_base, idx);
        uint32_t v_cur = scatter(sv, v_out_c, v_out_tensor_tile_id, read_base, idx);
        noc.async_write_barrier();  // one barrier for the whole block: the writes pipeline on the NoC
        cb.pop_front(qkv_tiles);

        // Advance to the next seq tile-row, or roll over to the next batch after one full CHtWt.
        out_h_dim++;
        if (out_h_dim < out_h_tiles) {
            q_out_tensor_tile_id += out_w_tiles;
            k_out_tensor_tile_id += out_w_tiles;
            v_out_tensor_tile_id += out_w_tiles;
        } else {
            q_out_tensor_tile_id = q_cur;
            k_out_tensor_tile_id = k_cur;
            v_out_tensor_tile_id = v_cur;
            out_h_dim = 0;
        }
    }
}
