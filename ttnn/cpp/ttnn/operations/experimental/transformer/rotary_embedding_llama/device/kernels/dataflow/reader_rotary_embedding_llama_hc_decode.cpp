// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for rotary_embedding_llama in HC-transpose decode mode.
//
// Input tensor layout:  [1, num_heads, batch_size, head_dim] (interleaved, tilized)
// Cos/sin tensor layout: [1, num_heads_cs, batch_size, head_dim] (interleaved, tilized)
//   where num_heads_cs is either num_heads or 1 (broadcast across heads)
// Trans mat layout:     [1, 1, TILE_HEIGHT, TILE_WIDTH]  (interleaved, tilized)
//
// Each core is responsible for one (head_idx, batch_tile_idx) pair:
//   - reads head_dim_t tiles from input
//   - reads head_dim_t tiles from cos and sin (with optional head broadcast)
//   - reads 1 tile from trans_mat (reused across all iterations via compile-time constant tile)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t argrt = 0;
    uint32_t src_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t cos_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t sin_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t trans_mat_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t head_start = get_arg_val<uint32_t>(argrt++);
    uint32_t head_end = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_t_start = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_t_end = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t batch_t = get_compile_time_arg_val(4);         // total batch tiles
    constexpr uint32_t Wt = get_compile_time_arg_val(5);              // head_dim_t
    constexpr bool freq_per_head = get_compile_time_arg_val(6) == 1;  // cos/sin has per-head freqs
    constexpr auto input_args = TensorAccessorArgs<7>();
    constexpr auto cos_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_mat_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1;

    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const uint32_t trans_mat_tile_bytes = get_tile_size(trans_mat_cb_id);

    const auto s_input = TensorAccessor(input_args, src_addr, input_tile_bytes);
    const auto s_cos = TensorAccessor(cos_args, cos_addr, cos_tile_bytes);
    const auto s_sin = TensorAccessor(sin_args, sin_addr, sin_tile_bytes);
    const auto s_trans_mat = TensorAccessor(trans_mat_args, trans_mat_addr, trans_mat_tile_bytes);

    // Read transformation matrix once (tile index 0); it is shared for all cores.
    cb_reserve_back(trans_mat_cb_id, onetile);
    uint32_t trans_mat_l1_write_addr = get_write_ptr(trans_mat_cb_id);
    noc_async_read_tile(0, s_trans_mat, trans_mat_l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(trans_mat_cb_id, onetile);

    // Input tile linear index: input[0, h, bt, w] = h * batch_t * Wt + bt * Wt + w
    // Cos/sin tile linear index (freq_per_head=true): cs[0, h, bt, w] = h * batch_t * Wt + bt * Wt + w
    // Cos/sin tile linear index (freq_per_head=false): cs[0, 0, bt, w] = bt * Wt + w

    for (uint32_t h = head_start; h < head_end; ++h) {
        for (uint32_t bt = batch_t_start; bt < batch_t_end; ++bt) {
            const uint32_t input_base = h * batch_t * Wt + bt * Wt;
            const uint32_t cs_base = freq_per_head ? (h * batch_t * Wt + bt * Wt) : (bt * Wt);

            cb_reserve_back(input_cb_id, Wt);
            cb_reserve_back(cos_cb_id, Wt);
            cb_reserve_back(sin_cb_id, Wt);

            uint32_t input_l1_write_addr = get_write_ptr(input_cb_id);
            uint32_t cos_l1_write_addr = get_write_ptr(cos_cb_id);
            uint32_t sin_l1_write_addr = get_write_ptr(sin_cb_id);

            for (uint32_t w = 0; w < Wt; ++w) {
                noc_async_read_tile(input_base + w, s_input, input_l1_write_addr);
                input_l1_write_addr += input_tile_bytes;

                noc_async_read_tile(cs_base + w, s_cos, cos_l1_write_addr);
                cos_l1_write_addr += cos_tile_bytes;

                noc_async_read_tile(cs_base + w, s_sin, sin_l1_write_addr);
                sin_l1_write_addr += sin_tile_bytes;
            }

            noc_async_read_barrier();

            cb_push_back(input_cb_id, Wt);
            cb_push_back(cos_cb_id, Wt);
            cb_push_back(sin_cb_id, Wt);
        }
    }
}
