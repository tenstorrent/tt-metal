// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for rotary_embedding_llama in HC-transpose decode mode.
//
// Input tensor layout:  [1, num_heads, batch_size, head_dim] (interleaved, tilized)
// Cos/sin tensor layout: [1, num_heads_cs, batch_size, head_dim]
//   - INTERLEAVED (num_heads_cs == 1 or num_heads), or
//   - HEIGHT_SHARDED with shard_shape=[TILE_HEIGHT, head_dim] (one row per core)
// Trans mat layout: [1, 1, TILE_HEIGHT, TILE_WIDTH]
//   - INTERLEAVED, or
//   - HEIGHT_SHARDED with shard_shape=[TILE_HEIGHT, TILE_WIDTH] (one tile per core)
//
// When a tensor is HEIGHT_SHARDED its data is already resident in each core's L1.
// The CB is globally-allocated at that address, so the reader only needs to
// signal compute via cb_reserve_back / cb_push_back — no NOC reads required.
//
// When reload_cos_sin=false and !freq_per_head and !cos_sin_sharded:
//   cos/sin tiles for all assigned batch tiles are loaded once before the head
//   loop and held in the CB for the duration; compute reads them with a
//   per-batch-tile offset, eliminating repeated NOC reads across heads.
//
// When reload_cos_sin=true or freq_per_head=true (and !cos_sin_sharded):
//   cos/sin are re-read for every (head, batch_tile) pair (fallback).

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
    constexpr uint32_t batch_t = get_compile_time_arg_val(4);             // total batch tiles in tensor
    constexpr uint32_t Wt = get_compile_time_arg_val(5);                  // head_dim_t
    constexpr bool freq_per_head = get_compile_time_arg_val(6) == 1;      // cos/sin has per-head freqs
    constexpr bool reload_cos_sin = get_compile_time_arg_val(7) == 1;     // re-read cos/sin every iteration
    constexpr bool trans_mat_sharded = get_compile_time_arg_val(8) == 1;  // trans_mat pre-loaded in L1 shard
    constexpr bool cos_sin_sharded = get_compile_time_arg_val(9) == 1;    // cos/sin pre-loaded in L1 shard

    // Compile-time arg layout starting at index 10:
    //
    //  [10 ...]                      = TensorAccessorArgs for input   (always present)
    //  [input_args.next() ...]       = TensorAccessorArgs for cos     (only when !cos_sin_sharded)
    //  [cos_args.next()   ...]       = TensorAccessorArgs for sin     (only when !cos_sin_sharded)
    //  [sin_args.next()   ...]       = TensorAccessorArgs for trans_mat (only when !trans_mat_sharded)
    //
    // Because the factory only appends accessor args for non-sharded tensors, and
    // TensorAccessorArgs slots are tightly packed, we must chain the offsets only
    // for those that are actually present.  The if-constexpr blocks below ensure
    // that TensorAccessorArgs for a sharded tensor is never instantiated, which
    // avoids reading stale/garbage compile-time arg slots.

    constexpr uint32_t onetile = 1;

    // Compile-time arg layout starting at index 10:
    //   [10..]              = TensorAccessorArgs for input (always present)
    //   [+N_input..]        = TensorAccessorArgs for cos   (only when !cos_sin_sharded)
    //   [+N_cos..]          = TensorAccessorArgs for sin   (only when !cos_sin_sharded)
    //   [+N_sin..]          = TensorAccessorArgs for trans_mat (only when !trans_mat_sharded)
    constexpr auto input_args = TensorAccessorArgs<10>();
    constexpr uint32_t after_input = input_args.next_compile_time_args_offset();
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const auto s_input = TensorAccessor(input_args, src_addr, input_tile_bytes);

    // ------------------------------------------------------------------
    // Transformation matrix
    // ------------------------------------------------------------------
    if constexpr (trans_mat_sharded) {
        // Data is already in L1 at the globally-allocated CB address.
        // Just advance the CB protocol to signal compute.
        cb_reserve_back(trans_mat_cb_id, onetile);
        cb_push_back(trans_mat_cb_id, onetile);
    } else {
        // When cos_sin_sharded: trans_mat args begin right after input args.
        // When !cos_sin_sharded: trans_mat args begin after input + cos + sin args.
        if constexpr (cos_sin_sharded) {
            constexpr auto trans_mat_args = TensorAccessorArgs<after_input>();
            const uint32_t trans_mat_tile_bytes = get_tile_size(trans_mat_cb_id);
            const auto s_trans_mat = TensorAccessor(trans_mat_args, trans_mat_addr, trans_mat_tile_bytes);
            cb_reserve_back(trans_mat_cb_id, onetile);
            uint32_t trans_mat_l1_write_addr = get_write_ptr(trans_mat_cb_id);
            noc_async_read_tile(0, s_trans_mat, trans_mat_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(trans_mat_cb_id, onetile);
        } else {
            constexpr auto cos_args_for_tm = TensorAccessorArgs<after_input>();
            constexpr uint32_t after_cos = cos_args_for_tm.next_compile_time_args_offset();
            constexpr auto sin_args_for_tm = TensorAccessorArgs<after_cos>();
            constexpr uint32_t after_sin = sin_args_for_tm.next_compile_time_args_offset();
            constexpr auto trans_mat_args = TensorAccessorArgs<after_sin>();
            const uint32_t trans_mat_tile_bytes = get_tile_size(trans_mat_cb_id);
            const auto s_trans_mat = TensorAccessor(trans_mat_args, trans_mat_addr, trans_mat_tile_bytes);
            cb_reserve_back(trans_mat_cb_id, onetile);
            uint32_t trans_mat_l1_write_addr = get_write_ptr(trans_mat_cb_id);
            noc_async_read_tile(0, s_trans_mat, trans_mat_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(trans_mat_cb_id, onetile);
        }
    }

    // Input tile linear index: input[0, h, bt, w] = h * batch_t * Wt + bt * Wt + w
    // Cos/sin tile linear index (freq_per_head=true):  cs[0, h, bt, w] = h * batch_t * Wt + bt * Wt + w
    // Cos/sin tile linear index (freq_per_head=false): cs[0, 0, bt, w] = bt * Wt + w

    const uint32_t my_n_bt = batch_t_end - batch_t_start;

    // ------------------------------------------------------------------
    // Sharded cos/sin path: data is already in L1 per shard.
    // ------------------------------------------------------------------
    if constexpr (cos_sin_sharded) {
        // Each core's shard holds exactly Wt tiles for one batch-tile row.
        // Signal compute for each (head, bt) iteration — no NOC reads needed.
        for (uint32_t h = head_start; h < head_end; ++h) {
            for (uint32_t bt = batch_t_start; bt < batch_t_end; ++bt) {
                const uint32_t input_base = h * batch_t * Wt + bt * Wt;

                cb_reserve_back(cos_cb_id, Wt);
                cb_push_back(cos_cb_id, Wt);
                cb_reserve_back(sin_cb_id, Wt);
                cb_push_back(sin_cb_id, Wt);

                cb_reserve_back(input_cb_id, Wt);
                uint32_t input_l1_write_addr = get_write_ptr(input_cb_id);
                for (uint32_t w = 0; w < Wt; ++w) {
                    noc_async_read_tile(input_base + w, s_input, input_l1_write_addr);
                    input_l1_write_addr += input_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(input_cb_id, Wt);
            }
        }
    } else {
        // ------------------------------------------------------------------
        // DRAM cos/sin paths
        // ------------------------------------------------------------------
        constexpr auto cos_args = TensorAccessorArgs<after_input>();
        constexpr uint32_t after_cos = cos_args.next_compile_time_args_offset();
        constexpr auto sin_args = TensorAccessorArgs<after_cos>();

        const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
        const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
        const auto s_cos = TensorAccessor(cos_args, cos_addr, cos_tile_bytes);
        const auto s_sin = TensorAccessor(sin_args, sin_addr, sin_tile_bytes);

        // Fast path: cache all cos/sin tiles for this core's batch range once.
        if constexpr (!freq_per_head && !reload_cos_sin) {
            const uint32_t my_cs_tiles = my_n_bt * Wt;

            cb_reserve_back(cos_cb_id, my_cs_tiles);
            cb_reserve_back(sin_cb_id, my_cs_tiles);
            uint32_t cos_l1_write_addr = get_write_ptr(cos_cb_id);
            uint32_t sin_l1_write_addr = get_write_ptr(sin_cb_id);

            for (uint32_t bt = batch_t_start; bt < batch_t_end; ++bt) {
                const uint32_t cs_base = bt * Wt;
                for (uint32_t w = 0; w < Wt; ++w) {
                    noc_async_read_tile(cs_base + w, s_cos, cos_l1_write_addr);
                    cos_l1_write_addr += cos_tile_bytes;
                    noc_async_read_tile(cs_base + w, s_sin, sin_l1_write_addr);
                    sin_l1_write_addr += sin_tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cos_cb_id, my_cs_tiles);
            cb_push_back(sin_cb_id, my_cs_tiles);

            // Head loop: only input changes per head; cos/sin are held in the CB.
            for (uint32_t h = head_start; h < head_end; ++h) {
                for (uint32_t bt = batch_t_start; bt < batch_t_end; ++bt) {
                    const uint32_t input_base = h * batch_t * Wt + bt * Wt;

                    cb_reserve_back(input_cb_id, Wt);
                    uint32_t input_l1_write_addr = get_write_ptr(input_cb_id);
                    for (uint32_t w = 0; w < Wt; ++w) {
                        noc_async_read_tile(input_base + w, s_input, input_l1_write_addr);
                        input_l1_write_addr += input_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(input_cb_id, Wt);
                }
            }

            // The compute kernel pops the full cos/sin cache after all heads.
        } else {
            // Fallback path: re-read cos/sin for every (head, batch_tile) pair.
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
    }
}
