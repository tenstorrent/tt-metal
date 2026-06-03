// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Forked from reader_rotary_embedding_llama_interleaved_start_id.cpp for KV-pad-aware indexed RoPE.
//
// The cos/sin caches are SP-sharded in block-cyclic order keyed by the per-device chunk size
// (chunk_local == Ht), so each device's shard already holds, in contiguous local-row order, the
// rope values for every global position that device will ever carry. This kernel only needs to
// derive WHERE in that shard the current chunk starts -- `update_idxt` -- exactly as the per-chip
// kv-cache writer does, then read cos/sin contiguously from there. The wrap of the boundary chip
// (older tokens finishing the current slab block, then newer tokens spilling into the next block)
// is absorbed by the shard layout, so the read stays contiguous.
//
// `kv_actual_global_t` is a common runtime arg (NOT in the program hash), so successive chunks with
// different prior KV lengths reuse one cached program.
void kernel_main() {
    uint32_t argrt = 0;
    uint32_t src_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t cos_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t sin_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t trans_mat_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_start = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_end = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_start = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_end = get_arg_val<uint32_t>(argrt++);

    // Common runtime args (same for all cores on this chip): per-chip inputs for the on-device
    // update_idxt derivation. kv_actual_global_t / my_sp_coord / sp_factor are kept out of the
    // program hash so the program is reused across chunks.
    const uint32_t kv_actual_global_t = get_common_arg_val<uint32_t>(0);
    const uint32_t my_sp_coord = get_common_arg_val<uint32_t>(1);
    const uint32_t sp_factor = get_common_arg_val<uint32_t>(2);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t n_heads = get_compile_time_arg_val(4);
    constexpr uint32_t Ht = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr bool freq_per_head = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t cos_Ht = get_compile_time_arg_val(8);
    constexpr uint32_t sin_Ht = get_compile_time_arg_val(9);
    constexpr uint32_t rotary_Ht = get_compile_time_arg_val(10);
    constexpr auto input_args = TensorAccessorArgs<11>();
    constexpr auto cos_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_mat_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    // Derive this chip's tile-row offset into its (block-cyclic) cos/sin shard from the global
    // valid KV length. Ht == chunk_local_t (per-device new chunk in tiles); chunk_global == sp*Ht.
    // Identical math to the per-chip kv-cache writer's update_idxt -- see writer_update_padded_kv_cache.
    const uint32_t chunk_global_t = sp_factor * Ht;
    const uint32_t boundary_slab_idx = chunk_global_t == 0 ? 0 : kv_actual_global_t / chunk_global_t;
    const uint32_t boundary_chip = Ht == 0 ? 0 : (kv_actual_global_t / Ht) % sp_factor;
    const uint32_t boundary_offset_t = Ht == 0 ? 0 : kv_actual_global_t % Ht;
    // From the current slab base, chips before the boundary advance a full slab, the boundary chip
    // advances by its pad offset, and chips after it stay at the base.
    const uint32_t update_idxt =
        boundary_slab_idx * Ht +
        (my_sp_coord < boundary_chip ? Ht : (my_sp_coord == boundary_chip ? boundary_offset_t : 0));

    const uint32_t rotary_seq_t_end = seq_t_end < rotary_Ht ? seq_t_end : rotary_Ht;
    const uint32_t my_rotary_seq_tiles = seq_t_start < rotary_seq_t_end ? rotary_seq_t_end - seq_t_start : 0;
    const uint32_t my_cos_sin_tiles = my_rotary_seq_tiles * Wt;

    constexpr uint32_t onetile = 1;
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const auto s0 = TensorAccessor(input_args, src_addr);

    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const auto s1 = TensorAccessor(cos_args, cos_addr);

    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const auto s2 = TensorAccessor(sin_args, sin_addr);

    const auto s3 = TensorAccessor(trans_mat_args, trans_mat_addr);

    uint32_t trans_mat_curr_idx = 0;

    // Read transformation matrix in CB (only once, because it will be reused)
    cb_reserve_back(trans_mat_cb_id, onetile);
    uint32_t trans_mat_l1_write_addr = get_write_ptr(trans_mat_cb_id);
    noc_async_read_page(trans_mat_curr_idx, s3, trans_mat_l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(trans_mat_cb_id, onetile);

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
        uint32_t sin_l1_write_addr = 0;
        uint32_t cos_l1_write_addr = 0;
#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            cb_reserve_back(sin_cb_id, my_cos_sin_tiles);
            cb_reserve_back(cos_cb_id, my_cos_sin_tiles);
            sin_l1_write_addr = get_write_ptr(sin_cb_id);
            cos_l1_write_addr = get_write_ptr(cos_cb_id);
        }
#endif

        // To make sure the sin/cos row are read only once
        uint32_t sin_cos_row_cnt = 0;
        bool done_sin_cos = false;

        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            for (uint32_t seq_tile = seq_t_start; seq_tile < rotary_seq_t_end; ++seq_tile) {
#if RELOAD_IMPL == 1
                cb_reserve_back(sin_cb_id, Wt);
                cb_reserve_back(cos_cb_id, Wt);
                uint32_t sin_l1_write_addr = get_write_ptr(sin_cb_id);
                uint32_t cos_l1_write_addr = get_write_ptr(cos_cb_id);
#endif

                cb_reserve_back(input_cb_id, Wt);
                uint32_t input_l1_write_addr = get_write_ptr(input_cb_id);
                uint32_t input_curr_idx = batch_id * n_heads * Ht * Wt + head_num * Ht * Wt + seq_tile * Wt;
                // Offset the cos/sin source index by update_idxt: the input local tile `seq_tile`
                // is rotated by the value at shard row (update_idxt + seq_tile).
                const uint32_t rope_seq_tile = update_idxt + seq_tile;
                uint32_t cos_curr_idx;
                uint32_t sin_curr_idx;
                if constexpr (freq_per_head) {
                    cos_curr_idx = head_num * cos_Ht * Wt + rope_seq_tile * Wt;
                    sin_curr_idx = head_num * sin_Ht * Wt + rope_seq_tile * Wt;
                } else {
                    cos_curr_idx = rope_seq_tile * Wt;
                    sin_curr_idx = rope_seq_tile * Wt;
                }
                for (uint32_t j = 0; j < Wt; ++j) {
                    // Read input into CB
                    noc_async_read_page(input_curr_idx, s0, input_l1_write_addr);
                    input_curr_idx++;
                    input_l1_write_addr += input_tile_bytes;

                    if (!done_sin_cos) {
                        noc_async_read_page(sin_curr_idx, s2, sin_l1_write_addr);
                        noc_async_read_page(cos_curr_idx, s1, cos_l1_write_addr);
                        sin_curr_idx++;
                        cos_curr_idx++;
                        sin_l1_write_addr += sin_tile_bytes;
                        cos_l1_write_addr += cos_tile_bytes;
                    }
                }

                noc_async_read_barrier();
                cb_push_back(input_cb_id, Wt);
#if RELOAD_IMPL == 1
                cb_push_back(sin_cb_id, Wt);
                cb_push_back(cos_cb_id, Wt);
#else

                if (!done_sin_cos) {
                    cb_push_back(sin_cb_id, Wt);
                    cb_push_back(cos_cb_id, Wt);

                    // Update sin_cos_row_cnt
                    sin_cos_row_cnt++;

                    if (sin_cos_row_cnt == my_rotary_seq_tiles) {
                        done_sin_cos = true;
                    }
                }
#endif
            }
        }
    }
}
