// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Compile-time args
constexpr uint32_t cb_in_act_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_scores_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_scores_rm_id = get_compile_time_arg_val(2);
constexpr uint32_t act_page_size = get_compile_time_arg_val(3);
constexpr uint32_t scores_page_size = get_compile_time_arg_val(4);
constexpr uint32_t scores_tile_size = get_compile_time_arg_val(5);  // BF16 CB tile = 2048 bytes
constexpr uint32_t num_cores_to_be_used = get_compile_time_arg_val(6);
constexpr uint32_t input_granularity = get_compile_time_arg_val(7);
constexpr uint32_t reduction_dim = get_compile_time_arg_val(8);
constexpr uint32_t reduction_dim_size = get_compile_time_arg_val(9);  // expert_k
constexpr uint32_t inner_num_tiles = get_compile_time_arg_val(10);
constexpr uint32_t reduction_num_tiles = get_compile_time_arg_val(11);
constexpr uint32_t num_tokens = get_compile_time_arg_val(12);  // tokens per device == TILE_HEIGHT
constexpr uint32_t cb_scores_rm_page_size = get_compile_time_arg_val(13);  // RM CB page (one token row)

// TensorAccessor CT args: activation starts at 14, scores starts after activation's args
constexpr uint32_t initial_ct_idx_act = 14;
constexpr uint32_t initial_ct_idx_scores = TensorAccessorArgs<initial_ct_idx_act>::next_compile_time_args_offset();

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t input_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t scores_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    // TensorAccessors
    constexpr auto act_tensor_args = TensorAccessorArgs<initial_ct_idx_act>();
    constexpr auto scores_tensor_args = TensorAccessorArgs<initial_ct_idx_scores>();
    auto act_accessor = TensorAccessor(act_tensor_args, input_address, act_page_size);
    // scores_accessor: each "page" = one token row (reduction_dim_size BF16 scores)
    auto scores_accessor = TensorAccessor(scores_tensor_args, scores_address, scores_page_size);

    ////////////////////////////////////////////////////////////////////////////
    // Prologue: Load all raw RM scores into scratch CB, then build
    // BF16 score tiles in cb_scores using BroadcastType::COL format.
    //
    // scores layout (ROW_MAJOR): [tokens, 1, seq, reduction_dim_size]
    // One page = one token row of reduction_dim_size BF16 scores (= cb_scores_rm_page_size bytes)
    //
    // Target: cb_scores holds reduction_dim_size tiles (one per expert e).
    // Each tile[e]: column 0 contains score[t, e] for token rows t=0..num_tokens-1.
    //               All other columns = 0 (required by BroadcastType::COL).
    //
    // BF16 32x32 tile face layout (4 faces, each 16x16):
    //   Face 0 (rows  0-15, cols  0-15): base offset 0   (uint16 index)
    //   Face 1 (rows  0-15, cols 16-31): base offset 256
    //   Face 2 (rows 16-31, cols  0-15): base offset 512
    //   Face 3 (rows 16-31, cols 16-31): base offset 768
    // Within a face, element at (row r, col c): index = r * 16 + c
    // So column-0 of row t:
    //   t < 16  → face 0 → uint16 index: t * 16
    //   t >= 16 → face 2 → uint16 index: 512 + (t-16) * 16
    ////////////////////////////////////////////////////////////////////////////

    // Step 1: Read all token rows from DRAM into scratch staging buffer
    cb_reserve_back(cb_scores_rm_id, num_tokens);
    uint32_t scores_rm_ptr = get_write_ptr(cb_scores_rm_id);
    for (uint32_t t = 0; t < num_tokens; ++t) {
        noc_async_read_page(t, scores_accessor, scores_rm_ptr + t * cb_scores_rm_page_size);
    }
    noc_async_read_barrier();

    // Step 2: Permute and to_layout scores; rm [token][expert] in uint16 units, row stride = reduction_dim_size
    cb_reserve_back(cb_scores_id, reduction_dim_size);
    uint32_t scores_write_ptr = get_write_ptr(cb_scores_id);

    volatile tt_l1_ptr uint16_t* scores_rm_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_rm_ptr);
    volatile tt_l1_ptr uint16_t* scores_tile_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_write_ptr);
    const uint32_t tile_u16_stride = scores_tile_size / sizeof(uint16_t);  // 1024 uint16 per tile

    // [token][1][s=1][k] -> [k][1][t][s_padded=32]
    for (uint32_t k = 0; k < reduction_dim_size; ++k) {
        volatile tt_l1_ptr uint16_t* expert_tile = scores_tile_u16 + k * tile_u16_stride;

        // Fill Face 0 (rows 0-15, col 0) for tokens t = 0..15
        for (uint32_t t = 0; t < 16 && t < num_tokens; ++t) {
            // score location in RM: t * (cb_scores_rm_page_size / 2) + k
            uint16_t score = scores_rm_u16[t * (cb_scores_rm_page_size / 2) + k];
            expert_tile[t * 16] = score;
        }
        // Fill Face 2 (rows 16-31, col 0) for tokens t = 16..num_tokens-1
        if (num_tokens > 16) {
            for (uint32_t t = 16; t < num_tokens; ++t) {
                uint16_t score = scores_rm_u16[t * (cb_scores_rm_page_size / 2) + k];
                expert_tile[512 + (t - 16) * 16] = score;
            }
        }
    }

    // Step 3: Release scores to compute kernel
    cb_push_back(cb_scores_id, reduction_dim_size);
    cb_push_back(cb_scores_rm_id, num_tokens);

    ////////////////////////////////////////////////////////////////////////////
    // Main loop: stream activation tiles into cb_in_act (same as original op)
    ////////////////////////////////////////////////////////////////////////////
    uint32_t l1_write_addr;
    uint32_t input_granularity_index = 0;

    for (uint32_t tiles_read = start_tiles_read; tiles_read < start_tiles_to_read; tiles_read += num_cores_to_be_used) {
        uint32_t read_tile_id;
        if constexpr (reduction_dim == 0) {
            read_tile_id = tiles_read;
        } else {
            read_tile_id = ((tiles_read / inner_num_tiles) * reduction_num_tiles) + (tiles_read % inner_num_tiles);
        }

        // Now reduce all tiles in the reduction dim. The first index is the
        // same as the output index. After that need to increment by the
        // size of the inner dimensions in tiles. E.g. for 130 tiles,
        // the increment is 130. If 4 tiles need to be reduced, then the
        // first core would access tiles at indices 0, 130, 260, 390, 64,
        // 64+130, 64+260, 64+390, 128, 128+130, 128+260, and 128+390.
        for (uint32_t j = 0; j < reduction_dim_size; ++j) {
            if (input_granularity_index == 0) {
                cb_reserve_back(cb_in_act_id, input_granularity);
                l1_write_addr = get_write_ptr(cb_in_act_id);
            }
            noc_async_read_page(read_tile_id, act_accessor, l1_write_addr);

            l1_write_addr += act_page_size;
            read_tile_id += inner_num_tiles;
            input_granularity_index++;

            if (input_granularity_index == input_granularity) {
                noc_async_read_barrier();
                cb_push_back(cb_in_act_id, input_granularity);
                input_granularity_index = 0;
            }
        }
    }
}
