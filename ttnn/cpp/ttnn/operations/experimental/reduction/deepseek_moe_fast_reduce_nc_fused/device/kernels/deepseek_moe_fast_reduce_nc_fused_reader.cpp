// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/dprint_pages.h"

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
constexpr uint32_t num_tokens = get_compile_time_arg_val(12);
constexpr uint32_t num_tokens_x32 = get_compile_time_arg_val(13);          // tokens per device == TILE_HEIGHT
constexpr uint32_t cb_scores_rm_page_size = get_compile_time_arg_val(14);  // RM CB page (one token row)
constexpr uint32_t expert_indices_page_size = get_compile_time_arg_val(15);
constexpr uint32_t expert_mapping_page_size = get_compile_time_arg_val(16);
constexpr uint32_t cluster_axis = get_compile_time_arg_val(17);
constexpr uint32_t cb_expert_indices_id = get_compile_time_arg_val(18);
constexpr uint32_t cb_expert_mapping_id = get_compile_time_arg_val(19);
constexpr uint32_t expert_indices_cb_page_size = get_compile_time_arg_val(20);  // L1-aligned CB stride
constexpr uint32_t expert_mapping_cb_page_size = get_compile_time_arg_val(21);  // L1-aligned CB stride
constexpr uint32_t expert_indices_num_pages = get_compile_time_arg_val(22);
constexpr uint32_t expert_mapping_num_pages = get_compile_time_arg_val(23);
constexpr uint32_t mesh_cols = get_compile_time_arg_val(24);  // mesh_shape[1]; needed to decode linearized device ids
constexpr uint32_t num_shared_experts = get_compile_time_arg_val(25);
// Host packs the BF16 scale (upper 16 bits of float32) into the low 16 bits of this CT arg.
constexpr uint16_t shared_expert_scale_bf16 = static_cast<uint16_t>(get_compile_time_arg_val(26) & 0xFFFF);

constexpr uint32_t num_routed_experts = reduction_dim_size - num_shared_experts;

constexpr uint32_t face2_offset = 512;  // face 2 starts 512 elements into tile

// TensorAccessor CT args, chained:
//   activation @ 27, scores @ next, expert_indices @ next, expert_mapping @ next.
constexpr uint32_t initial_ct_idx_act = 27;
constexpr uint32_t initial_ct_idx_scores = TensorAccessorArgs<initial_ct_idx_act>::next_compile_time_args_offset();
constexpr uint32_t initial_ct_idx_expert_indices =
    TensorAccessorArgs<initial_ct_idx_scores>::next_compile_time_args_offset();
constexpr uint32_t initial_ct_idx_expert_mapping =
    TensorAccessorArgs<initial_ct_idx_expert_indices>::next_compile_time_args_offset();

void kernel_main() {
    Noc noc;
    CircularBuffer cb_in_act(cb_in_act_id);
    CircularBuffer cb_scores(cb_scores_id);
    CircularBuffer cb_scores_rm(cb_scores_rm_id);
    CircularBuffer cb_expert_indices(cb_expert_indices_id);
    CircularBuffer cb_expert_mapping(cb_expert_mapping_id);

    uint32_t arg_idx = 0;
    const uint32_t input_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t scores_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    // Mesh coordinate of the executing device (row, col). Used together with cluster_axis to
    // decide, per (token, k), whether the expert routed for that slot lives on this device's
    // cluster axis (same column when cluster_axis==0, same row when cluster_axis==1).
    const uint32_t mesh_coord_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mesh_coord_col = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t expert_indices_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t expert_mapping_address = get_arg_val<uint32_t>(arg_idx++);

    // TensorAccessors
    constexpr auto act_tensor_args = TensorAccessorArgs<initial_ct_idx_act>();
    constexpr auto scores_tensor_args = TensorAccessorArgs<initial_ct_idx_scores>();
    constexpr auto expert_indices_tensor_args = TensorAccessorArgs<initial_ct_idx_expert_indices>();
    constexpr auto expert_mapping_tensor_args = TensorAccessorArgs<initial_ct_idx_expert_mapping>();
    auto act_accessor = TensorAccessor(act_tensor_args, input_address, act_page_size);
    // scores_accessor: each "page" = one token row (reduction_dim_size BF16 scores)
    auto scores_accessor = TensorAccessor(scores_tensor_args, scores_address, scores_page_size);
    auto expert_indices_accessor =
        TensorAccessor(expert_indices_tensor_args, expert_indices_address, expert_indices_page_size);
    auto expert_mapping_accessor =
        TensorAccessor(expert_mapping_tensor_args, expert_mapping_address, expert_mapping_page_size);

    ////////////////////////////////////////////////////////////////////////////
    // Prologue 0: Mirror the full expert_indices_tensor and expert_mapping_tensor
    // into their dedicated L1 CBs. Both tensors are small enough to fit entirely;
    // pulling the data up front lets downstream consumers index them without
    // re-issuing DRAM reads on the hot path. Each CB is single-buffered with one
    // slot per DRAM page (host-side sizing in the program factory).
    ////////////////////////////////////////////////////////////////////////////

    cb_expert_indices.reserve_back(expert_indices_num_pages);
    uint32_t expert_indices_ptr = cb_expert_indices.get_write_ptr();
    for (uint32_t p = 0; p < expert_indices_num_pages; ++p) {
        noc.async_read(
            expert_indices_accessor,
            CoreLocalMem<uint32_t>(expert_indices_ptr + p * expert_indices_cb_page_size),
            expert_indices_page_size,
            {.page_id = p},
            {});
    }

    cb_expert_mapping.reserve_back(expert_mapping_num_pages);
    uint32_t expert_mapping_ptr = cb_expert_mapping.get_write_ptr();
    for (uint32_t p = 0; p < expert_mapping_num_pages; ++p) {
        noc.async_read(
            expert_mapping_accessor,
            CoreLocalMem<uint32_t>(expert_mapping_ptr + p * expert_mapping_cb_page_size),
            expert_mapping_page_size,
            {.page_id = p},
            {});
    }
    noc.async_read_barrier();
    cb_expert_indices.push_back(expert_indices_num_pages);
    cb_expert_mapping.push_back(expert_mapping_num_pages);

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
    cb_scores_rm.reserve_back(num_tokens);
    uint32_t scores_rm_ptr = cb_scores_rm.get_write_ptr();
    for (uint32_t t = 0; t < num_tokens; ++t) {
        noc.async_read(
            scores_accessor,
            CoreLocalMem<uint32_t>(scores_rm_ptr + t * cb_scores_rm_page_size),
            scores_page_size,
            {.page_id = t},
            {});
    }
    noc.async_read_barrier();

    // Step 2: Permute and to_layout scores; rm [token][expert] in uint16 units, row stride = reduction_dim_size
    cb_scores.reserve_back(reduction_dim_size);
    uint32_t scores_write_ptr = cb_scores.get_write_ptr();

    volatile tt_l1_ptr uint16_t* scores_rm_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_rm_ptr);
    volatile tt_l1_ptr uint16_t* scores_tile_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scores_write_ptr);
    volatile tt_l1_ptr uint16_t* expert_indices_u16 =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(expert_indices_ptr);
    volatile tt_l1_ptr uint16_t* expert_mapping_u16 =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(expert_mapping_ptr);
    const uint32_t tile_u16_stride = scores_tile_size / sizeof(uint16_t);  // 1024 uint16 per tile

    // ---- On-axis predicate setup --------------------------------------------------------------
    // For each (token, k) the prologue checks whether the expert this slot routes to lives on
    // this device's cluster axis. If it doesn't, the score is zeroed so the compute kernel's
    // weighted sum drops the contribution; otherwise the original score is written through.
    //
    // expert_id    = expert_indices_u16[t * indices_stride + k]
    // owner_device = expert_mapping_u16[mapping_row * mapping_stride + expert_id]   (linearized)
    // owner_row    = owner_device / mesh_cols
    // owner_col    = owner_device % mesh_cols
    // cluster_axis == 0: cluster is a column → on_axis iff owner_col == mesh_coord_col
    // cluster_axis == 1: cluster is a row    → on_axis iff owner_row == mesh_coord_row
    //
    // Note on the zero literal: the score tiles hold BF16 values addressed as uint16. BF16 +0.0
    // is bit pattern 0x0000, so writing static_cast<uint16_t>(0) is bit-identical to BF16 zero.
    constexpr uint16_t bf16_zero = 0x0000;
    const uint32_t indices_page_stride_u16 = expert_indices_cb_page_size / sizeof(uint16_t);
    const uint32_t mapping_page_stride_u16 = expert_mapping_cb_page_size / sizeof(uint16_t);
    const uint32_t local_device_idx = mesh_coord_row * mesh_cols + mesh_coord_col;
    // If the mapping was supplied as a single replicated page (shape [1, experts]) every device
    // sees the same row, so just read row 0. If it was supplied per dispatch-device (shape
    // [devices, experts] from map_shared_experts) read this device's row.
    const uint32_t mapping_row = (expert_mapping_num_pages == 1) ? 0u : local_device_idx;
    volatile tt_l1_ptr uint16_t* mapping_row_base = expert_mapping_u16 + mapping_row * mapping_page_stride_u16;

    auto compute_on_axis = [&](uint32_t t, uint32_t k) -> bool {
        const uint16_t expert_id = expert_indices_u16[t * indices_page_stride_u16 + k];
        const uint32_t owner_device = mapping_row_base[expert_id];
        if constexpr (cluster_axis == 0) {
            return (owner_device % mesh_cols) == mesh_coord_col;
        } else {
            return (owner_device / mesh_cols) == mesh_coord_row;
        }
    };

    // [token][1][s=1][k] -> [k][1][t][s_padded=32]
    for (uint32_t k = 0; k < reduction_dim_size; ++k) {
        volatile tt_l1_ptr uint16_t* expert_tile = scores_tile_u16 + k * tile_u16_stride;
        const bool is_shared_expert = (k >= num_routed_experts);

        // Fill Face 0 (rows 0-15, col 0) for tokens t = 0..15
        for (uint32_t t = 0; t < 16 && t < num_tokens; ++t) {
            if (is_shared_expert) {
                expert_tile[t * 16] = shared_expert_scale_bf16;
            } else {
                // score location in RM: t * (cb_scores_rm_page_size / 2) + k
                const uint16_t score = scores_rm_u16[t * (cb_scores_rm_page_size / 2) + k];
                expert_tile[t * 16] = compute_on_axis(t, k) ? score : bf16_zero;
            }
        }
        // Fill Face 2 (rows 16-31, col 0) for tokens t = 16..num_tokens-1
        if (num_tokens > 16) {
            for (uint32_t t = 16; t < num_tokens; ++t) {
                if (is_shared_expert) {
                    expert_tile[face2_offset + (t - 16) * 16] = shared_expert_scale_bf16;
                } else {
                    const uint16_t score = scores_rm_u16[t * (cb_scores_rm_page_size / 2) + k];
                    expert_tile[face2_offset + (t - 16) * 16] = compute_on_axis(t, k) ? score : bf16_zero;
                }
            }
        }
    }
    if ((num_tokens < num_tokens_x32) && (num_tokens < 16)) {
        // Fill remaining Face 0 with BF16 +0.0 (bit pattern 0x0000).
        for (uint32_t k = 0; k < reduction_dim_size; ++k) {
            volatile tt_l1_ptr uint16_t* expert_tile = scores_tile_u16 + k * tile_u16_stride;
            for (uint32_t t = num_tokens; t < 16; ++t) {
                expert_tile[t * 16] = bf16_zero;
            }
        }
    }
    if (num_tokens < num_tokens_x32) {
        // Fill remaining Face 2 with BF16 +0.0. Clamp the start to 16 so that the
        // (t - 16) row offset cannot underflow for num_tokens < 16 cases.
        const uint32_t face_2_start = num_tokens < 16 ? 16 : num_tokens;
        for (uint32_t k = 0; k < reduction_dim_size; ++k) {
            volatile tt_l1_ptr uint16_t* expert_tile = scores_tile_u16 + k * tile_u16_stride;
            for (uint32_t t = face_2_start; t < 32; ++t) {
                expert_tile[face2_offset + (t - 16) * 16] = bf16_zero;
            }
        }
    }

    // Step 3: Release scores to compute kernel
    cb_scores.push_back(reduction_dim_size);
    cb_scores_rm.push_back(num_tokens);

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
                cb_in_act.reserve_back(input_granularity);
                l1_write_addr = cb_in_act.get_write_ptr();
            }
            noc.async_read(
                act_accessor, CoreLocalMem<uint32_t>(l1_write_addr), act_page_size, {.page_id = read_tile_id}, {});

            l1_write_addr += act_page_size;
            read_tile_id += inner_num_tiles;
            input_granularity_index++;

            if (input_granularity_index == input_granularity) {
                noc.async_read_barrier();
                cb_in_act.push_back(input_granularity);
                input_granularity_index = 0;
            }
        }
    }
}
