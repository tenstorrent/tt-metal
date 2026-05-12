// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint_pages.h"

void print_tile_rows(
    uint32_t cb_idx,
    uint32_t tile_idx,
    bool untilize = false,
    uint16_t start_row = 0,
    uint16_t end_row = 32,
    uint8_t start_col = 0,
    uint8_t end_col = 32) {
    DPRINT << "cb_idx: " << cb_idx << " tile_idx: " << tile_idx << ENDL();
    DEVICE_PRINT("cb_idx: {} tile_idx: {}\n", cb_idx, tile_idx);
    DPRINT << "======" << ENDL();
    DEVICE_PRINT("======\n");
    for (uint16_t r = start_row; r < end_row; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_idx,
                      tile_idx,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)start_col,
                          .w1 = (uint8_t)end_col,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
        DEVICE_PRINT(
            "{} : {}\n",
            r,
            TileSlice(
                cb_idx,
                tile_idx,
                SliceRange{
                    .h0 = (uint8_t)r,
                    .h1 = (uint8_t)(r + 1),
                    .hs = (uint8_t)1,
                    .w0 = (uint8_t)start_col,
                    .w1 = (uint8_t)end_col,
                    .ws = (uint8_t)1},
                true,
                untilize));
    }
    DPRINT << "++++++" << ENDL();
    DEVICE_PRINT("++++++\n");
}

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

// TensorAccessor CT args, chained:
//   activation @ 25, scores @ next, expert_indices @ next, expert_mapping @ next.
constexpr uint32_t initial_ct_idx_act = 25;
constexpr uint32_t initial_ct_idx_scores = TensorAccessorArgs<initial_ct_idx_act>::next_compile_time_args_offset();
constexpr uint32_t initial_ct_idx_expert_indices =
    TensorAccessorArgs<initial_ct_idx_scores>::next_compile_time_args_offset();
constexpr uint32_t initial_ct_idx_expert_mapping =
    TensorAccessorArgs<initial_ct_idx_expert_indices>::next_compile_time_args_offset();

void kernel_main() {
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

    cb_reserve_back(cb_expert_indices_id, expert_indices_num_pages);
    uint32_t expert_indices_ptr = get_write_ptr(cb_expert_indices_id);
    for (uint32_t p = 0; p < expert_indices_num_pages; ++p) {
        noc_async_read_page(p, expert_indices_accessor, expert_indices_ptr + p * expert_indices_cb_page_size);
    }

    cb_reserve_back(cb_expert_mapping_id, expert_mapping_num_pages);
    uint32_t expert_mapping_ptr = get_write_ptr(cb_expert_mapping_id);
    for (uint32_t p = 0; p < expert_mapping_num_pages; ++p) {
        noc_async_read_page(p, expert_mapping_accessor, expert_mapping_ptr + p * expert_mapping_cb_page_size);
    }
    noc_async_read_barrier();
    cb_push_back(cb_expert_indices_id, expert_indices_num_pages);
    cb_push_back(cb_expert_mapping_id, expert_mapping_num_pages);

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
    // ---------------------------------------------------------------------------------------------

    // ---- Diagnostic DPRINTs (one core only) -----------------------------------------------------
    // Compare these against the python test for the same device:
    //   torch_expert_indices_global[t0:t0+4]   ↔ expert_id values printed below
    //   torch_expert_mapping[0, expert_id]     ↔ owner_device values printed below
    //   _on_axis_mask(...)                     ↔ on_axis bools printed below
    // If indices/owner/on_axis disagree with python for the same (t, k), the bug is in the read
    // path (page stride / num_pages / mapping pointer). If they agree, the bug is downstream.
    if (start_tiles_read == 0) {
        DPRINT << "[reader] mesh_coord=(" << mesh_coord_row << "," << mesh_coord_col << ") mesh_cols=" << mesh_cols
               << " cluster_axis=" << (uint32_t)cluster_axis << ENDL();
        DPRINT << "[reader] indices: num_pages=" << expert_indices_num_pages
               << " cb_page_size=" << expert_indices_cb_page_size << " stride_u16=" << indices_page_stride_u16
               << ENDL();
        DPRINT << "[reader] mapping: num_pages=" << expert_mapping_num_pages
               << " cb_page_size=" << expert_mapping_cb_page_size << " stride_u16=" << mapping_page_stride_u16
               << " mapping_row=" << mapping_row << ENDL();
        DPRINT << "[reader] mapping[0..15]=";
        for (uint32_t e = 0; e < 16; ++e) {
            DPRINT << (uint32_t)mapping_row_base[e] << " ";
        }
        DPRINT << ENDL();
        const uint32_t t_max = num_tokens < 4 ? num_tokens : 4;
        for (uint32_t t = 0; t < t_max; ++t) {
            DPRINT << "[reader] t=" << t << " ids=";
            for (uint32_t k = 0; k < reduction_dim_size; ++k) {
                DPRINT << (uint32_t)expert_indices_u16[t * indices_page_stride_u16 + k] << " ";
            }
            DPRINT << "owners=";
            for (uint32_t k = 0; k < reduction_dim_size; ++k) {
                const uint16_t expert_id = expert_indices_u16[t * indices_page_stride_u16 + k];
                DPRINT << (uint32_t)mapping_row_base[expert_id] << " ";
            }
            DPRINT << "on_axis=";
            uint32_t on_axis_count = 0;
            for (uint32_t k = 0; k < reduction_dim_size; ++k) {
                const bool oa = compute_on_axis(t, k);
                if (oa) {
                    ++on_axis_count;
                }
                DPRINT << (uint32_t)oa << " ";
            }
            DPRINT << "count=" << on_axis_count << ENDL();
        }
    }
    // ---------------------------------------------------------------------------------------------

    // [token][1][s=1][k] -> [k][1][t][s_padded=32]
    for (uint32_t k = 0; k < reduction_dim_size; ++k) {
        volatile tt_l1_ptr uint16_t* expert_tile = scores_tile_u16 + k * tile_u16_stride;

        // Fill Face 0 (rows 0-15, col 0) for tokens t = 0..15
        for (uint32_t t = 0; t < 16 && t < num_tokens; ++t) {
            // score location in RM: t * (cb_scores_rm_page_size / 2) + k
            const uint16_t score = scores_rm_u16[t * (cb_scores_rm_page_size / 2) + k];
            expert_tile[t * 16] = compute_on_axis(t, k) ? score : bf16_zero;
        }
        // Fill Face 2 (rows 16-31, col 0) for tokens t = 16..num_tokens-1
        if (num_tokens > 16) {
            for (uint32_t t = 16; t < num_tokens; ++t) {
                const uint16_t score = scores_rm_u16[t * (cb_scores_rm_page_size / 2) + k];
                expert_tile[512 + (t - 16) * 16] = compute_on_axis(t, k) ? score : bf16_zero;
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
                expert_tile[512 + (t - 16) * 16] = bf16_zero;
            }
        }
    }

    if (start_tiles_read == 0) {
        for (uint32_t k = 0; k < reduction_dim_size; ++k) {
            DPRINT << "k: " << k << "\n";
            print_tile_rows(cb_scores_id, k, true, 0, 32, 0, 1);
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
