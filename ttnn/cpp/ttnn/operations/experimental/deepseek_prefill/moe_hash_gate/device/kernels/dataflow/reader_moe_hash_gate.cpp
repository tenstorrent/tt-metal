// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Hash-gate reader. Two jobs per height tile:
//   1. Stream the gate logits tiles into cb_in_scores (consumed by compute's apply_score_func).
//   2. Fuse the DeepSeek-V4 hash lookup: for each token in the tile, read tid2eid[token_id] from DRAM
//      and place the n_activated expert ids into cb_out_indices in the exact face/column layout the
//      writer's gather() reads (row = token, cols [0, n_activated)). This replaces the top-k stage.
#include "ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/dataflow/moe_gate_common_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_in_scores_id = get_named_compile_time_arg_val("cb_in_scores");
    constexpr uint32_t cb_out_indices_id = get_named_compile_time_arg_val("cb_out_indices");
    constexpr uint32_t cb_input_ids_id = get_named_compile_time_arg_val("cb_input_ids");
    constexpr uint32_t cb_tid2eid_row_id = get_named_compile_time_arg_val("cb_tid2eid_row");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t scores_page_size = get_named_compile_time_arg_val("scores_page_size");
    constexpr uint32_t input_ids_page_size = get_named_compile_time_arg_val("input_ids_page_size");
    constexpr uint32_t tid2eid_page_size = get_named_compile_time_arg_val("tid2eid_page_size");
    constexpr uint32_t tid2eid_row_stride = get_named_compile_time_arg_val("tid2eid_row_stride");
    constexpr uint32_t n_activated_experts = get_named_compile_time_arg_val("n_activated_experts");
    constexpr uint32_t n_activated_expert_tiles = get_named_compile_time_arg_val("n_activated_expert_tiles");
    constexpr uint32_t tile_height = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t seq_len_tiles = get_named_compile_time_arg_val("seq_len_tiles");
    constexpr uint32_t remainder_tokens_per_tile = get_named_compile_time_arg_val("remainder_tokens_per_tile");

    constexpr auto scores_args = TensorAccessorArgs<0>();
    constexpr auto input_ids_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto tid2eid_args = TensorAccessorArgs<input_ids_args.next_compile_time_args_offset()>();

    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_ids_addr = get_arg_val<uint32_t>(1);
    const uint32_t tid2eid_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_height_tile = get_arg_val<uint32_t>(3);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(4);

    const auto scores_accessor = TensorAccessor(scores_args, scores_addr, scores_page_size);
    const auto input_ids_accessor = TensorAccessor(input_ids_args, input_ids_addr, input_ids_page_size);
    const auto tid2eid_accessor = TensorAccessor(tid2eid_args, tid2eid_addr, tid2eid_page_size);

    Noc noc;
    CircularBuffer cb_in_scores(cb_in_scores_id);
    CircularBuffer cb_out_indices(cb_out_indices_id);
    CircularBuffer cb_input_ids(cb_input_ids_id);
    CircularBuffer cb_tid2eid_row(cb_tid2eid_row_id);
    const uint32_t scores_tile_bytes = cb_in_scores.get_tile_size();

    // Reserve the scratch regions once and reuse across tiles (single-kernel scratch, no consumer).
    cb_input_ids.reserve_back(1);
    cb_tid2eid_row.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ids_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_input_ids.get_write_ptr());
    volatile tt_l1_ptr uint16_t* rows_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_tid2eid_row.get_write_ptr());

    // Scratch rows are strided by the DRAM-aligned size (tid2eid_row_stride) so each per-token read
    // lands at an aligned L1 destination; the uint16 stride between rows follows the same value.
    const uint32_t tid2eid_row_uint16 = tid2eid_row_stride / 2;  // uint16 entries per padded tid2eid row slot

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        uint32_t tokens_per_tile = ((height_tile + 1) % seq_len_tiles == 0) ? remainder_tokens_per_tile : tile_height;

        // 1. Stream the logits tiles for this height tile.
        uint32_t base_page = height_tile * width_tiles;
        for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
            cb_in_scores.reserve_back(1);
            noc.async_read(
                scores_accessor,
                cb_in_scores,
                scores_tile_bytes,
                {.page_id = base_page + width_tile},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in_scores.push_back(1);
        }

        // 2. Read this tile's token ids (one ROW_MAJOR page = tile_height uint32 ids).
        noc.async_read(
            input_ids_accessor, cb_input_ids, input_ids_page_size, {.page_id = height_tile}, {.offset_bytes = 0});
        noc.async_read_barrier();

        // 3. Look up tid2eid[token_id] for each token (scattered DRAM rows -> contiguous scratch).
        for (uint32_t token = 0; token < tokens_per_tile; token++) {
            uint32_t token_id = ids_ptr[token];
            noc.async_read(
                tid2eid_accessor,
                cb_tid2eid_row,
                tid2eid_page_size,
                {.page_id = token_id},
                {.offset_bytes = token * tid2eid_row_stride});
        }
        noc.async_read_barrier();

        // 4. Assemble the index tile: place n_activated expert ids per token at cols [0, n_activated)
        //    using the exact face/column addressing that the writer's gather() reads.
        cb_out_indices.reserve_back(n_activated_expert_tiles);
        volatile tt_l1_ptr uint16_t* idx_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_out_indices.get_write_ptr());

        // Zero the whole index tile so unused rows/cols are deterministic when written out.
        for (uint32_t i = 0; i < elements_per_tile; i++) {
            idx_ptr[i] = 0;
        }

        for (uint32_t token = 0; token < tokens_per_tile; token++) {
            uint32_t token_face_row = token % rows_per_face;
            uint32_t token_face_base = (token < rows_per_face) ? 0 : 2;
            uint32_t row_base = token * tid2eid_row_uint16;
            for (uint32_t expert = 0; expert < n_activated_experts; expert++) {
                uint32_t face_col = expert % columns_per_face;
                uint32_t face = token_face_base + (expert < columns_per_face ? 0 : 1);
                uint32_t offset = face * elements_per_face + token_face_row * columns_per_face + face_col;
                idx_ptr[offset] = rows_ptr[row_base + expert];
            }
        }
        cb_out_indices.push_back(n_activated_expert_tiles);
    }
}
