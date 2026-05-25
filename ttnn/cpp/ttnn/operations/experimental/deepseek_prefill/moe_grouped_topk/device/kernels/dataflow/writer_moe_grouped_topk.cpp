// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for moe_grouped_topk with iterative max (replaces bitonic sort).
// Performs 3-stage topk on TILE layout biased scores, constructs output indices
// tile (UINT16), gathers sigmoid scores, and writes results to DRAM.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_ELEMENTS = 1024;  // 32 * 32
constexpr uint32_t FACE_SIZE = 256;       // 16 * 16
constexpr uint32_t COLS_PER_TILE = 32;
constexpr uint32_t ROWS_PER_FACE = 16;
constexpr uint32_t COLS_PER_FACE = 16;

union FloatUint32 {
    float f;
    uint32_t u;
};

// Read a float element from contiguous tiles in L1.
// tiles_base points to width_tiles contiguous tiles (as uint32_t*).
FORCE_INLINE float read_f32_tile_element(
    volatile tt_l1_ptr uint32_t* tiles_base, uint32_t intra_tile_row, uint32_t col) {
    uint32_t tile_col_idx = col >> 5;
    uint32_t intra_col = col & 31;
    uint32_t face = ((intra_tile_row >> 4) << 1) + (intra_col >> 4);
    uint32_t face_row = intra_tile_row & 15;
    uint32_t face_col = intra_col & 15;
    uint32_t offset = tile_col_idx * TILE_ELEMENTS + face * FACE_SIZE + face_row * COLS_PER_FACE + face_col;
    FloatUint32 conv;
    conv.u = tiles_base[offset];
    return conv.f;
}

// Read a uint32 element from contiguous tiles.
FORCE_INLINE uint32_t
read_u32_tile_element(volatile tt_l1_ptr uint32_t* tiles_base, uint32_t intra_tile_row, uint32_t col) {
    uint32_t tile_col_idx = col >> 5;
    uint32_t intra_col = col & 31;
    uint32_t face = ((intra_tile_row >> 4) << 1) + (intra_col >> 4);
    uint32_t face_row = intra_tile_row & 15;
    uint32_t face_col = intra_col & 15;
    uint32_t offset = tile_col_idx * TILE_ELEMENTS + face * FACE_SIZE + face_row * COLS_PER_FACE + face_col;
    return tiles_base[offset];
}

// Write a uint32 value to a tile element position.
FORCE_INLINE void write_u32_tile_element(
    volatile tt_l1_ptr uint32_t* tiles_base, uint32_t intra_tile_row, uint32_t col, uint32_t value) {
    uint32_t tile_col_idx = col >> 5;
    uint32_t intra_col = col & 31;
    uint32_t face = ((intra_tile_row >> 4) << 1) + (intra_col >> 4);
    uint32_t face_row = intra_tile_row & 15;
    uint32_t face_col = intra_col & 15;
    uint32_t offset = tile_col_idx * TILE_ELEMENTS + face * FACE_SIZE + face_row * COLS_PER_FACE + face_col;
    tiles_base[offset] = value;
}

// Write a uint16 value into a UINT16 tile at face-aware position.
FORCE_INLINE void write_u16_tile_element(
    volatile tt_l1_ptr uint16_t* tiles_base, uint32_t intra_tile_row, uint32_t col, uint16_t value) {
    uint32_t tile_col_idx = col >> 5;
    uint32_t intra_col = col & 31;
    uint32_t face = ((intra_tile_row >> 4) << 1) + (intra_col >> 4);
    uint32_t face_row = intra_tile_row & 15;
    uint32_t face_col = intra_col & 15;
    uint32_t offset = tile_col_idx * TILE_ELEMENTS + face * FACE_SIZE + face_row * COLS_PER_FACE + face_col;
    tiles_base[offset] = value;
}

void generate_reduce_scalar(
    const uint32_t cb_reduce_ones_scalar, const uint32_t packed_scalar, const uint32_t n_activated_experts) {
    cb_reserve_back(cb_reduce_ones_scalar, 1);

    uint32_t write_addr = get_write_ptr(cb_reduce_ones_scalar);
    tt_l1_ptr uint32_t* write_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(write_addr);
    for (uint32_t i = 0; i < n_activated_experts; i++) {
        write_ptr[i] = packed_scalar;
        if (i > ROWS_PER_FACE - 1) {
            write_ptr[i + FACE_SIZE - COLS_PER_FACE + 1] = packed_scalar;
        }
    }
    for (uint32_t i = n_activated_experts; i < COLS_PER_TILE; i++) {
        write_ptr[i] = 0;
        if (i == ROWS_PER_FACE) {
            noc_async_read(
                get_noc_addr(MEM_ZEROS_BASE),
                write_addr + FACE_SIZE * sizeof(uint32_t),
                COLS_PER_FACE * sizeof(uint32_t));
        }
    }
    uint32_t face_3_write_addr = write_addr + 2 * FACE_SIZE * sizeof(uint32_t);
    uint32_t face_4_write_addr = write_addr + 3 * FACE_SIZE * sizeof(uint32_t);
    noc_async_read_barrier();
    noc_async_read(get_noc_addr(write_addr), face_3_write_addr, COLS_PER_FACE * sizeof(uint32_t));
    noc_async_read(
        get_noc_addr(write_addr + FACE_SIZE * sizeof(uint32_t)), face_4_write_addr, COLS_PER_FACE * sizeof(uint32_t));
    noc_async_read_barrier();

    cb_push_back(cb_reduce_ones_scalar, 1);
}

void write_single_scalar(const uint32_t cb_scalar, const uint32_t packed_scalar) {
    cb_reserve_back(cb_scalar, 1);
    uint32_t write_addr = get_write_ptr(cb_scalar);
    tt_l1_ptr uint32_t* write_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(write_addr);
    write_ptr[0] = packed_scalar;
    cb_push_back(cb_scalar, 1);
}

void zero_buffer(uint32_t write_addr, int bytes) {
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        uint32_t curr_bytes = std::min(bytes, MEM_ZEROS_SIZE);
        noc_async_read(zeros_noc_addr, write_addr, curr_bytes);
        write_addr += curr_bytes;
        bytes -= curr_bytes;
    }
    noc_async_read_barrier();
}

void kernel_main() {
    constexpr uint32_t cb_out_weights = get_named_compile_time_arg_val("cb_out_weights");
    constexpr uint32_t cb_out_indices = get_named_compile_time_arg_val("cb_out_indices");
    constexpr uint32_t cb_sigmoid_scores = get_named_compile_time_arg_val("cb_sigmoid_scores");
    constexpr uint32_t cb_biased_scores = get_named_compile_time_arg_val("cb_biased_scores");
    constexpr uint32_t cb_gathered_sigmoid = get_named_compile_time_arg_val("cb_gathered_sigmoid");
    constexpr uint32_t cb_reduce_ones_scalar = get_named_compile_time_arg_val("cb_reduce_ones_scalar");
    constexpr uint32_t cb_epsilon_scalar = get_named_compile_time_arg_val("cb_epsilon_scalar");
    constexpr uint32_t cb_route_scale_scalar = get_named_compile_time_arg_val("cb_route_scale_scalar");
    constexpr uint32_t scores_page_size = get_named_compile_time_arg_val("scores_page_size");
    constexpr uint32_t weights_page_size = get_named_compile_time_arg_val("weights_page_size");
    constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t tile_height = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t topk_groups = get_named_compile_time_arg_val("topk_groups");
    constexpr uint32_t n_groups = get_named_compile_time_arg_val("n_groups");
    constexpr uint32_t summed_experts_per_group = get_named_compile_time_arg_val("summed_experts_per_group");
    constexpr uint32_t n_activated_experts = get_named_compile_time_arg_val("n_activated_experts");
    constexpr uint32_t packed_one_scalar = get_named_compile_time_arg_val("packed_one_scalar");
    constexpr uint32_t packed_epsilon = get_named_compile_time_arg_val("packed_epsilon");
    constexpr uint32_t packed_route_scale = get_named_compile_time_arg_val("packed_route_scale");
    constexpr uint32_t seq_len_tiles = get_named_compile_time_arg_val("seq_len_tiles");
    constexpr uint32_t remainder_tokens_per_tile = get_named_compile_time_arg_val("remainder_tokens_per_tile");
    constexpr uint32_t n_activated_expert_tiles = get_named_compile_time_arg_val("n_activated_expert_tiles");

    constexpr uint32_t group_size = experts / n_groups;  // 32

    const uint32_t weights_addr = get_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_height_tile = get_arg_val<uint32_t>(2);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(3);

    constexpr auto weights_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();

    const auto weights_accessor = TensorAccessor(weights_args, weights_addr, weights_page_size);
    const auto indices_accessor = TensorAccessor(indices_args, indices_addr, indices_page_size);

    constexpr uint32_t NEG_INF_U32 = 0xFF800000u;
    FloatUint32 neg_inf_conv;
    neg_inf_conv.u = NEG_INF_U32;
    const float NEG_INF_F = neg_inf_conv.f;

    // One-time setup: generate scalar tiles for compute kernel's normalize step
    generate_reduce_scalar(cb_reduce_ones_scalar, packed_one_scalar, n_activated_experts);
    write_single_scalar(cb_epsilon_scalar, packed_epsilon);
    write_single_scalar(cb_route_scale_scalar, packed_route_scale);

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        uint32_t tokens_per_tile = ((height_tile + 1) % seq_len_tiles == 0) ? remainder_tokens_per_tile : tile_height;

        // Wait for biased scores and sigmoid scores from compute
        cb_wait_front(cb_biased_scores, width_tiles);
        cb_wait_front(cb_sigmoid_scores, width_tiles);

        volatile tt_l1_ptr uint32_t* biased_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_biased_scores));
        volatile tt_l1_ptr uint32_t* sigmoid_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_sigmoid_scores));

        // Reserve output tile buffers
        cb_reserve_back(cb_out_indices, 1);
        cb_reserve_back(cb_gathered_sigmoid, 1);

        uint32_t indices_tile_addr = get_write_ptr(cb_out_indices);
        uint32_t gathered_tile_addr = get_write_ptr(cb_gathered_sigmoid);

        // Zero the output tiles (only n_activated_experts columns per row will be filled)
        zero_buffer(indices_tile_addr, indices_page_size);
        zero_buffer(gathered_tile_addr, scores_page_size);

        volatile tt_l1_ptr uint16_t* indices_tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(indices_tile_addr);
        volatile tt_l1_ptr uint32_t* gathered_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gathered_tile_addr);

        for (uint32_t token = 0; token < tokens_per_tile; token++) {
            // --- Stage 1: Compute group scores (sum of top-summed_experts_per_group per group) ---
            // We do NOT mask the biased_scores here; stage 3 needs unmodified data.
            // Instead we track indices locally to avoid double-counting.
            float group_scores[8];
            for (uint32_t g = 0; g < n_groups; g++) {
                float score_sum = 0.0f;
                uint32_t group_col_base = g * group_size;
                uint32_t masked_cols[2] = {0xFFFFFFFF, 0xFFFFFFFF};

                for (uint32_t s = 0; s < summed_experts_per_group; s++) {
                    float max_val = NEG_INF_F;
                    uint32_t max_col = 0;
                    for (uint32_t col = 0; col < group_size; col++) {
                        bool skip = false;
                        for (uint32_t m = 0; m < s; m++) {
                            if (col == masked_cols[m]) {
                                skip = true;
                                break;
                            }
                        }
                        if (skip) {
                            continue;
                        }
                        float val = read_f32_tile_element(biased_base, token, group_col_base + col);
                        if (val > max_val) {
                            max_val = val;
                            max_col = col;
                        }
                    }
                    score_sum += max_val;
                    masked_cols[s] = max_col;
                }
                group_scores[g] = score_sum;
            }

            // --- Stage 2: Find top-topk_groups groups ---
            uint32_t winning_groups[4];
            for (uint32_t ki = 0; ki < topk_groups; ki++) {
                float max_val = NEG_INF_F;
                uint32_t max_idx = 0;
                for (uint32_t g = 0; g < n_groups; g++) {
                    if (group_scores[g] > max_val) {
                        max_val = group_scores[g];
                        max_idx = g;
                    }
                }
                winning_groups[ki] = max_idx;
                group_scores[max_idx] = NEG_INF_F;
            }

            // --- Stage 3: Find top-n_activated_experts experts from winning groups ---
            // Scan biased_scores (unmodified) across all winning groups' columns.
            // Use in-place masking here since we're done reading this token row for scoring.
            uint16_t final_indices[8];
            for (uint32_t ki = 0; ki < n_activated_experts; ki++) {
                float max_val = NEG_INF_F;
                uint32_t max_global_col = 0;
                for (uint32_t wgi = 0; wgi < topk_groups; wgi++) {
                    uint32_t g = winning_groups[wgi];
                    uint32_t group_col_base = g * group_size;
                    for (uint32_t col = 0; col < group_size; col++) {
                        float val = read_f32_tile_element(biased_base, token, group_col_base + col);
                        if (val > max_val) {
                            max_val = val;
                            max_global_col = group_col_base + col;
                        }
                    }
                }
                final_indices[ki] = (uint16_t)max_global_col;
                write_u32_tile_element(biased_base, token, max_global_col, NEG_INF_U32);
            }

            // --- Write indices into output UINT16 tile (face-aware) ---
            for (uint32_t ki = 0; ki < n_activated_experts; ki++) {
                write_u16_tile_element(indices_tile, token, ki, final_indices[ki]);
            }

            // --- Gather sigmoid scores for the selected experts ---
            for (uint32_t ki = 0; ki < n_activated_experts; ki++) {
                uint32_t expert_col = final_indices[ki];
                uint32_t sigmoid_val = read_u32_tile_element(sigmoid_base, token, expert_col);
                write_u32_tile_element(gathered_tile, token, ki, sigmoid_val);
            }
        }

        // Signal compute kernel that gathered sigmoid is ready
        cb_push_back(cb_gathered_sigmoid, 1);

        // Write indices tile to DRAM
        noc_async_write_page(height_tile, indices_accessor, indices_tile_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out_indices, 1);

        // Release input tiles
        cb_pop_front(cb_biased_scores, width_tiles);
        cb_pop_front(cb_sigmoid_scores, width_tiles);

        // Wait for normalized weights from compute, write to DRAM
        cb_wait_front(cb_out_weights, 1);
        noc_async_write_page(height_tile, weights_accessor, get_read_ptr(cb_out_weights));
        noc_async_write_barrier();
        cb_pop_front(cb_out_weights, 1);
    }
}
