// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

constexpr uint32_t rows_per_face = 16;
constexpr uint32_t columns_per_face = 16;
constexpr uint32_t rows_per_tile = 32;
constexpr uint32_t columns_per_tile = 32;
constexpr uint32_t elements_per_face = rows_per_face * columns_per_face;  // 256
constexpr uint32_t elements_per_tile = rows_per_tile * columns_per_tile;  // 1024

namespace score_tile {
constexpr uint32_t bytes_per_element = 4;                                    // float32
constexpr uint32_t face_line_bytes = columns_per_face * bytes_per_element;   // 64
constexpr uint32_t face_size_bytes = elements_per_face * bytes_per_element;  // 1024
constexpr uint32_t tile_size_bytes = elements_per_tile * bytes_per_element;  // 4096
}  // namespace score_tile

namespace index_tile {
constexpr uint32_t bytes_per_element = 2;                                    // uint16
constexpr uint32_t face_line_bytes = columns_per_face * bytes_per_element;   // 32
constexpr uint32_t face_size_bytes = elements_per_face * bytes_per_element;  // 512
constexpr uint32_t tile_size_bytes = elements_per_tile * bytes_per_element;  // 2048
}  // namespace index_tile

FORCE_INLINE void generate_index_tile(
    const uint32_t cb_expert_index_template, const uint32_t index_write_addr, uint32_t start_expert_index) {
    cb_reserve_back(cb_expert_index_template, 1);
    for (uint32_t width_face = 0; width_face < 2; width_face++) {
        uint32_t current_index = start_expert_index + width_face * columns_per_face;
        uint32_t index_write_face_offset = index_write_addr + width_face * index_tile::face_size_bytes;

        uint64_t base_index_noc_addr = get_noc_addr(index_write_face_offset);

        volatile tt_l1_ptr uint32_t* index_cb_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_write_face_offset);
        for (uint32_t i = 0; i < columns_per_face / 2; i++) {
            index_cb_ptr[i] = (current_index + 1) << 16 | current_index;
            current_index += 2;
        }
        uint32_t dm_engine_index_write_offset = index_write_face_offset + index_tile::face_line_bytes;
        for (uint32_t i = 1; i < rows_per_face; i++) {
            noc_async_read(base_index_noc_addr, dm_engine_index_write_offset, index_tile::face_line_bytes);
            dm_engine_index_write_offset += index_tile::face_line_bytes;
        }
    }

    uint64_t index_noc_addr_base = get_noc_addr(index_write_addr);
    noc_async_read_barrier();

    noc_async_read(
        index_noc_addr_base, index_write_addr + 2 * index_tile::face_size_bytes, 2 * index_tile::face_size_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_expert_index_template, 1);
}

FORCE_INLINE void generate_index_tiles(
    const uint32_t cb_expert_index_template, uint32_t width_tiles, uint32_t page_size) {
    for (uint32_t i = 0; i < width_tiles; i++) {
        generate_index_tile(cb_expert_index_template, get_write_ptr(cb_expert_index_template), columns_per_tile * i);
    }
}

// Vertically along each tile, write index 0, ..., n_groups - 1
FORCE_INLINE void generate_group_indices_tiles(
    const uint32_t cb_group_index_template, uint32_t width_tiles, uint32_t n_groups) {
    cb_reserve_back(cb_group_index_template, 1);
    uint32_t base_write_addr = get_write_ptr(cb_group_index_template);
    volatile tt_l1_ptr uint32_t* write_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_write_addr);
    for (uint32_t group_index = 0; group_index < n_groups; group_index++) {
        for (uint32_t i = 0; i < columns_per_face / 2; i++) {
            write_ptr[i] = (group_index) << 16 | group_index;
        }
        if (group_index > rows_per_face - 1) {
            constexpr uint32_t skip_elements = index_tile::face_size_bytes / sizeof(uint32_t);
            write_ptr += skip_elements;
        } else {
            constexpr uint32_t skip_elements = index_tile::face_line_bytes / sizeof(uint32_t);
            write_ptr += skip_elements;
        }
    }
    uint64_t dm_engine_index_write_offset_face_1 = get_noc_addr(base_write_addr);
    uint64_t dm_engine_index_write_offset_face_3 = get_noc_addr(base_write_addr + 2 * index_tile::face_size_bytes);

    uint32_t face_2_l1_write_addr = base_write_addr + index_tile::face_size_bytes;
    uint32_t face_4_l1_write_addr = base_write_addr + 3 * index_tile::face_size_bytes;
    noc_async_read(dm_engine_index_write_offset_face_1, face_2_l1_write_addr, index_tile::face_size_bytes);
    noc_async_read(dm_engine_index_write_offset_face_3, face_4_l1_write_addr, index_tile::face_size_bytes);
    uint32_t tile_write_addr = base_write_addr + index_tile::tile_size_bytes;
    noc_async_read_barrier();
    cb_push_back(cb_group_index_template, 1);
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

FORCE_INLINE void generate_reduce_scalar(
    const uint32_t cb_reduce_ones_scalar, const uint32_t packed_scalar, const uint32_t n_activated_experts) {
    cb_reserve_back(cb_reduce_ones_scalar, 1);

    uint32_t write_addr = get_write_ptr(cb_reduce_ones_scalar);
    tt_l1_ptr uint32_t* write_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(write_addr);
    uint32_t scalar = packed_scalar;
    for (uint32_t i = 0; i < n_activated_experts; i++) {
        write_ptr[i] = scalar;
        if (i > rows_per_face - 1) {
            write_ptr[i + elements_per_face - columns_per_face + 1] = scalar;
        }
    }
    for (uint32_t i = n_activated_experts; i < rows_per_tile; i++) {
        write_ptr[i] = 0;
        if (i == rows_per_face) {
            noc_async_read(
                get_noc_addr(MEM_ZEROS_BASE), write_addr + score_tile::face_size_bytes, score_tile::face_line_bytes);
        }
    }
    uint32_t face_3_write_addr = write_addr + 2 * score_tile::face_size_bytes;
    uint32_t face_4_write_addr = write_addr + 3 * score_tile::face_size_bytes;
    noc_async_read_barrier();
    noc_async_read(get_noc_addr(write_addr), face_3_write_addr, score_tile::face_line_bytes);
    noc_async_read(
        get_noc_addr(write_addr + score_tile::face_size_bytes), face_4_write_addr, score_tile::face_line_bytes);
    noc_async_read_barrier();

    cb_push_back(cb_reduce_ones_scalar, 1);
}

FORCE_INLINE void generate_summed_experts_tiles(
    const uint32_t cb_top_experts_per_group,
    const uint32_t cb_sorted_group_scores,
    uint32_t width_tiles,
    uint32_t summed_experts_per_group,
    uint32_t tokens_per_tile) {
    // copy 0,...,summed_experts_per_group-1 rows from cb_sorted_group_scores to 0,...,summed_experts_per_group-1 tile
    // in cb_top_experts_per_group for each width_tile
    // for each group, copy the top experts_per_group rows to cb_top_experts_per_group
    // summed_experts_per_group has experts_per_group tiles, each tile is 32x32 fp32/bf16 elements, divided into 16x16
    // faces in our case, for now, width_tiles = n_groups

    // for each group, copy the top experts_per_group rows to cb_top_experts_per_group
    cb_reserve_back(cb_top_experts_per_group, summed_experts_per_group);
    for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
        cb_wait_front(cb_sorted_group_scores, 1);
        uint64_t group_sorted_tile_ptr = get_noc_addr(get_read_ptr(cb_sorted_group_scores));

        if (width_tile % 2 == 0) {
            for (uint32_t i = 0; i < summed_experts_per_group; i++) {
                noc_async_read(
                    group_sorted_tile_ptr + i * score_tile::face_line_bytes,
                    get_write_ptr(cb_top_experts_per_group) + i * score_tile::tile_size_bytes +
                        width_tile * score_tile::face_line_bytes,
                    score_tile::face_line_bytes);
                if (tokens_per_tile > rows_per_face) {
                    noc_async_read(
                        group_sorted_tile_ptr + score_tile::face_size_bytes + i * score_tile::face_line_bytes,
                        get_write_ptr(cb_top_experts_per_group) + score_tile::face_size_bytes +
                            i * score_tile::tile_size_bytes + width_tile * score_tile::face_line_bytes,
                        score_tile::face_line_bytes);
                }
            }
        } else {
            group_sorted_tile_ptr += score_tile::face_size_bytes * 2;
            for (uint32_t i = 0; i < summed_experts_per_group; i++) {
                noc_async_read(
                    group_sorted_tile_ptr + (rows_per_face - 1 - i) * score_tile::face_line_bytes,
                    get_write_ptr(cb_top_experts_per_group) + i * score_tile::tile_size_bytes +
                        width_tile * score_tile::face_line_bytes,
                    score_tile::face_line_bytes);
                if (tokens_per_tile > rows_per_face) {
                    noc_async_read(
                        group_sorted_tile_ptr + score_tile::face_size_bytes +
                            (rows_per_face - 1 - i) * score_tile::face_line_bytes,
                        get_write_ptr(cb_top_experts_per_group) + score_tile::face_size_bytes +
                            i * score_tile::tile_size_bytes + width_tile * score_tile::face_line_bytes,
                        score_tile::face_line_bytes);
                }
            }
        }
        noc_async_read_barrier();
        cb_pop_front(cb_sorted_group_scores, 1);
    }
    cb_push_back(cb_top_experts_per_group, summed_experts_per_group);
}

template <
    uint32_t cb_sorted_group_order,
    uint32_t cb_biased_scores,
    uint32_t cb_expert_index_template,
    uint32_t cb_winning_group_scores,
    uint32_t cb_winning_group_indices,
    uint32_t width_tiles,
    uint32_t topk_groups,
    uint32_t num_group_tiles>
FORCE_INLINE void generate_winning_group_tiles(uint32_t tokens_per_tile) {
    cb_wait_front(cb_biased_scores, width_tiles);
    cb_wait_front(cb_expert_index_template, width_tiles);
    cb_wait_front(cb_sorted_group_order, num_group_tiles);

    cb_reserve_back(cb_winning_group_scores, topk_groups);
    cb_reserve_back(cb_winning_group_indices, topk_groups);

    uint64_t scores_base_noc_addr = get_noc_addr(get_read_ptr(cb_biased_scores));
    uint64_t indices_base_noc_addr = get_noc_addr(get_read_ptr(cb_expert_index_template));
    uint32_t scores_dest_base_addr = get_write_ptr(cb_winning_group_scores);
    uint32_t indices_dest_base_addr = get_write_ptr(cb_winning_group_indices);

    volatile tt_l1_ptr uint16_t* sorted_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_sorted_group_order));

    for (uint32_t k = 0; k < topk_groups; k++) {
        uint32_t scores_dest_addr = scores_dest_base_addr + k * score_tile::tile_size_bytes;
        uint32_t indices_dest_addr = indices_dest_base_addr + k * index_tile::tile_size_bytes;

        uint32_t k_indices_offset_0_15;
        uint32_t k_indices_offset_16_31;

        if (k < rows_per_face) {
            k_indices_offset_0_15 = k * rows_per_face;
            k_indices_offset_16_31 = elements_per_face + k * rows_per_face;
        } else {
            k_indices_offset_0_15 = 2 * elements_per_face + (k - rows_per_face) * rows_per_face;
            k_indices_offset_16_31 = 3 * elements_per_face + (k - rows_per_face) * rows_per_face;
        }

        uint32_t last_row = (tokens_per_tile < rows_per_face) ? tokens_per_tile : rows_per_face;

#pragma GCC unroll 16
        for (uint32_t t = 0; t < last_row; t++) {
            uint16_t winning_group_idx = sorted_indices_ptr[k_indices_offset_0_15 + t];
            uint64_t score_src_tile_offset = winning_group_idx * score_tile::tile_size_bytes;
            uint64_t index_src_tile_offset = winning_group_idx * index_tile::tile_size_bytes;

            uint32_t score_row_fl1 = t * score_tile::face_line_bytes;
            uint32_t score_row_fl2 = score_tile::face_size_bytes + score_row_fl1;
            uint32_t index_row_fl1 = t * index_tile::face_line_bytes;
            uint32_t index_row_fl2 = index_tile::face_size_bytes + index_row_fl1;

            noc_async_read(
                scores_base_noc_addr + score_src_tile_offset + score_row_fl1,
                scores_dest_addr + score_row_fl1,
                score_tile::face_line_bytes);
            noc_async_read(
                indices_base_noc_addr + index_src_tile_offset + index_row_fl1,
                indices_dest_addr + index_row_fl1,
                index_tile::face_line_bytes);
            noc_async_read(
                scores_base_noc_addr + score_src_tile_offset + score_row_fl2,
                scores_dest_addr + score_row_fl2,
                score_tile::face_line_bytes);
            noc_async_read(
                indices_base_noc_addr + index_src_tile_offset + index_row_fl2,
                indices_dest_addr + index_row_fl2,
                index_tile::face_line_bytes);
        }

        if (tokens_per_tile > rows_per_face) {
#pragma GCC unroll 16
            for (uint32_t t = rows_per_face; t < tokens_per_tile; t++) {
                uint16_t winning_group_idx = sorted_indices_ptr[k_indices_offset_16_31 + (t - rows_per_face)];
                uint64_t score_src_tile_offset = winning_group_idx * score_tile::tile_size_bytes;
                uint64_t index_src_tile_offset = winning_group_idx * index_tile::tile_size_bytes;

                uint32_t t_off_s = (t - rows_per_face) * score_tile::face_line_bytes;
                uint32_t score_row_fl1 = 2 * score_tile::face_size_bytes + t_off_s;
                uint32_t score_row_fl2 = 3 * score_tile::face_size_bytes + t_off_s;
                uint32_t t_off_i = (t - rows_per_face) * index_tile::face_line_bytes;
                uint32_t index_row_fl1 = 2 * index_tile::face_size_bytes + t_off_i;
                uint32_t index_row_fl2 = 3 * index_tile::face_size_bytes + t_off_i;

                noc_async_read(
                    scores_base_noc_addr + score_src_tile_offset + score_row_fl1,
                    scores_dest_addr + score_row_fl1,
                    score_tile::face_line_bytes);
                noc_async_read(
                    indices_base_noc_addr + index_src_tile_offset + index_row_fl1,
                    indices_dest_addr + index_row_fl1,
                    index_tile::face_line_bytes);
                noc_async_read(
                    scores_base_noc_addr + score_src_tile_offset + score_row_fl2,
                    scores_dest_addr + score_row_fl2,
                    score_tile::face_line_bytes);
                noc_async_read(
                    indices_base_noc_addr + index_src_tile_offset + index_row_fl2,
                    indices_dest_addr + index_row_fl2,
                    index_tile::face_line_bytes);
            }
        }
    }

    noc_async_read_barrier();

    cb_pop_front(cb_biased_scores, width_tiles);
    cb_pop_front(cb_sorted_group_order, num_group_tiles);

    cb_push_back(cb_winning_group_scores, topk_groups);
    cb_push_back(cb_winning_group_indices, topk_groups);
}

void write_single_scalar(const uint32_t cb_scalar, const uint32_t packed_scalar) {
    cb_reserve_back(cb_scalar, 1);
    uint32_t write_addr = get_write_ptr(cb_scalar);
    tt_l1_ptr uint32_t* write_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(write_addr);
    write_ptr[0] = packed_scalar;
    cb_push_back(cb_scalar, 1);
}

template <
    uint32_t cb_out_indices,
    uint32_t cb_sigmoid_scores,
    uint32_t cb_gathered_sigmoid,
    uint32_t width_tiles,
    uint32_t n_activated_experts,
    uint32_t n_activated_expert_tiles>
FORCE_INLINE void gather(uint32_t tokens_per_tile) {
    cb_wait_front(cb_sigmoid_scores, width_tiles);
    cb_wait_front(cb_out_indices, 1);

    cb_reserve_back(cb_gathered_sigmoid, n_activated_expert_tiles);

    uint32_t sigmoid_base_addr = get_read_ptr(cb_sigmoid_scores);
    uint32_t indices_addr = get_read_ptr(cb_out_indices);
    uint32_t gathered_addr = get_write_ptr(cb_gathered_sigmoid);

    volatile tt_l1_ptr uint16_t* indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(indices_addr);
    volatile tt_l1_ptr uint32_t* sigmoid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sigmoid_base_addr);
    volatile tt_l1_ptr uint32_t* gathered_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gathered_addr);

    for (uint32_t token = 0; token < tokens_per_tile; token++) {
        uint32_t token_face_row = token % rows_per_face;
        uint32_t token_face_base = (token < rows_per_face) ? 0 : 2;

        for (uint32_t expert = 0; expert < n_activated_experts; expert++) {
            uint32_t idx_col = expert;
            uint32_t idx_face_col = idx_col % columns_per_face;
            uint32_t idx_face = token_face_base + (idx_col < columns_per_face ? 0 : 1);
            uint32_t idx_offset =
                idx_face * (index_tile::face_size_bytes / 2) + token_face_row * columns_per_face + idx_face_col;

            uint16_t expert_idx = indices_ptr[idx_offset];

            uint32_t sigmoid_tile = expert_idx / columns_per_tile;
            uint32_t sigmoid_col = expert_idx % columns_per_tile;
            uint32_t sigmoid_face_col = sigmoid_col % columns_per_face;
            uint32_t sigmoid_face = token_face_base + (sigmoid_col < columns_per_face ? 0 : 1);
            uint32_t sigmoid_offset = sigmoid_tile * (score_tile::tile_size_bytes / 4) +
                                      sigmoid_face * (score_tile::face_size_bytes / 4) +
                                      token_face_row * columns_per_face + sigmoid_face_col;

            uint32_t sigmoid_val = sigmoid_ptr[sigmoid_offset];

            uint32_t gathered_face_col = idx_col % columns_per_face;
            uint32_t gathered_face = token_face_base + (idx_col < columns_per_face ? 0 : 1);
            uint32_t gathered_offset = gathered_face * (score_tile::face_size_bytes / 4) +
                                       token_face_row * columns_per_face + gathered_face_col;

            gathered_ptr[gathered_offset] = sigmoid_val;
        }
    }

    cb_pop_front(cb_sigmoid_scores, width_tiles);

    cb_push_back(cb_gathered_sigmoid, n_activated_expert_tiles);
}

void kernel_main() {
    constexpr uint32_t cb_out_weights = get_named_compile_time_arg_val("cb_out_weights");
    constexpr uint32_t cb_out_indices = get_named_compile_time_arg_val("cb_out_indices");
    constexpr uint32_t scores_page_size = get_named_compile_time_arg_val("scores_page_size");
    constexpr uint32_t weights_page_size = get_named_compile_time_arg_val("weights_page_size");
    constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t cb_expert_index_template = get_named_compile_time_arg_val("cb_expert_index_template");
    constexpr uint32_t cb_in_scores = get_named_compile_time_arg_val("cb_in_scores");
    constexpr uint32_t cb_group_index_template = get_named_compile_time_arg_val("cb_group_index_template");
    constexpr uint32_t cb_top_experts_per_group = get_named_compile_time_arg_val("cb_top_experts_per_group");
    constexpr uint32_t cb_sorted_group_scores = get_named_compile_time_arg_val("cb_sorted_group_scores");
    constexpr uint32_t cb_sorted_group_order = get_named_compile_time_arg_val("cb_sorted_group_order");
    constexpr uint32_t cb_winning_group_scores = get_named_compile_time_arg_val("cb_winning_group_scores");
    constexpr uint32_t cb_winning_group_indices = get_named_compile_time_arg_val("cb_winning_group_indices");
    constexpr uint32_t cb_sigmoid_scores = get_named_compile_time_arg_val("cb_sigmoid_scores");
    constexpr uint32_t cb_biased_scores = get_named_compile_time_arg_val("cb_biased_scores");
    constexpr uint32_t n_activated_expert_tiles = get_named_compile_time_arg_val("n_activated_expert_tiles");

    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("tile_width");
    constexpr uint32_t tile_height = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t topk_groups = get_named_compile_time_arg_val("topk_groups");
    constexpr uint32_t n_groups = get_named_compile_time_arg_val("n_groups");
    constexpr uint32_t summed_experts_per_group = get_named_compile_time_arg_val("summed_experts_per_group");
    constexpr uint32_t num_group_tiles = get_named_compile_time_arg_val("num_group_tiles");
    constexpr uint32_t cb_reduce_ones_scalar = get_named_compile_time_arg_val("cb_reduce_ones_scalar");
    constexpr uint32_t packed_one_scalar = get_named_compile_time_arg_val("packed_one_scalar");
    constexpr uint32_t packed_route_scale = get_named_compile_time_arg_val("packed_route_scale");
    constexpr uint32_t n_activated_experts = get_named_compile_time_arg_val("n_activated_experts");

    constexpr uint32_t packed_epsilon = get_named_compile_time_arg_val("packed_epsilon");
    constexpr uint32_t cb_epsilon_scalar = get_named_compile_time_arg_val("cb_epsilon_scalar");
    constexpr uint32_t cb_route_scale_scalar = get_named_compile_time_arg_val("cb_route_scale_scalar");
    constexpr uint32_t seq_len_tiles = get_named_compile_time_arg_val("seq_len_tiles");
    constexpr uint32_t remainder_tokens_per_tile = get_named_compile_time_arg_val("remainder_tokens_per_tile");
    constexpr uint32_t cb_gathered_sigmoid = get_named_compile_time_arg_val("cb_gathered_sigmoid");

    const uint32_t weights_addr = get_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_height_tile = get_arg_val<uint32_t>(2);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(3);

    constexpr auto weights_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();

    const auto weights_accessor = TensorAccessor(weights_args, weights_addr, weights_page_size);
    const auto indices_accessor = TensorAccessor(indices_args, indices_addr, indices_page_size);

    // while reader and compute kernels are applying the sigmoid, we can create the topk indices
    // I see no performance difference generating these internally inside the writer kernel
    generate_index_tiles(cb_expert_index_template, width_tiles, indices_page_size);
    generate_group_indices_tiles(cb_group_index_template, width_tiles, n_groups);
    generate_reduce_scalar(cb_reduce_ones_scalar, packed_one_scalar, n_activated_experts);
    write_single_scalar(cb_epsilon_scalar, packed_epsilon);
    write_single_scalar(cb_route_scale_scalar, packed_route_scale);

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        // Use remainder_tokens_per_tile only for the LAST tile of the sequence, otherwise use full tile_height
        uint32_t tokens_per_tile = ((height_tile + 1) % seq_len_tiles == 0) ? remainder_tokens_per_tile : tile_height;
        generate_summed_experts_tiles(
            cb_top_experts_per_group, cb_sorted_group_scores, width_tiles, summed_experts_per_group, tokens_per_tile);
        generate_winning_group_tiles<
            cb_sorted_group_order,
            cb_biased_scores,  // Use biased scores for selection/routing; unbiased (sigmoid-only) scores are gathered
                               // later for final weight computation
            cb_expert_index_template,
            cb_winning_group_scores,
            cb_winning_group_indices,
            width_tiles,
            topk_groups,
            num_group_tiles>(tokens_per_tile);

        cb_wait_front(cb_out_indices, 1);

        // Gather unbiased sigmoid scores using the final expert indices
        gather<
            cb_out_indices,
            cb_sigmoid_scores,
            cb_gathered_sigmoid,
            width_tiles,
            n_activated_experts,
            n_activated_expert_tiles>(tokens_per_tile);

        noc_async_write_page(height_tile, indices_accessor, get_read_ptr(cb_out_indices));
        cb_wait_front(cb_out_weights, 1);
        noc_async_write_page(height_tile, weights_accessor, get_read_ptr(cb_out_weights));
        noc_async_writes_flushed();
        cb_pop_front(cb_out_indices, 1);
        cb_pop_front(cb_out_weights, 1);
    }
    noc_async_write_barrier();
}
