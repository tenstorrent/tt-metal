// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

FORCE_INLINE void generate_index_tile(
    const uint32_t cb_expert_index_template, const uint32_t index_write_addr, uint32_t start_expert_index) {
    constexpr uint32_t face_line = 16;
    constexpr uint32_t face_line_bytes = 32;
    constexpr uint32_t face_size_bytes = 512;

    // Create the top two faces by writing 1 face line, then using noc to write the rest of the face
    cb_reserve_back(cb_expert_index_template, 1);
    for (uint32_t width_face = 0; width_face < 2; width_face++) {
        uint32_t current_index = start_expert_index + width_face * face_line;
        uint32_t index_write_face_offset = index_write_addr + width_face * face_size_bytes;

        uint64_t base_index_noc_addr = get_noc_addr(index_write_face_offset);

        // write first 16 uint16_t values as 8 uint32_t writes
        volatile tt_l1_ptr uint32_t* index_cb_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_write_face_offset);
        for (uint32_t i = 0; i < face_line / 2; i++) {
            index_cb_ptr[i] = (current_index + 1) << 16 | current_index;
            current_index += 2;
        }
        // then use noc to write the rest of the face
        uint32_t dm_engine_index_write_offset = index_write_face_offset + face_line_bytes;
        for (uint32_t i = 1; i < face_line; i++) {
            noc_async_read(base_index_noc_addr, dm_engine_index_write_offset, face_line_bytes);
            dm_engine_index_write_offset += face_line_bytes;
        }
    }

    uint64_t index_noc_addr_base = get_noc_addr(index_write_addr);
    noc_async_read_barrier();

    // Create the bottom two faces by doing a noc copy of the top two faces
    noc_async_read(index_noc_addr_base, index_write_addr + 2 * face_size_bytes, 2 * face_size_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_expert_index_template, 1);
}

FORCE_INLINE void generate_index_tiles(
    const uint32_t cb_expert_index_template, uint32_t width_tiles, uint32_t page_size) {
    uint32_t write_addr = get_write_ptr(cb_expert_index_template);
    constexpr uint32_t face_size = 16;
    for (uint32_t i = 0; i < width_tiles; i++) {
        generate_index_tile(cb_expert_index_template, get_write_ptr(cb_expert_index_template), 32 * i);
    }
}

// Vertically along each tile, write index 0, ..., n_groups - 1
FORCE_INLINE void generate_group_indices_tiles(
    const uint32_t cb_group_index_template, uint32_t width_tiles, uint32_t n_groups) {
    cb_reserve_back(cb_group_index_template, 1);  // max of 32 groups
    uint32_t base_write_addr = get_write_ptr(cb_group_index_template);
    constexpr uint32_t face_line = 16;
    constexpr uint32_t face_line_bytes = 32;
    constexpr uint32_t num_tile_elements = 1024;
    constexpr uint32_t num_tile_elements_div_2 = num_tile_elements / 2;
    constexpr uint32_t face_size_bytes = 512;
    constexpr uint32_t tile_size_bytes = 2048;
    // G x W x T slice of the tile is written
    // G x T subset of the tile is written, where G is the n_groups and T is the tokens
    // handle first face line
    // first row –> 0, 0, 0, second row –> 1, 1, 1, n_groups - 1 row –> n_groups - 1, n_groups - 1, n_groups
    // - 1
    volatile tt_l1_ptr uint32_t* write_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_write_addr);
    for (uint32_t group_index = 0; group_index < n_groups; group_index++) {
        // 16 uint16_t values is 8 uint32_t writes
        for (uint32_t i = 0; i < face_line / 2; i++) {
            write_ptr[i] = (group_index) << 16 | group_index;
        }
        if (group_index > 15) {
            // skip to face 3
            constexpr uint32_t skip_elements = face_size_bytes / sizeof(uint32_t);
            write_ptr += skip_elements;
        } else {
            // skip to next face line
            constexpr uint32_t skip_elements = face_line_bytes / sizeof(uint32_t);
            write_ptr += skip_elements;
        }
    }
    // then use noc to write the rest of the faces
    uint64_t dm_engine_index_write_offset_face_1 = get_noc_addr(base_write_addr);
    // if n_groups is greater than 16, we need to write the third face
    uint64_t dm_engine_index_write_offset_face_3 = get_noc_addr(base_write_addr + 2 * face_size_bytes);

    // copy face 1 to face 2 and face 3 to face 4
    uint32_t face_2_l1_write_addr = base_write_addr + face_size_bytes;
    uint32_t face_4_l1_write_addr = base_write_addr + 3 * face_size_bytes;
    noc_async_read(dm_engine_index_write_offset_face_1, face_2_l1_write_addr, face_size_bytes);
    noc_async_read(dm_engine_index_write_offset_face_3, face_4_l1_write_addr, face_size_bytes);
    uint32_t tile_write_addr = base_write_addr + tile_size_bytes;
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
    // 32x32 tile where first row of each face should be {1.0bf16, 1.0bf16, ..., 1.0bf16} up until n_activated_experts
    cb_reserve_back(cb_reduce_ones_scalar, 1);
    constexpr uint32_t face_size_bytes = 512;
    constexpr uint32_t face_line_bytes = 32;
    uint32_t write_addr = get_write_ptr(cb_reduce_ones_scalar);
    tt_l1_ptr uint16_t* write_ptr = reinterpret_cast<tt_l1_ptr uint16_t*>(write_addr);
    // the uint32_t contains two bf16 values, so we write one face line/2 elements through pointer access:
    uint16_t scalar = packed_scalar >> 16;
    for (uint32_t i = 0; i < n_activated_experts; i++) {
        write_ptr[i] = scalar;
        if (i > 15) {
            write_ptr[i + 241] = scalar;
        }
    }
    for (uint32_t i = n_activated_experts; i < 32; i++) {
        write_ptr[i] = 0;
        if (i == 16) {
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), write_addr + face_size_bytes, face_line_bytes);
        }
    }
    uint32_t face_3_write_addr = write_addr + 2 * face_size_bytes;
    uint32_t face_4_write_addr = write_addr + 3 * face_size_bytes;
    // write first face line in face 1 to face 3 and face 2 to face 4
    noc_async_read_barrier();
    noc_async_read(get_noc_addr(write_addr), face_3_write_addr, face_line_bytes);
    noc_async_read(get_noc_addr(write_addr + face_size_bytes), face_4_write_addr, face_line_bytes);
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
    constexpr uint32_t face_line_bytes = 32;
    constexpr uint32_t tile_size_bytes = 2048;
    constexpr uint32_t face_size_bytes = 512;

    // for each group, copy the top experts_per_group rows to cb_top_experts_per_group
    // summed_experts_per_group has experts_per_group tiles, each tile is 32x32 bf16 values, divided into 16x16 faces
    // in our case, for now, width_tiles = n_groups

    // for each group, copy the top experts_per_group rows to cb_top_experts_per_group
    cb_reserve_back(cb_top_experts_per_group, summed_experts_per_group);
    for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
        // get one width tile
        cb_wait_front(cb_sorted_group_scores, 1);
        // offset to relevant tile in cb_sorted_group_scores
        uint64_t group_sorted_tile_ptr = get_noc_addr(get_read_ptr(cb_sorted_group_scores));

        if (width_tile % 2 == 0) {
            // even width tiles are sorted descending, best at row 0
            for (uint32_t i = 0; i < summed_experts_per_group; i++) {
                // Source: Face 0 row i (tokens 0-15 after transpose)
                // Dest: Face 0 of tile i, row = width_tile
                noc_async_read(
                    group_sorted_tile_ptr + i * face_line_bytes,
                    get_write_ptr(cb_top_experts_per_group) + i * tile_size_bytes + width_tile * face_line_bytes,
                    face_line_bytes);
                if (tokens_per_tile > 16) {
                    // Source: Face 1 row i (tokens 16-31 after transpose)
                    // Dest: Face 1 of tile i, row = width_tile (layout is [groups, tokens])
                    noc_async_read(
                        group_sorted_tile_ptr + face_size_bytes + i * face_line_bytes,
                        get_write_ptr(cb_top_experts_per_group) + face_size_bytes + i * tile_size_bytes +
                            width_tile * face_line_bytes,
                        face_line_bytes);
                }
            }
        } else {
            // odd width tiles are sorted ascending, best at row 15
            // offset to Face 2 for source reads
            group_sorted_tile_ptr += face_size_bytes * 2;
            for (uint32_t i = 0; i < summed_experts_per_group; i++) {
                // Source: Face 2 row (15-i) for tokens 0-15
                // Dest: Face 0 of tile i, row = width_tile
                noc_async_read(
                    group_sorted_tile_ptr + (15 - i) * face_line_bytes,
                    get_write_ptr(cb_top_experts_per_group) + i * tile_size_bytes + width_tile * face_line_bytes,
                    face_line_bytes);
                if (tokens_per_tile > 16) {
                    // Source: Face 3 row (15-i) for tokens 16-31
                    // Dest: Face 1 of tile i, row = width_tile (layout is [groups, tokens])
                    noc_async_read(
                        group_sorted_tile_ptr + face_size_bytes + (15 - i) * face_line_bytes,
                        get_write_ptr(cb_top_experts_per_group) + face_size_bytes + i * tile_size_bytes +
                            width_tile * face_line_bytes,
                        face_line_bytes);
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
    constexpr uint32_t tile_size_bytes = 2048;
    constexpr uint32_t face_size_bytes = 512;
    constexpr uint32_t face_line_bytes = 32;

    cb_wait_front(cb_biased_scores, width_tiles);
    cb_wait_front(cb_expert_index_template, width_tiles);
    cb_wait_front(cb_sorted_group_order, num_group_tiles);

    cb_reserve_back(cb_winning_group_scores, topk_groups);
    cb_reserve_back(cb_winning_group_indices, topk_groups);

    // Pointers
    uint64_t scores_base_noc_addr = get_noc_addr(get_read_ptr(cb_biased_scores));
    uint64_t indices_base_noc_addr = get_noc_addr(get_read_ptr(cb_expert_index_template));
    uint32_t scores_dest_base_addr = get_write_ptr(cb_winning_group_scores);
    uint32_t indices_dest_base_addr = get_write_ptr(cb_winning_group_indices);

    // Indices pointer (in L1)
    volatile tt_l1_ptr uint16_t* sorted_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_sorted_group_order));

    for (uint32_t k = 0; k < topk_groups; k++) {
        uint32_t dest_tile_offset = k * tile_size_bytes;
        uint32_t scores_dest_addr = scores_dest_base_addr + dest_tile_offset;
        uint32_t indices_dest_addr = indices_dest_base_addr + dest_tile_offset;

        uint32_t k_indices_offset_0_15;
        uint32_t k_indices_offset_16_31;

        if (k < 16) {
            k_indices_offset_0_15 = k * 16;
            k_indices_offset_16_31 = 256 + k * 16;
        } else {
            k_indices_offset_0_15 = 512 + (k - 16) * 16;
            k_indices_offset_16_31 = 768 + (k - 16) * 16;
        }

        // Part 1: t < 16
        uint32_t limit_part1 = (tokens_per_tile < 16) ? tokens_per_tile : 16;

#pragma GCC unroll 16
        for (uint32_t t = 0; t < limit_part1; t++) {
            uint16_t winning_group_idx = sorted_indices_ptr[k_indices_offset_0_15 + t];
            uint64_t src_tile_offset = winning_group_idx * tile_size_bytes;

            uint32_t ro_p1 = t * face_line_bytes;
            uint32_t ro_p2 = face_size_bytes + ro_p1;

            noc_async_read(scores_base_noc_addr + src_tile_offset + ro_p1, scores_dest_addr + ro_p1, face_line_bytes);
            noc_async_read(indices_base_noc_addr + src_tile_offset + ro_p1, indices_dest_addr + ro_p1, face_line_bytes);
            noc_async_read(scores_base_noc_addr + src_tile_offset + ro_p2, scores_dest_addr + ro_p2, face_line_bytes);
            noc_async_read(indices_base_noc_addr + src_tile_offset + ro_p2, indices_dest_addr + ro_p2, face_line_bytes);
        }

        // Part 2: t >= 16
        if (tokens_per_tile > 16) {
#pragma GCC unroll 16
            for (uint32_t t = 16; t < tokens_per_tile; t++) {
                uint16_t winning_group_idx = sorted_indices_ptr[k_indices_offset_16_31 + (t - 16)];
                uint64_t src_tile_offset = winning_group_idx * tile_size_bytes;

                uint32_t t_off = (t - 16) * face_line_bytes;
                uint32_t ro_p1 = 2 * face_size_bytes + t_off;
                uint32_t ro_p2 = 3 * face_size_bytes + t_off;

                noc_async_read(
                    scores_base_noc_addr + src_tile_offset + ro_p1, scores_dest_addr + ro_p1, face_line_bytes);
                noc_async_read(
                    indices_base_noc_addr + src_tile_offset + ro_p1, indices_dest_addr + ro_p1, face_line_bytes);
                noc_async_read(
                    scores_base_noc_addr + src_tile_offset + ro_p2, scores_dest_addr + ro_p2, face_line_bytes);
                noc_async_read(
                    indices_base_noc_addr + src_tile_offset + ro_p2, indices_dest_addr + ro_p2, face_line_bytes);
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
    tt_l1_ptr uint16_t* write_ptr = reinterpret_cast<tt_l1_ptr uint16_t*>(write_addr);
    write_ptr[0] = packed_scalar >> 16;
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
    constexpr uint32_t tile_size_bytes = 2048;
    constexpr uint32_t face_size_bytes = 512;
    constexpr uint32_t face_line_bytes = 32;
    constexpr uint32_t elements_per_face_row = 16;

    cb_wait_front(cb_sigmoid_scores, width_tiles);
    cb_wait_front(cb_out_indices, 1);

    cb_reserve_back(cb_gathered_sigmoid, n_activated_expert_tiles);

    // Get base addresses
    uint32_t sigmoid_base_addr = get_read_ptr(cb_sigmoid_scores);
    uint32_t indices_addr = get_read_ptr(cb_out_indices);
    uint32_t gathered_addr = get_write_ptr(cb_gathered_sigmoid);

    volatile tt_l1_ptr uint16_t* indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(indices_addr);
    volatile tt_l1_ptr uint16_t* sigmoid_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sigmoid_base_addr);
    volatile tt_l1_ptr uint16_t* gathered_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(gathered_addr);

    for (uint32_t token = 0; token < tokens_per_tile; token++) {
        // Token's row position within tile faces
        uint32_t token_face_row = token % 16;
        // Face base: 0 for rows 0-15, 2 for rows 16-31 (faces 0/1 vs 2/3)
        uint32_t token_face_base = (token < 16) ? 0 : 2;

        for (uint32_t expert = 0; expert < n_activated_experts; expert++) {
            // Calculate index position in indices tile
            // indices tile layout: row=token, col=expert
            uint32_t idx_col = expert;
            uint32_t idx_face_col = idx_col % 16;
            uint32_t idx_face = token_face_base + (idx_col < 16 ? 0 : 1);
            // Each face is 256 uint16 elements (16x16), laid out row-major within face
            uint32_t idx_offset =
                idx_face * (face_size_bytes / 2) + token_face_row * elements_per_face_row + idx_face_col;

            // Read the expert index
            uint16_t expert_idx = indices_ptr[idx_offset];

            // Calculate sigmoid position: tile = expert_idx / 32, col = expert_idx % 32
            uint32_t sigmoid_tile = expert_idx / 32;
            uint32_t sigmoid_col = expert_idx % 32;
            uint32_t sigmoid_face_col = sigmoid_col % 16;
            uint32_t sigmoid_face = token_face_base + (sigmoid_col < 16 ? 0 : 1);
            // Offset into sigmoid buffer (multiple tiles)
            uint32_t sigmoid_offset = sigmoid_tile * (tile_size_bytes / 2) + sigmoid_face * (face_size_bytes / 2) +
                                      token_face_row * elements_per_face_row + sigmoid_face_col;

            uint16_t sigmoid_val = sigmoid_ptr[sigmoid_offset];

            // Write to gathered CB at position [token, expert]
            uint32_t gathered_face_col = idx_col % 16;
            uint32_t gathered_face = token_face_base + (idx_col < 16 ? 0 : 1);
            uint32_t gathered_offset =
                gathered_face * (face_size_bytes / 2) + token_face_row * elements_per_face_row + gathered_face_col;

            gathered_ptr[gathered_offset] = sigmoid_val;
        }
    }

    // Pop the sigmoid scores now that we're done gathering from them
    cb_pop_front(cb_sigmoid_scores, width_tiles);

    cb_push_back(cb_gathered_sigmoid, n_activated_expert_tiles);
}

void kernel_main() {
    constexpr uint32_t cb_out_weights = get_named_compile_time_arg_val("cb_out_weights");
    constexpr uint32_t cb_out_indices = get_named_compile_time_arg_val("cb_out_indices");
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
