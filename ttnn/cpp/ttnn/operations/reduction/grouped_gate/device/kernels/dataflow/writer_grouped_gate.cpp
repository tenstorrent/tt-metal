// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void print_tile(
    uint32_t cb_idx,
    uint32_t tile_idx,
    bool untilize = true,
    uint16_t start_row = 0,
    uint16_t end_row = 32,
    uint8_t start_col = 0,
    uint8_t end_col = 32) {
    DPRINT << "cb_idx: " << cb_idx << " tile_idx: " << tile_idx << ENDL();
    DPRINT << "======" << ENDL();
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
    }
    DPRINT << "++++++" << ENDL();
}

FORCE_INLINE void generate_index_tile(
    const uint32_t topk_index_creation_cb_index, const uint32_t index_write_addr, uint32_t start_expert_index) {
    constexpr uint32_t face_line = 16;
    constexpr uint32_t face_line_bytes = 32;
    constexpr uint32_t face_size_bytes = 512;

    // Create the top two faces by writing 1 face line, then using noc to write the rest of the face
    cb_reserve_back(topk_index_creation_cb_index, 1);
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
    cb_push_back(topk_index_creation_cb_index, 1);
}

FORCE_INLINE void generate_index_tiles(
    const uint32_t topk_index_creation_cb_index, uint32_t width_tiles, uint32_t page_size) {
    uint32_t write_addr = get_write_ptr(topk_index_creation_cb_index);
    constexpr uint32_t face_size = 16;
    for (uint32_t i = 0; i < width_tiles; i++) {
        generate_index_tile(topk_index_creation_cb_index, get_write_ptr(topk_index_creation_cb_index), 32 * i);
    }
}

// Vertically along each tile, write index 0, ..., n_groups - 1
FORCE_INLINE void generate_group_indices_tiles(
    const uint32_t group_indices_cb_index, uint32_t width_tiles, uint32_t n_groups) {
    cb_reserve_back(group_indices_cb_index, 1);  // max of 32 groups
    uint32_t base_write_addr = get_write_ptr(group_indices_cb_index);
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
    cb_push_back(group_indices_cb_index, 1);
}

FORCE_INLINE void generate_summed_experts_tiles(
    const uint32_t summed_experts_cb_index,
    const uint32_t topk_input_cb_index,
    uint32_t width_tiles,
    uint32_t summed_experts_per_group) {
    // copy 0,...,summed_experts_per_group-1 rows from topk_input_cb_index to 0,...,summed_experts_per_group-1 tile in
    // summed_experts_cb_index for each width_tile
    constexpr uint32_t tokens_per_tile = 32;  // only 1 for decode but let's just do this for now
    constexpr uint32_t face_line_bytes = 32;
    constexpr uint32_t tile_size_bytes = 2048;
    constexpr uint32_t face_size_bytes = 512;

    // for each group, copy the top experts_per_group rows to the summed_experts_cb_index
    // summed_experts_per_group has experts_per_group tiles, each tile is 32x32 bf16 values, divided into 16x16 faces
    // in our case, for now, width_tiles = n_groups

    // for each group, copy the top experts_per_group rows to the summed_experts_cb_index
    cb_reserve_back(summed_experts_cb_index, summed_experts_per_group);
    for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
        // get one width tile
        cb_wait_front(topk_input_cb_index, 1);
        // offset to relevant tile in topk_input_cb_index
        uint64_t group_sorted_tile_ptr = get_noc_addr(get_read_ptr(topk_input_cb_index));
        if (width_tile % 2 == 0) {
            // even width tiles are in the first face lines
            for (uint32_t i = 0; i < summed_experts_per_group; i++) {
                // read first face line to summed_experts_cb_index
                noc_async_read(
                    group_sorted_tile_ptr + i * face_line_bytes,
                    get_write_ptr(summed_experts_cb_index) + i * tile_size_bytes + width_tile * face_line_bytes,
                    face_line_bytes);
                if constexpr (tokens_per_tile > 16) {
                    noc_async_read(
                        group_sorted_tile_ptr + face_size_bytes + i * face_line_bytes,
                        get_write_ptr(summed_experts_cb_index) + face_size_bytes + i * tile_size_bytes,
                        face_line_bytes);
                }
            }
        } else {
            // odd width tiles are in the last face lines
            // offset to face 3
            group_sorted_tile_ptr += face_size_bytes * 2;
            for (uint32_t i = 0; i < summed_experts_per_group; i++) {
                noc_async_read(
                    group_sorted_tile_ptr + (15 - i) * face_line_bytes,
                    get_write_ptr(summed_experts_cb_index) + i * tile_size_bytes + width_tile * face_line_bytes,
                    face_line_bytes);
                if constexpr (tokens_per_tile > 16) {
                    noc_async_read(
                        group_sorted_tile_ptr + face_size_bytes + (15 - i) * face_line_bytes,
                        get_write_ptr(summed_experts_cb_index) + face_size_bytes + i * tile_size_bytes,
                        face_line_bytes);
                }
            }
        }
        noc_async_read_barrier();
        // if (width_tile % 2 == 1) {
        //     DPRINT << ENDL() << "++++++" << ENDL();
        //     uint32_t start_row = width_tile % 2 == 0 ? 0 : 32 - summed_experts_per_group;
        //     uint32_t end_row = width_tile % 2 == 0 ? summed_experts_per_group : 32;
        //     print_tile(topk_input_cb_index, 0, true, start_row, end_row);
        //     for (uint32_t i = 0; i < summed_experts_per_group; i++) {
        //         print_tile(summed_experts_cb_index, i, true, width_tile, width_tile+1);
        //     }
        //     DPRINT << "++++++" << ENDL();
        // }
        cb_pop_front(topk_input_cb_index, 1);
    }
    cb_push_back(summed_experts_cb_index, summed_experts_per_group);
}

FORCE_INLINE void generate_winning_group_tiles(
    const uint32_t sorted_group_indices_cb_index,
    const uint32_t scores_cb_index,
    const uint32_t topk_index_creation_cb_index,
    const uint32_t winning_group_scores_cb_index,
    const uint32_t winning_group_indices_cb_index,
    uint32_t width_tiles,
    uint32_t topk_groups,
    uint32_t num_group_tiles,
    uint32_t tokens_per_tile) {
    constexpr uint32_t tile_size_bytes = 2048;
    constexpr uint32_t face_size_bytes = 512;
    constexpr uint32_t face_line_bytes = 32;  // 16 elements * 2 bytes

    cb_wait_front(scores_cb_index, width_tiles);
    cb_wait_front(topk_index_creation_cb_index, width_tiles);
    cb_wait_front(sorted_group_indices_cb_index, num_group_tiles);
    // DPRINT << "Sorted group indices cb 0" << ENDL();
    // print_tile(sorted_group_indices_cb_index, 0, true, 0, topk_groups, 0, 1);

    // uint32_t tile_idx = 5;
    // DPRINT << "Topk index creation cb " << tile_idx << ENDL();
    // print_tile(topk_index_creation_cb_index, tile_idx, true, 0, 1);
    // print_tile(scores_cb_index, tile_idx, true, 0, 1);

    cb_reserve_back(winning_group_scores_cb_index, topk_groups);
    cb_reserve_back(winning_group_indices_cb_index, topk_groups);

    // Pointers
    uint64_t scores_base_noc_addr = get_noc_addr(get_read_ptr(scores_cb_index));
    uint64_t indices_base_noc_addr = get_noc_addr(get_read_ptr(topk_index_creation_cb_index));
    uint32_t scores_dest_base_addr = get_write_ptr(winning_group_scores_cb_index);
    uint32_t indices_dest_base_addr = get_write_ptr(winning_group_indices_cb_index);

    // Indices pointer (in L1)
    volatile tt_l1_ptr uint16_t* sorted_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(sorted_group_indices_cb_index));

    // Iterate over rows (tokens)
    for (uint32_t t = 0; t < tokens_per_tile; t++) {
        // Calculate indices for the token `t` in sorted_group_indices
        // Transposed layout: topk_groups along rows, tokens along columns (width)

        // sorted_group_indices is a single tile (32x32)
        // We want the column `t` for this token.
        // The column `t` contains `topk_groups` rows, which are the winning group indices.

        // For a tile, column `t` spans across faces vertically.
        // Face 0: cols 0-15, rows 0-15
        // Face 1: cols 16-31, rows 0-15
        // Face 2: cols 0-15, rows 16-31
        // Face 3: cols 16-31, rows 16-31

        // To get the `k`-th winning group for token `t`:
        // We need to access (row=k, col=t) in the sorted_group_indices tile.

        // Calculate row offset part for the output/scores/indices tiles (which are standard row-major per token)
        // These follow standard layout: row `t` corresponds to token `t`.
        uint32_t row_offset_part1;  // Face 0 or 2
        uint32_t row_offset_part2;  // Face 1 or 3

        if (t < 16) {
            row_offset_part1 = t * face_line_bytes;
            row_offset_part2 = face_size_bytes + t * face_line_bytes;
        } else {
            row_offset_part1 = 2 * face_size_bytes + (t - 16) * face_line_bytes;
            row_offset_part2 = 3 * face_size_bytes + (t - 16) * face_line_bytes;
        }

        // Iterate over ranks (k) to copy
        for (uint32_t k = 0; k < topk_groups; k++) {
            // Get the winning group index from (row=k, col=t) in sorted_group_indices
            // We need to map (k, t) to the linear offset in the tile buffer.

            uint16_t winning_group_idx;
            uint32_t index_offset_elements = 0;

            // Determine face of (k, t)
            if (k < 16 && t < 16) {
                // Face 0
                index_offset_elements = k * 16 + t;
            } else if (k < 16 && t >= 16) {
                // Face 1
                index_offset_elements = 256 + k * 16 + (t - 16);
            } else if (k >= 16 && t < 16) {
                // Face 2
                index_offset_elements = 512 + (k - 16) * 16 + t;
            } else {
                // Face 3
                index_offset_elements = 768 + (k - 16) * 16 + (t - 16);
            }

            winning_group_idx = sorted_indices_ptr[index_offset_elements];

            // Source Tile Address: Base + winning_group_idx * tile_size
            uint64_t src_tile_offset = winning_group_idx * tile_size_bytes;

            // Dest Tile Address: Base + k * tile_size
            uint32_t dest_tile_offset = k * tile_size_bytes;

            // Copy Part 1 (Face 0 or 2)
            noc_async_read(
                scores_base_noc_addr + src_tile_offset + row_offset_part1,
                scores_dest_base_addr + dest_tile_offset + row_offset_part1,
                face_line_bytes);
            noc_async_read(
                indices_base_noc_addr + src_tile_offset + row_offset_part1,
                indices_dest_base_addr + dest_tile_offset + row_offset_part1,
                face_line_bytes);

            // Copy Part 2 (Face 1 or 3)
            noc_async_read(
                scores_base_noc_addr + src_tile_offset + row_offset_part2,
                scores_dest_base_addr + dest_tile_offset + row_offset_part2,
                face_line_bytes);
            noc_async_read(
                indices_base_noc_addr + src_tile_offset + row_offset_part2,
                indices_dest_base_addr + dest_tile_offset + row_offset_part2,
                face_line_bytes);
        }
    }

    noc_async_read_barrier();
    // DPRINT << ENDL() << ENDL();
    // for (uint32_t i = 0; i < topk_groups; i++) {
    //     DPRINT << "Winning group scores cb " << i << ENDL();
    //     print_tile(winning_group_scores_cb_index, i, true, 0, 1);
    // }
    cb_push_back(winning_group_scores_cb_index, topk_groups);
    cb_push_back(winning_group_indices_cb_index, topk_groups);

    // Pop inputs
    cb_pop_front(scores_cb_index, width_tiles);
    cb_pop_front(topk_index_creation_cb_index, width_tiles);
    cb_pop_front(sorted_group_indices_cb_index, num_group_tiles);
}

void kernel_main() {
    constexpr uint32_t weights_cb_index = get_named_compile_time_arg_val("weights_cb_index");
    constexpr uint32_t indices_cb_index = get_named_compile_time_arg_val("indices_cb_index");
    constexpr uint32_t weights_page_size = get_named_compile_time_arg_val("weights_page_size");
    constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t topk_index_creation_cb_index = get_named_compile_time_arg_val("topk_index_creation_cb_index");
    constexpr uint32_t scores_cb_index = get_named_compile_time_arg_val("scores_cb_index");
    constexpr uint32_t group_indices_cb_index = get_named_compile_time_arg_val("group_indices_cb_index");
    constexpr uint32_t summed_experts_cb_index = get_named_compile_time_arg_val("summed_experts_cb_index");
    constexpr uint32_t topk_input_cb_index = get_named_compile_time_arg_val("topk_input_cb_index");
    constexpr uint32_t sorted_group_indices_cb_index = get_named_compile_time_arg_val("sorted_group_indices_cb_index");
    constexpr uint32_t winning_group_scores_cb_index = get_named_compile_time_arg_val("winning_group_scores_cb_index");
    constexpr uint32_t winning_group_indices_cb_index =
        get_named_compile_time_arg_val("winning_group_indices_cb_index");
    constexpr uint32_t sigmoid_input_cb_index = get_named_compile_time_arg_val("sigmoid_input_cb_index");

    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("tile_width");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t topk_groups = get_named_compile_time_arg_val("topk_groups");
    constexpr uint32_t n_groups = get_named_compile_time_arg_val("n_groups");
    constexpr uint32_t summed_experts_per_group = get_named_compile_time_arg_val("summed_experts_per_group");
    constexpr uint32_t num_group_tiles = get_named_compile_time_arg_val("num_group_tiles");

    const uint32_t weights_addr = get_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_height_tile = get_arg_val<uint32_t>(2);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(3);

    constexpr auto weights_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();

    const auto weights_accessor = TensorAccessor(weights_args, weights_addr, weights_page_size);
    const auto indices_accessor = TensorAccessor(indices_args, indices_addr, indices_page_size);

    constexpr uint32_t tokens_per_tile = 32;  // hardcoded for now, but this is std::min(tokens, tile_width)

    // while reader and compute kernels are applying the sigmoid, we can create the topk indices
    // I see no performance difference generating these internally inside the writer kernel
    generate_index_tiles(topk_index_creation_cb_index, width_tiles, indices_page_size);
    generate_group_indices_tiles(group_indices_cb_index, width_tiles, n_groups);

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        generate_summed_experts_tiles(
            summed_experts_cb_index, topk_input_cb_index, width_tiles, summed_experts_per_group);
        generate_winning_group_tiles(
            sorted_group_indices_cb_index,
            sigmoid_input_cb_index,
            topk_index_creation_cb_index,
            winning_group_scores_cb_index,
            winning_group_indices_cb_index,
            width_tiles,
            topk_groups,
            num_group_tiles,
            tokens_per_tile);

        cb_wait_front(indices_cb_index, 1);
        DPRINT << "Page size: " << indices_page_size << ENDL();
        noc_async_write_page(height_tile, indices_accessor, get_write_ptr(indices_cb_index));
        noc_async_writes_flushed();
        cb_pop_front(indices_cb_index, 1);
    }
    noc_async_write_barrier();
}
