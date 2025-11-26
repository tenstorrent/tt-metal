// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void print_tile(
    uint32_t cb_idx, uint32_t tile_idx, bool untilize = true, uint16_t start_row = 0, uint16_t end_row = 32) {
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
                          .w0 = (uint8_t)0,
                          .w1 = (uint8_t)32,
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
    cb_reserve_back(group_indices_cb_index, width_tiles);
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
    // then copy the tile to the rest of the tiles
    for (uint32_t tile_index = 1; tile_index < width_tiles; tile_index++) {
        noc_async_read(dm_engine_index_write_offset_face_1, tile_write_addr, tile_size_bytes);
        tile_write_addr += tile_size_bytes;
    }
    noc_async_read_barrier();
    for (uint32_t tile_index = 0; tile_index < width_tiles; tile_index++) {
        print_tile(group_indices_cb_index, tile_index, true, 0, n_groups + 1);
    }
    cb_push_back(group_indices_cb_index, width_tiles);
}

void kernel_main() {
    constexpr uint32_t weights_cb_index = get_named_compile_time_arg_val("weights_cb_index");
    constexpr uint32_t indices_cb_index = get_named_compile_time_arg_val("indices_cb_index");
    constexpr uint32_t weights_page_size = get_named_compile_time_arg_val("weights_page_size");
    constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t topk_index_creation_cb_index = get_named_compile_time_arg_val("topk_index_creation_cb_index");
    constexpr uint32_t group_indices_cb_index = get_named_compile_time_arg_val("group_indices_cb_index");
    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("tile_width");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t topk_groups = get_named_compile_time_arg_val("topk_groups");
    constexpr uint32_t n_groups = get_named_compile_time_arg_val("n_groups");

    const uint32_t weights_addr = get_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_arg_val<uint32_t>(1);

    constexpr auto weights_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();

    const auto weights_accessor = TensorAccessor(weights_args, weights_addr, weights_page_size);
    const auto indices_accessor = TensorAccessor(indices_args, indices_addr, indices_page_size);

    constexpr uint32_t tokens_per_tile = 32;  // hardcoded for now, but this is std::min(tokens, tile_width)

    // while reader and compute kernels are applying the sigmoid, we can create the topk indices
    generate_index_tiles(topk_index_creation_cb_index, width_tiles, indices_page_size);
    generate_group_indices_tiles(group_indices_cb_index, width_tiles, n_groups);
}
