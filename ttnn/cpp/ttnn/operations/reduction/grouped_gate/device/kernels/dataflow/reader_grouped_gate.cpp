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

void kernel_main() {
    constexpr uint32_t scores_cb_index = get_named_compile_time_arg_val("scores_cb_index");
    constexpr uint32_t bias_cb_index = get_named_compile_time_arg_val("bias_cb_index");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t scores_page_size = get_named_compile_time_arg_val("scores_page_size");
    constexpr uint32_t bias_page_size = get_named_compile_time_arg_val("bias_page_size");

    // Get scores and bias tensor accessors
    constexpr auto scores_args = TensorAccessorArgs<0>();
    constexpr auto bias_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();

    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_height_tile = get_arg_val<uint32_t>(2);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(3);

    const auto scores_accessor = TensorAccessor(scores_args, scores_addr, scores_page_size);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr, bias_page_size);

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        uint32_t base_page = height_tile * width_tiles;
        DPRINT << "Height tile: " << height_tile << ENDL();
        for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
            uint32_t page = base_page + width_tile;
            DPRINT << "Page: " << page << ENDL();
            cb_reserve_back(scores_cb_index, 1);
            cb_reserve_back(bias_cb_index, 1);
            if (width_tile == 0) {
                DPRINT << "Scores cb wr ptr: " << get_write_ptr(scores_cb_index) << ENDL();
            }
            noc_async_read_page(page, scores_accessor, get_write_ptr(scores_cb_index));
            noc_async_read_page(page, bias_accessor, get_write_ptr(bias_cb_index));
            noc_async_read_barrier();
            print_tile(scores_cb_index, 0, true, 0, 1, 0, 8);
            print_tile(bias_cb_index, 0, true, 0, 1, 0, 8);
            cb_push_back(scores_cb_index, 1);
            cb_push_back(bias_cb_index, 1);
        }
    }
}
