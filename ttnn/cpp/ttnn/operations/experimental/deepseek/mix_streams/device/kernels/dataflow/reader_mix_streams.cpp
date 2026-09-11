// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {

// Tile face layout: 4 faces of 16x16. face = (r>=16?2:0) + (c>=16?1:0); element (r,c) lives
// at element index (face*256 + (r%16)*16 + (c%16)) within the tile.
FORCE_INLINE uint32_t tile_face_index(uint32_t r, uint32_t c) {
    const uint32_t face = ((r >= 16) ? 2u : 0u) + ((c >= 16) ? 1u : 0u);
    return face * 256u + (r & 15u) * 16u + (c & 15u);
}

}  // namespace

// TILE post / comb / streams (32x32 pages). ROW_MAJOR sublayer_out is read as 1x32
// faces (32 bf16 cols) and scattered into row 0 of a 32x32 tile so the placement
// matmul still sees a single-tile B.
void kernel_main() {
    const uint32_t post_addr = get_arg_val<uint32_t>(0);
    const uint32_t comb_addr = get_arg_val<uint32_t>(1);
    const uint32_t sub_addr = get_arg_val<uint32_t>(2);
    const uint32_t streams_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_tile = get_arg_val<uint32_t>(4);
    const uint32_t num_tiles = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_comb_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(1);
    constexpr uint32_t cb_post_src = get_compile_time_arg_val(2);
    constexpr uint32_t cb_post = get_compile_time_arg_val(3);
    constexpr uint32_t cb_streams = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sub = get_compile_time_arg_val(5);
    constexpr uint32_t hc = get_compile_time_arg_val(6);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t comb_post_cb_pages = get_compile_time_arg_val(8);
    constexpr uint32_t sub_elems_per_page = get_compile_time_arg_val(9);
    constexpr uint32_t sub_is_rm = get_compile_time_arg_val(10);

    constexpr auto post_args = TensorAccessorArgs<11>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr auto sub_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();
    constexpr auto streams_args = TensorAccessorArgs<sub_args.next_compile_time_args_offset()>();

    const auto post = TensorAccessor(post_args, post_addr);
    const auto comb = TensorAccessor(comb_args, comb_addr);
    const auto sub = TensorAccessor(sub_args, sub_addr);
    const auto streams = TensorAccessor(streams_args, streams_addr);

    Noc noc;
    CircularBuffer comb_src_cb(cb_comb_src);
    CircularBuffer comb_cb(cb_comb);
    CircularBuffer post_src_cb(cb_post_src);
    CircularBuffer post_cb(cb_post);
    CircularBuffer streams_cb(cb_streams);
    CircularBuffer sub_cb(cb_sub);

    constexpr uint32_t one_tile = 1;
    constexpr uint32_t tile_w = 32;
    constexpr uint32_t face_bytes = tile_w * sizeof(uint16_t);
    const uint32_t tile_size_bytes = streams_cb.get_tile_size();
    constexpr uint32_t pages_per_token = (n_tiles * tile_w) / sub_elems_per_page;

    comb_src_cb.reserve_back(one_tile);
    post_src_cb.reserve_back(one_tile);
    const volatile tt_l1_ptr uint16_t* comb_src =
        reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(comb_src_cb.get_write_ptr());
    const volatile tt_l1_ptr uint16_t* post_src =
        reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(post_src_cb.get_write_ptr());

    comb_cb.reserve_back(comb_post_cb_pages);
    post_cb.reserve_back(comb_post_cb_pages);
    noc.async_write_zeros(comb_cb, comb_post_cb_pages * tile_size_bytes, {.offset_bytes = 0});
    noc.async_write_zeros(post_cb, comb_post_cb_pages * tile_size_bytes, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();

    const uint32_t end_tile = start_tile + num_tiles;
    uint32_t tile = start_tile;
    while (tile < end_tile) {
        const uint32_t t = tile / n_tiles;
        const uint32_t group_end = (end_tile < (t + 1) * n_tiles) ? end_tile : (t + 1) * n_tiles;

        noc.async_read(comb, comb_src_cb, tile_size_bytes, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read(post, post_src_cb, tile_size_bytes, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read_barrier();

        comb_cb.reserve_back(one_tile);
        volatile tt_l1_ptr uint16_t* comb_dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(comb_cb.get_write_ptr());
        for (uint32_t r = 0; r < hc; ++r) {
            for (uint32_t c = 0; c < hc; ++c) {
                comb_dst[tile_face_index(c, r)] = comb_src[tile_face_index(r, c)];
            }
        }
        comb_cb.push_back(one_tile);

        post_cb.reserve_back(one_tile);
        volatile tt_l1_ptr uint16_t* post_dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(post_cb.get_write_ptr());
        for (uint32_t r = 0; r < hc; ++r) {
            post_dst[tile_face_index(r, 0)] = post_src[tile_face_index(r, 0)];
        }
        post_cb.push_back(one_tile);

        for (uint32_t page = tile; page < group_end; ++page) {
            streams_cb.reserve_back(one_tile);
            sub_cb.reserve_back(one_tile);
            noc.async_read(streams, streams_cb, tile_size_bytes, {.page_id = page}, {.offset_bytes = 0});
            if constexpr (sub_is_rm) {
                const uint32_t n_idx = page % n_tiles;
                const uint32_t col = n_idx * tile_w;
                const uint32_t sub_page_id = t * pages_per_token + col / sub_elems_per_page;
                const uint32_t sub_offset = (col % sub_elems_per_page) * sizeof(uint16_t);
                noc.async_read(
                    sub,
                    comb_src_cb,
                    face_bytes,
                    {.page_id = sub_page_id, .offset_bytes = sub_offset},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                volatile tt_l1_ptr uint16_t* sub_dst =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sub_cb.get_write_ptr());
                const volatile tt_l1_ptr uint16_t* sub_row =
                    reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(comb_src_cb.get_write_ptr());
                for (uint32_t c = 0; c < tile_w; ++c) {
                    sub_dst[tile_face_index(0, c)] = sub_row[c];
                }
            } else {
                noc.async_read(sub, sub_cb, tile_size_bytes, {.page_id = page}, {.offset_bytes = 0});
                noc.async_read_barrier();
            }
            streams_cb.push_back(one_tile);
            sub_cb.push_back(one_tile);
        }

        tile = group_end;
    }
}
