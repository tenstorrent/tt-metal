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

// Feeds one core's slice of the [T, D] output-tile grid:
//   * comb  -> a zero-padded, *transposed* single tile (the mix reduces over comb's first
//              hc axis, and zeroing everything outside the hc x hc block makes the K
//              reduction ignore the streams tile's padding rows).
//   * post  -> a zero-padded single tile holding only column 0, so that
//              matmul(post, sublayer_out) degenerates to the placement outer product
//              regardless of what sublayer_out carries in its padding rows.
//   * streams / sublayer_out -> one tile each per output tile.
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

    constexpr auto post_args = TensorAccessorArgs<9>();
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
    const uint32_t tile_size_bytes = streams_cb.get_tile_size();

    // The two scratch buffers are never pushed: they only stage the raw comb / post
    // pages so the loops below can re-lay them out in place.
    comb_src_cb.reserve_back(one_tile);
    post_src_cb.reserve_back(one_tile);
    const volatile tt_l1_ptr uint16_t* comb_src =
        reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(comb_src_cb.get_write_ptr());
    const volatile tt_l1_ptr uint16_t* post_src =
        reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(post_src_cb.get_write_ptr());

    // Zero every page of the comb / post buffers once. The per-token rebuilds below only ever
    // touch the hc x hc (resp. hc x 1) valid block, so the padding stays zero for the whole
    // kernel and the two matmuls ignore whatever the streams / sublayer_out padding holds.
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

        // comb / post are one tile per token, shared by every output tile of this token.
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
            noc.async_read(sub, sub_cb, tile_size_bytes, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            streams_cb.push_back(one_tile);
            sub_cb.push_back(one_tile);
        }

        tile = group_end;
    }
}
