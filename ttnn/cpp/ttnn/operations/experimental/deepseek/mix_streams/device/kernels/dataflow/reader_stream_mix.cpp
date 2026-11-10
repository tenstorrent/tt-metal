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

// A work unit is tile (r, c) of the [T, D] grid every stream shares. All the units of a row
// block r use the same per-token weights, so the reader turns them once per row block into
// diagonal tiles -- diag_k(w[r*32 + k]) -- which the compute kernel multiplies into the stream
// tiles: diag(w) @ S scales row k of S by token r*32 + k's weight.
//   mixing tile (i, j) = diag(weights[t, j, i]) (comb) or diag(weights[t, 0, j]) (pre, n_out = 1)
//   post tile i        = diag(post[t, i, 0])
// The per-token weights are hc <= 16 wide, so they all sit in face 0 of their token's tile.
void kernel_main() {
    const uint32_t streams_addr = get_arg_val<uint32_t>(0);
    const uint32_t weights_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t sub_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_unit = get_arg_val<uint32_t>(4);
    const uint32_t num_units = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_in_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w_src_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_post_src_id = get_compile_time_arg_val(3);
    constexpr uint32_t hc = get_compile_time_arg_val(4);
    constexpr uint32_t n_out = get_compile_time_arg_val(5);
    constexpr bool has_post = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t d_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t t_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(9);

    constexpr auto streams_args = TensorAccessorArgs<10>();
    constexpr auto weights_args = TensorAccessorArgs<streams_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto sub_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    const auto streams = TensorAccessor(streams_args, streams_addr);
    const auto weights = TensorAccessor(weights_args, weights_addr);
    const auto post = TensorAccessor(post_args, post_addr);
    const auto sub = TensorAccessor(sub_args, sub_addr);

    Noc noc;
    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_w(cb_w_id);
    CircularBuffer cb_w_src(cb_w_src_id);
    CircularBuffer cb_post_src(cb_post_src_id);

    constexpr uint32_t tile_h = 32;
    constexpr uint32_t face_elems = 256;
    constexpr uint32_t face_bytes = face_elems * sizeof(uint16_t);
    constexpr uint32_t n_in = hc + (has_post ? 1u : 0u);
    constexpr uint32_t n_w = n_in * n_out;
    constexpr uint32_t stream_tiles = t_tiles * d_tiles;
    const uint32_t tile_bytes = cb_in.get_tile_size();
    const uint32_t tile_elems = tile_bytes / sizeof(uint16_t);

    // The diagonal tiles are rebuilt in place every row block; only their diagonals are written,
    // so zero them once. cb_w holds exactly n_w pages, so every reserve returns this same region.
    cb_w.reserve_back(n_w);
    noc.async_write_zeros(cb_w, n_w * tile_bytes, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    volatile tt_l1_ptr uint16_t* w_dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_w.get_write_ptr());

    // Staging scratches: never pushed, so their write pointers stay put.
    cb_w_src.reserve_back(1);
    const volatile tt_l1_ptr uint16_t* w_src =
        reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(cb_w_src.get_write_ptr());
    const volatile tt_l1_ptr uint16_t* post_src = nullptr;
    if constexpr (has_post) {
        cb_post_src.reserve_back(1);
        post_src = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(cb_post_src.get_write_ptr());
    }

    const uint32_t end_unit = start_unit + num_units;
    uint32_t unit = start_unit;
    while (unit < end_unit) {
        const uint32_t r = unit / d_tiles;
        const uint32_t group_end = (end_unit < (r + 1) * d_tiles) ? end_unit : (r + 1) * d_tiles;
        const uint32_t t0 = r * tile_h;
        const uint32_t valid = (num_tokens - t0 < tile_h) ? num_tokens - t0 : tile_h;

        for (uint32_t k = 0; k < valid; ++k) {
            noc.async_read(weights, cb_w_src, face_bytes, {.page_id = t0 + k}, {.offset_bytes = k * face_bytes});
            if constexpr (has_post) {
                noc.async_read(post, cb_post_src, face_bytes, {.page_id = t0 + k}, {.offset_bytes = k * face_bytes});
            }
        }
        noc.async_read_barrier();

        cb_w.reserve_back(n_w);
        for (uint32_t i = 0; i < n_out; ++i) {
            for (uint32_t j = 0; j < hc; ++j) {
                volatile tt_l1_ptr uint16_t* tile = w_dst + (i * hc + j) * tile_elems;
                // comb [hc, hc]: element (j, i). pre [1, hc]: element (0, j).
                const uint32_t src = n_out == 1 ? j : j * 16u + i;
                for (uint32_t k = 0; k < tile_h; ++k) {
                    tile[tile_face_index(k, k)] = k < valid ? w_src[k * face_elems + src] : 0;
                }
            }
        }
        if constexpr (has_post) {
            for (uint32_t i = 0; i < n_out; ++i) {
                volatile tt_l1_ptr uint16_t* tile = w_dst + (hc * n_out + i) * tile_elems;
                // post [n_out, 1]: element (i, 0).
                for (uint32_t k = 0; k < tile_h; ++k) {
                    tile[tile_face_index(k, k)] = k < valid ? post_src[k * face_elems + i * 16u] : 0;
                }
            }
        }
        cb_w.push_back(n_w);

        for (uint32_t u = unit; u < group_end; ++u) {
            const uint32_t page = r * d_tiles + (u % d_tiles);
            cb_in.reserve_back(n_in);
            for (uint32_t j = 0; j < hc; ++j) {
                noc.async_read(
                    streams, cb_in, tile_bytes, {.page_id = j * stream_tiles + page}, {.offset_bytes = j * tile_bytes});
            }
            if constexpr (has_post) {
                noc.async_read(sub, cb_in, tile_bytes, {.page_id = page}, {.offset_bytes = hc * tile_bytes});
            }
            noc.async_read_barrier();
            cb_in.push_back(n_in);
        }
        unit = group_end;
    }
}
