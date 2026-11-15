// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {

// Element (r, c) of a 32x32 tile: 4 faces of 16x16, row-major inside each face.
FORCE_INLINE uint32_t tile_face_index(uint32_t r, uint32_t c) {
    const uint32_t face = ((r >= 16) ? 2u : 0u) + ((c >= 16) ? 1u : 0u);
    return face * 256u + (r & 15u) * 16u + (c & 15u);
}

// Byte offset of row r's left half in a 32x32 bf16 tile (its right half is 512 B further).
FORCE_INLINE uint32_t row_offset(uint32_t r) { return (r < 16 ? 0u : 1024u) + (r & 15u) * 32u; }

}  // namespace

// A several-row block (h x 32 tiles). Once per core: M = every block row's comb^T, P = every row's post,
//   M[i*h + j, s*h + j] = comb_j[s, i],   P[i*h + j, j] = post_j[i],
// then per column tile c: the hc stream tiles (h x 32) of column c stacked into rows s*h .. s*h + h-1 of one
// 32x32 tile, and the sublayer block tile into rows 0..h-1 (all other rows stay zero).
void kernel_main() {
    const uint32_t post_addr = get_arg_val<uint32_t>(0);
    const uint32_t comb_addr = get_arg_val<uint32_t>(1);
    const uint32_t sub_addr = get_arg_val<uint32_t>(2);
    const uint32_t streams_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_tile = get_arg_val<uint32_t>(4);
    const uint32_t num_tiles = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(1);
    constexpr uint32_t cb_post = get_compile_time_arg_val(2);
    constexpr uint32_t cb_streams = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sub = get_compile_time_arg_val(4);
    constexpr uint32_t hc = get_compile_time_arg_val(5);
    constexpr uint32_t h = get_compile_time_arg_val(6);
    constexpr uint32_t n_rows = get_compile_time_arg_val(7);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t buffering = get_compile_time_arg_val(9);

    constexpr auto post_args = TensorAccessorArgs<10>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr auto sub_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();
    constexpr auto streams_args = TensorAccessorArgs<sub_args.next_compile_time_args_offset()>();

    const auto post = TensorAccessor(post_args, post_addr);
    const auto comb = TensorAccessor(comb_args, comb_addr);
    const auto sub = TensorAccessor(sub_args, sub_addr);
    const auto streams = TensorAccessor(streams_args, streams_addr);

    if (num_tiles == 0) {
        return;
    }

    Noc noc;
    CircularBuffer src_cb(cb_src);
    CircularBuffer comb_cb(cb_comb);
    CircularBuffer post_cb(cb_post);
    CircularBuffer streams_cb(cb_streams);
    CircularBuffer sub_cb(cb_sub);
    const uint32_t tile_bytes = streams_cb.get_tile_size();
    // One (h x 32) tile's face: h rows of 16 bf16.
    constexpr uint32_t face_rows_bytes = h * 16 * sizeof(uint16_t);

    comb_cb.reserve_back(1);
    post_cb.reserve_back(1);
    streams_cb.reserve_back(buffering);
    sub_cb.reserve_back(buffering);
    noc.async_write_zeros(comb_cb, tile_bytes, {.offset_bytes = 0});
    noc.async_write_zeros(post_cb, tile_bytes, {.offset_bytes = 0});
    noc.async_write_zeros(streams_cb, buffering * tile_bytes, {.offset_bytes = 0});
    noc.async_write_zeros(sub_cb, buffering * tile_bytes, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();

    src_cb.reserve_back(1);
    const volatile tt_l1_ptr uint16_t* src =
        reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(src_cb.get_write_ptr());
    volatile tt_l1_ptr uint16_t* m = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(comb_cb.get_write_ptr());
    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(post_cb.get_write_ptr());
    for (uint32_t j = 0; j < n_rows; ++j) {
        noc.async_read(comb, src_cb, tile_bytes, {.page_id = j}, {.offset_bytes = 0});
        noc.async_read_barrier();
        for (uint32_t s = 0; s < hc; ++s) {
            for (uint32_t i = 0; i < hc; ++i) {
                m[tile_face_index(i * h + j, s * h + j)] = src[tile_face_index(s, i)];
            }
        }
        noc.async_read(post, src_cb, tile_bytes, {.page_id = j}, {.offset_bytes = 0});
        noc.async_read_barrier();
        for (uint32_t i = 0; i < hc; ++i) {
            p[tile_face_index(i * h + j, j)] = src[tile_face_index(i, 0)];
        }
    }
    comb_cb.push_back(1);
    post_cb.push_back(1);

    for (uint32_t c = start_tile; c < start_tile + num_tiles; ++c) {
        streams_cb.reserve_back(1);
        sub_cb.reserve_back(1);
        for (uint32_t s = 0; s < hc; ++s) {
            const uint32_t dst = row_offset(s * h);
            const uint32_t page = s * n_tiles + c;
            noc.async_read(streams, streams_cb, face_rows_bytes, {.page_id = page}, {.offset_bytes = dst});
            noc.async_read(
                streams,
                streams_cb,
                face_rows_bytes,
                {.page_id = page, .offset_bytes = face_rows_bytes},
                {.offset_bytes = dst + 512});
        }
        noc.async_read(sub, sub_cb, face_rows_bytes, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read(
            sub, sub_cb, face_rows_bytes, {.page_id = c, .offset_bytes = face_rows_bytes}, {.offset_bytes = 512});
        noc.async_read_barrier();
        streams_cb.push_back(1);
        sub_cb.push_back(1);
    }
}
