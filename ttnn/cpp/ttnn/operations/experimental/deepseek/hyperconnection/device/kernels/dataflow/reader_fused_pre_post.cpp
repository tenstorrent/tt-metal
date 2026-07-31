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

// fused_w [1,1,1,(2+H)*H] in TILE layout: value k lives in tile floor(k/32), row 0, col (k%32).
// Returns the in-tile element index (multiply by sizeof(bf16) for the byte offset).
FORCE_INLINE uint32_t fused_w_face_index(uint32_t k) { return tile_face_index(0, k & 31u); }

}  // namespace

void kernel_main() {
    const uint32_t fused_w_addr = get_arg_val<uint32_t>(0);
    const uint32_t pre_bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_bias_addr = get_arg_val<uint32_t>(2);
    const uint32_t hidden_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_streams = get_arg_val<uint32_t>(4);  // H
    const uint32_t fused_w_tiles = get_arg_val<uint32_t>(5);
    const uint32_t d_tiles = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_fused_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_pre_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_post_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_comb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_pre_bias = get_compile_time_arg_val(4);
    constexpr uint32_t cb_post_bias = get_compile_time_arg_val(5);
    constexpr uint32_t cb_hidden = get_compile_time_arg_val(6);

    constexpr auto fused_w_args = TensorAccessorArgs<7>();
    constexpr auto pre_bias_args = TensorAccessorArgs<fused_w_args.next_compile_time_args_offset()>();
    constexpr auto post_bias_args = TensorAccessorArgs<pre_bias_args.next_compile_time_args_offset()>();
    constexpr auto hidden_args = TensorAccessorArgs<post_bias_args.next_compile_time_args_offset()>();

    const auto fused_w = TensorAccessor(fused_w_args, fused_w_addr);
    const auto pre_bias = TensorAccessor(pre_bias_args, pre_bias_addr);
    const auto post_bias = TensorAccessor(post_bias_args, post_bias_addr);
    const auto hidden = TensorAccessor(hidden_args, hidden_addr);

    Noc noc;
    CircularBuffer cb_fw(cb_fused_w);
    CircularBuffer cb_pw(cb_pre_w);
    CircularBuffer cb_ppw(cb_post_w);
    CircularBuffer cb_cw(cb_comb_w);
    CircularBuffer cb_pb(cb_pre_bias);
    CircularBuffer cb_ppb(cb_post_bias);
    CircularBuffer cb_h(cb_hidden);

    constexpr uint32_t one_tile = 1;
    const uint32_t tile_size_bytes = cb_fw.get_tile_size();

    // 1. Pull the packed fused_w projection into L1 scratch (cb_fused_w). It is mined
    //    below for the pre_w / post_w / comb_w slices; no consumer pops it.
    cb_fw.reserve_back(fused_w_tiles);
    const uint32_t fused_w_l1 = cb_fw.get_write_ptr();
    for (uint32_t t = 0; t < fused_w_tiles; ++t) {
        noc.async_read(fused_w, cb_fw, tile_size_bytes, {.page_id = t}, {.offset_bytes = t * tile_size_bytes});
    }

    // 2. Bias + hidden tiles (consumed by the compute kernel as before).
    cb_pb.reserve_back(one_tile);
    cb_ppb.reserve_back(one_tile);
    noc.async_read(pre_bias, cb_pb, cb_pb.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(post_bias, cb_ppb, cb_ppb.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});

    const uint32_t hidden_tile_size = cb_h.get_tile_size();
    cb_h.reserve_back(d_tiles);
    for (uint32_t n = 0; n < d_tiles; ++n) {
        noc.async_read(hidden, cb_h, hidden_tile_size, {.page_id = n}, {.offset_bytes = n * hidden_tile_size});
    }

    noc.async_read_barrier();

    // 3. Split fused_w into pre_w / post_w / comb_w tiles directly in L1.
    //    pre_w[k]  = fused_w[k]         (k = 0..H-1)    -> row 0, col k
    //    post_w[k] = fused_w[H + k]      (k = 0..H-1)    -> row 0, col k
    //    comb_w[k] = fused_w[2H + k]     (k = 0..H*H-1) -> row k/H, col k%H
    const volatile tt_l1_ptr uint16_t* fused_w_ptr = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(fused_w_l1);

    auto zero_tile = [](volatile tt_l1_ptr uint16_t* dst) {
        for (uint32_t i = 0; i < 1024; ++i) {
            dst[i] = 0;
        }
    };

    auto fill_row0_slice = [&](CircularBuffer& cb, uint32_t base_k, uint32_t count) {
        cb.reserve_back(one_tile);
        volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_write_ptr());
        zero_tile(dst);
        for (uint32_t k = 0; k < count; ++k) {
            const uint32_t src_idx = fused_w_face_index(base_k + k);
            const uint32_t src_tile = (base_k + k) / 32u;
            const uint32_t dst_idx = tile_face_index(0, k);
            dst[dst_idx] = fused_w_ptr[src_tile * (tile_size_bytes / 2u) + src_idx];
        }
        cb.push_back(one_tile);
    };

    fill_row0_slice(cb_pw, 0u, num_streams);
    fill_row0_slice(cb_ppw, num_streams, num_streams);

    // comb_w -> [1,1,H,H] grid (single tile).
    cb_cw.reserve_back(one_tile);
    volatile tt_l1_ptr uint16_t* comb_dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_cw.get_write_ptr());
    zero_tile(comb_dst);
    const uint32_t comb_base = 2u * num_streams;
    const uint32_t comb_count = num_streams * num_streams;
    for (uint32_t k = 0; k < comb_count; ++k) {
        const uint32_t r = k / num_streams;
        const uint32_t c = k % num_streams;
        const uint32_t src_idx = fused_w_face_index(comb_base + k);
        const uint32_t src_tile = (comb_base + k) / 32u;
        const uint32_t dst_idx = tile_face_index(r, c);
        comb_dst[dst_idx] = fused_w_ptr[src_tile * (tile_size_bytes / 2u) + src_idx];
    }
    cb_cw.push_back(one_tile);

    cb_pb.push_back(one_tile);
    cb_ppb.push_back(one_tile);
    cb_h.push_back(d_tiles);
}
