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

// Feeds one core's slice of the T-token grid. fused_w is [1,1,T,(2+H)*H] in TILE layout, so
// value k of token t lives in tile (t/32, k/32) at element (t%32, k%32); a whole tile row of
// fused_w is staged in L1 and mined for every token of this core that falls in it.
void kernel_main() {
    const uint32_t fused_w_addr = get_arg_val<uint32_t>(0);
    const uint32_t pre_bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_bias_addr = get_arg_val<uint32_t>(2);
    const uint32_t hidden_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_streams = get_arg_val<uint32_t>(4);  // H
    const uint32_t fused_w_row_tiles = get_arg_val<uint32_t>(5);
    const uint32_t d_tiles = get_arg_val<uint32_t>(6);
    const uint32_t start_token = get_arg_val<uint32_t>(7);
    const uint32_t num_tokens = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_fused_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_pre_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_post_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_comb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_pre_bias = get_compile_time_arg_val(4);
    constexpr uint32_t cb_post_bias = get_compile_time_arg_val(5);
    constexpr uint32_t cb_hidden = get_compile_time_arg_val(6);
    constexpr uint32_t slice_cb_pages = get_compile_time_arg_val(7);

    constexpr auto fused_w_args = TensorAccessorArgs<8>();
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
    const uint32_t tile_elems = tile_size_bytes / 2u;

    // 1. The two bias rows are the same for every token: read them once and never pop them,
    //    so the compute kernel can re-wait on the same tiles each iteration.
    cb_pb.reserve_back(one_tile);
    cb_ppb.reserve_back(one_tile);
    noc.async_read(pre_bias, cb_pb, cb_pb.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(post_bias, cb_ppb, cb_ppb.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_pb.push_back(one_tile);
    cb_ppb.push_back(one_tile);

    // 2. Zero every page of the pre_w / post_w / comb_w staging buffers once. The per-token
    //    rebuilds below only touch the valid H (resp. HxH) block, so the padding stays zero for
    //    the whole kernel: the collapse matmul's K reduction and the Sinkhorn masking both rely
    //    on that.
    cb_pw.reserve_back(slice_cb_pages);
    cb_ppw.reserve_back(slice_cb_pages);
    cb_cw.reserve_back(slice_cb_pages);
    noc.async_write_zeros(cb_pw, slice_cb_pages * tile_size_bytes, {.offset_bytes = 0});
    noc.async_write_zeros(cb_ppw, slice_cb_pages * tile_size_bytes, {.offset_bytes = 0});
    noc.async_write_zeros(cb_cw, slice_cb_pages * tile_size_bytes, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();

    // 3. fused_w scratch: one tile row at a time. It is never pushed, so its write pointer is
    //    stable and re-reads land in the same place.
    cb_fw.reserve_back(fused_w_row_tiles);
    const volatile tt_l1_ptr uint16_t* fused_w_ptr =
        reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(cb_fw.get_write_ptr());

    //    pre_w[k]  = fused_w[t][k]        (k = 0..H-1)   -> row 0, col k
    //    post_w[k] = fused_w[t][H + k]    (k = 0..H-1)   -> row 0, col k
    //    comb_w[k] = fused_w[t][2H + k]   (k = 0..H*H-1) -> row k/H, col k%H
    auto fused_w_at = [&](uint32_t row, uint32_t k) {
        return fused_w_ptr[(k >> 5) * tile_elems + tile_face_index(row, k & 31u)];
    };
    auto fill_row0_slice = [&](CircularBuffer& cb, uint32_t row, uint32_t base_k) {
        cb.reserve_back(one_tile);
        volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_write_ptr());
        for (uint32_t k = 0; k < num_streams; ++k) {
            dst[tile_face_index(0, k)] = fused_w_at(row, base_k + k);
        }
        cb.push_back(one_tile);
    };

    const uint32_t hidden_tile_size = cb_h.get_tile_size();
    const uint32_t comb_base = 2u * num_streams;
    uint32_t staged_tile_row = 0xFFFFFFFFu;

    for (uint32_t i = 0; i < num_tokens; ++i) {
        const uint32_t token = start_token + i;
        const uint32_t tile_row = token >> 5;
        const uint32_t row = token & 31u;

        if (tile_row != staged_tile_row) {
            for (uint32_t c = 0; c < fused_w_row_tiles; ++c) {
                noc.async_read(
                    fused_w,
                    cb_fw,
                    tile_size_bytes,
                    {.page_id = tile_row * fused_w_row_tiles + c},
                    {.offset_bytes = c * tile_size_bytes});
            }
            noc.async_read_barrier();
            staged_tile_row = tile_row;
        }

        fill_row0_slice(cb_pw, row, 0u);
        fill_row0_slice(cb_ppw, row, num_streams);

        cb_cw.reserve_back(one_tile);
        volatile tt_l1_ptr uint16_t* comb_dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_cw.get_write_ptr());
        for (uint32_t r = 0; r < num_streams; ++r) {
            for (uint32_t c = 0; c < num_streams; ++c) {
                comb_dst[tile_face_index(r, c)] = fused_w_at(row, comb_base + r * num_streams + c);
            }
        }
        cb_cw.push_back(one_tile);

        // H <= 32, so token `token` is tile row `token` of the flattened [T*H, D] hidden grid.
        cb_h.reserve_back(d_tiles);
        for (uint32_t n = 0; n < d_tiles; ++n) {
            noc.async_read(
                hidden,
                cb_h,
                hidden_tile_size,
                {.page_id = token * d_tiles + n},
                {.offset_bytes = n * hidden_tile_size});
        }
        noc.async_read_barrier();
        cb_h.push_back(d_tiles);
    }
}
