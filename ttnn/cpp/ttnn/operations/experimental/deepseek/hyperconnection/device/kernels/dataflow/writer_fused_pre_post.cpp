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

void kernel_main() {
    const uint32_t post_out_addr = get_arg_val<uint32_t>(0);
    const uint32_t collapsed_addr = get_arg_val<uint32_t>(1);
    const uint32_t comb_w_mat_addr = get_arg_val<uint32_t>(2);
    const uint32_t d_tiles = get_arg_val<uint32_t>(3);
    const uint32_t start_token = get_arg_val<uint32_t>(4);
    const uint32_t num_tokens = get_arg_val<uint32_t>(5);
    const uint32_t pre_out_addr = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_post_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_collapsed = get_compile_time_arg_val(1);
    constexpr uint32_t cb_comb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_post_col = get_compile_time_arg_val(3);
    constexpr uint32_t num_streams = get_compile_time_arg_val(4);
    // Given pre_mix, the computed pre is returned too: [1,T,1,H], one tile per token.
    constexpr bool use_pre_mix = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t cb_pre_out = get_compile_time_arg_val(6);
    constexpr uint32_t cb_pre_row = get_compile_time_arg_val(7);

    constexpr auto post_out_args = TensorAccessorArgs<8>();
    constexpr auto collapsed_args = TensorAccessorArgs<post_out_args.next_compile_time_args_offset()>();
    constexpr auto comb_w_args = TensorAccessorArgs<collapsed_args.next_compile_time_args_offset()>();
    constexpr auto pre_out_args = TensorAccessorArgs<comb_w_args.next_compile_time_args_offset()>();

    const auto post_out = TensorAccessor(post_out_args, post_out_addr);
    const auto collapsed = TensorAccessor(collapsed_args, collapsed_addr);
    const auto comb_w = TensorAccessor(comb_w_args, comb_w_mat_addr);
    const auto pre_out = TensorAccessor(pre_out_args, pre_out_addr);

    Noc noc;
    CircularBuffer cb_post(cb_post_out);
    CircularBuffer cb_col(cb_collapsed);
    CircularBuffer cb_cw(cb_comb_w);
    CircularBuffer cb_pc(cb_post_col);
    CircularBuffer cb_p(cb_pre_out);
    CircularBuffer cb_pr(cb_pre_row);

    constexpr uint32_t one_tile = 1;
    const uint32_t post_tile_size = cb_post.get_tile_size();
    const uint32_t col_tile_size = cb_col.get_tile_size();

    // post is emitted as [1,T,H,1] -- one column per token -- but the compute kernel produces
    // it as a row (row 0, cols 0..H-1). Transpose it into a scratch tile that is zeroed once;
    // only the H entries of column 0 are rewritten per token, so the padding stays zero.
    cb_pc.reserve_back(one_tile);
    noc.async_write_zeros(cb_pc, post_tile_size, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    volatile tt_l1_ptr uint16_t* post_col = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_pc.get_write_ptr());

    // The same for pre, whose compute tile carries sigmoid(bias) + eps in its padding: only row 0's
    // H values are copied out, so the returned pre is zero-padded.
    volatile tt_l1_ptr uint16_t* pre_row = nullptr;
    if constexpr (use_pre_mix) {
        cb_pr.reserve_back(one_tile);
        noc.async_write_zeros(cb_pr, post_tile_size, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        pre_row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_pr.get_write_ptr());
    }

    for (uint32_t i = 0; i < num_tokens; ++i) {
        const uint32_t token = start_token + i;

        cb_post.wait_front(one_tile);
        const volatile tt_l1_ptr uint16_t* post_row =
            reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(cb_post.get_read_ptr());
        for (uint32_t k = 0; k < num_streams; ++k) {
            post_col[tile_face_index(k, 0)] = post_row[tile_face_index(0, k)];
        }
        noc.async_write(cb_pc, post_out, post_tile_size, {.offset_bytes = 0}, {.page_id = token});

        // collapsed [1,T,1,D] -> d_tiles tiles along the width, one row of the grid per token.
        cb_col.wait_front(d_tiles);
        for (uint32_t n = 0; n < d_tiles; ++n) {
            noc.async_write(
                cb_col,
                collapsed,
                col_tile_size,
                {.offset_bytes = n * col_tile_size},
                {.page_id = token * d_tiles + n});
        }

        // comb_w_mat [1,T,H,H] -> one tile per token (laid out by the reader).
        cb_cw.wait_front(one_tile);
        noc.async_write(cb_cw, comb_w, cb_cw.get_tile_size(), {.offset_bytes = 0}, {.page_id = token});

        // pre [1,T,1,H] -> one tile per token.
        if constexpr (use_pre_mix) {
            cb_p.wait_front(one_tile);
            const volatile tt_l1_ptr uint16_t* pre =
                reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(cb_p.get_read_ptr());
            for (uint32_t k = 0; k < num_streams; ++k) {
                pre_row[tile_face_index(0, k)] = pre[tile_face_index(0, k)];
            }
            noc.async_write(cb_pr, pre_out, post_tile_size, {.offset_bytes = 0}, {.page_id = token});
        }

        // The post / pre scratches are reused next iteration, so every write must land before the pops.
        noc.async_write_barrier();

        cb_post.pop_front(one_tile);
        cb_col.pop_front(d_tiles);
        cb_cw.pop_front(one_tile);
        if constexpr (use_pre_mix) {
            cb_p.pop_front(one_tile);
        }
    }
}
