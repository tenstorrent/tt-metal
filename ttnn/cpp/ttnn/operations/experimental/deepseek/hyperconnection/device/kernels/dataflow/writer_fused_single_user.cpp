// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {

FORCE_INLINE uint32_t tile_face_index(uint32_t r, uint32_t c) {
    const uint32_t face = ((r >= 16) ? 2u : 0u) + ((c >= 16) ? 1u : 0u);
    return face * 256u + (r & 15u) * 16u + (c & 15u);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_post_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_post_col = get_compile_time_arg_val(2);
    constexpr uint32_t cb_comb_out = get_compile_time_arg_val(3);

    constexpr auto post_out_args = TensorAccessorArgs<4>();
    constexpr auto comb_out_args = TensorAccessorArgs<post_out_args.next_compile_time_args_offset()>();

    const uint32_t post_out_addr = get_arg_val<uint32_t>(0);
    const uint32_t comb_out_addr = get_arg_val<uint32_t>(1);
    const auto post_out = TensorAccessor(post_out_args, post_out_addr);
    const auto comb_out = TensorAccessor(comb_out_args, comb_out_addr);

    constexpr uint32_t one_tile = 1;
    Noc noc;

    if constexpr (role == 1) {
        CircularBuffer cb_post(cb_post_out);
        CircularBuffer cb_col(cb_post_col);
        const uint32_t tile_size_bytes = cb_post.get_tile_size();

        cb_col.reserve_back(one_tile);
        noc.async_write_zeros(cb_col, tile_size_bytes, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        auto* post_col = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_col.get_write_ptr());

        cb_post.wait_front(one_tile);
        const auto* post_row = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(cb_post.get_read_ptr());
        for (uint32_t k = 0; k < 32; ++k) {
            post_col[tile_face_index(k, 0)] = post_row[tile_face_index(0, k)];
        }
        noc.async_write(cb_col, post_out, tile_size_bytes, {.offset_bytes = 0}, {.page_id = 0});
        noc.async_write_barrier();
        cb_post.pop_front(one_tile);
    } else {
        CircularBuffer cb_comb(cb_comb_out);
        cb_comb.wait_front(one_tile);
        noc.async_write(cb_comb, comb_out, cb_comb.get_tile_size(), {.offset_bytes = 0}, {.page_id = 0});
        noc.async_write_barrier();
        cb_comb.pop_front(one_tile);
    }
}
