// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Streams one (token-tile, column-tile) work unit at a time: the y tile followed by the n
// residual-stream tiles at that position, as one contiguous CB_IN block. The raw post/comb tiles
// are pushed only when the token-tile changes; compute derives the same token-tile from the same
// unit index, so the two stay in lockstep without extra runtime args.
void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_arg_val<uint32_t>(3);
    const uint32_t consts_addr = get_arg_val<uint32_t>(4);
    const uint32_t num_units = get_arg_val<uint32_t>(5);
    const uint32_t start_unit = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_pc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_consts = get_compile_time_arg_val(2);
    constexpr uint32_t n = get_compile_time_arg_val(3);
    constexpr uint32_t col_tiles = get_compile_time_arg_val(4);
    constexpr auto y_args = TensorAccessorArgs<5>();
    constexpr auto residual_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr auto consts_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();

    const uint32_t page = get_local_cb_interface(cb_in).fifo_page_size;

    const auto s_y = TensorAccessor(y_args, y_addr);
    const auto s_residual = TensorAccessor(residual_args, residual_addr);
    const auto s_post = TensorAccessor(post_args, post_addr);
    const auto s_comb = TensorAccessor(comb_args, comb_addr);
    const auto s_consts = TensorAccessor(consts_args, consts_addr);

    Noc noc;
    CircularBuffer cbi(cb_in), cbp(cb_pc), cbc(cb_consts);

    // Column-broadcast tiles: loaded once, never popped by compute -> resident for the whole op.
    for (uint32_t k = 0; k < n * n; ++k) {
        cbc.reserve_back(1);
        noc.async_read(s_consts, cbc, page, {.page_id = k}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbc.push_back(1);
    }

    uint32_t cached_t0 = 0xFFFFFFFFu;
    for (uint32_t w = 0; w < num_units; ++w) {
        const uint32_t unit = start_unit + w;
        const uint32_t t0 = unit / col_tiles;
        const uint32_t c0 = unit - t0 * col_tiles;

        if (t0 != cached_t0) {
            cached_t0 = t0;
            cbp.reserve_back(2);
            noc.async_read(s_post, cbp, page, {.page_id = t0}, {.offset_bytes = 0});
            noc.async_read(s_comb, cbp, page, {.page_id = t0}, {.offset_bytes = page});
            noc.async_read_barrier();
            cbp.push_back(2);
        }

        cbi.reserve_back(1 + n);
        noc.async_read(s_y, cbi, page, {.page_id = t0 * col_tiles + c0}, {.offset_bytes = 0});
        const uint32_t residual_base = t0 * n * col_tiles + c0;
        for (uint32_t i = 0; i < n; ++i) {
            noc.async_read(
                s_residual, cbi, page, {.page_id = residual_base + i * col_tiles}, {.offset_bytes = (i + 1) * page});
        }
        noc.async_read_barrier();
        cbi.push_back(1 + n);
    }
}
