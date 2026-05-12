// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"

void kernel_main() {
    int i{0};
    const auto output_addr = get_arg_val<uint32_t>(i++);
    const auto mean_addr = get_arg_val<uint32_t>(i++);
    const auto rstd_addr = get_arg_val<uint32_t>(i++);

    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto num_inner_tiles = get_arg_val<uint32_t>(i++);

    const auto num_groups = get_arg_val<uint32_t>(i++);
    const auto block_size = get_arg_val<uint32_t>(i++);

    constexpr bool mean_has_value = get_compile_time_arg_val(0) == 1;
    constexpr bool rstd_has_value = get_compile_time_arg_val(1) == 1;
    constexpr auto output_args = TensorAccessorArgs<2>();
    constexpr auto mean_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto rstd_args = TensorAccessorArgs<mean_args.next_compile_time_args_offset()>();

    uint32_t cb_id{16};
    const auto cb_id_output = cb_id++;
    const auto cb_id_mean = cb_id++;
    const auto cb_id_rstd = cb_id++;

    // output
    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);
    const auto output_addrg = TensorAccessor(output_args, output_addr);

    // mean
    const uint32_t mean_tile_bytes = get_tile_size(cb_id_mean);
    const auto mean_addrg = TensorAccessor(mean_args, mean_addr);

    // rstd
    const uint32_t rstd_tile_bytes = get_tile_size(cb_id_rstd);
    const auto rstd_addrg = TensorAccessor(rstd_args, rstd_addr);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const auto start_mean_rstd_idx = tile_offset / num_inner_tiles;

    experimental::Noc noc;
    experimental::CircularBuffer cb_output(cb_id_output);
    experimental::CircularBuffer cb_mean(cb_id_mean);
    experimental::CircularBuffer cb_rstd(cb_id_rstd);

    const auto output_l1_read_ptr = cb_output.get_read_ptr();
    uint32_t output_tile_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; ++outer_idx) {
        // mean, rstd (1, 1, N, num_groups)
        // mean_rstd_tile_idx = n * num_groups + g
        const auto mean_rstd_idx = start_mean_rstd_idx + outer_idx;
        const auto mean_rstd_n_idx = mean_rstd_idx / num_groups;
        const auto mean_rstd_g_idx = mean_rstd_idx % num_groups;

        const auto mean_rstd_tile_h_idx = mean_rstd_n_idx / TILE_H;
        const auto mean_rstd_tile_w_idx = mean_rstd_g_idx / TILE_W;

        const auto mean_rstd_h_idx_in_tile = mean_rstd_n_idx % TILE_H;
        const auto mean_rstd_w_idx_in_tile = mean_rstd_g_idx % TILE_W;

        const auto mean_rstd_Wt = (num_groups + TILE_W - 1) / TILE_W;

        const auto mean_rstd_tile_idx = mean_rstd_tile_h_idx * mean_rstd_Wt + mean_rstd_tile_w_idx;

        const auto tilized_mean_rstd_idx_in_tile =
            get_tilized_idx(mean_rstd_h_idx_in_tile, mean_rstd_w_idx_in_tile, TILE_H, TILE_W);

        // mean (1, 1, N, num_groups)
        if (mean_has_value) {
            const auto mean_dtype_bytes = mean_tile_bytes / (TILE_H * TILE_W);
            const auto mean_l1_read_ptr = cb_mean.get_read_ptr();
            cb_mean.wait_front(onetile);
            if (tilized_mean_rstd_idx_in_tile != 0) {
                experimental::CoreLocalMem<uint16_t> mean_ptr(mean_l1_read_ptr);
                mean_ptr[tilized_mean_rstd_idx_in_tile] = mean_ptr[0];
            }
            noc.async_write(
                cb_mean,
                mean_addrg,
                mean_dtype_bytes,
                {.offset_bytes = tilized_mean_rstd_idx_in_tile * mean_dtype_bytes},
                {.page_id = mean_rstd_tile_idx, .offset_bytes = tilized_mean_rstd_idx_in_tile * mean_dtype_bytes});
            noc.async_write_barrier();
            cb_mean.pop_front(onetile);
        }

        // rstd (1, 1, N, num_groups)
        if (rstd_has_value) {
            const auto rstd_dtype_bytes = rstd_tile_bytes / (TILE_H * TILE_W);
            const auto rstd_l1_read_ptr = cb_rstd.get_read_ptr();
            cb_rstd.wait_front(onetile);
            if (tilized_mean_rstd_idx_in_tile != 0) {
                experimental::CoreLocalMem<uint16_t> rstd_ptr(rstd_l1_read_ptr);
                rstd_ptr[tilized_mean_rstd_idx_in_tile] = rstd_ptr[0];
            }
            noc.async_write(
                cb_rstd,
                rstd_addrg,
                rstd_dtype_bytes,
                {.offset_bytes = tilized_mean_rstd_idx_in_tile * rstd_dtype_bytes},
                {.page_id = mean_rstd_tile_idx, .offset_bytes = tilized_mean_rstd_idx_in_tile * rstd_dtype_bytes});
            noc.async_write_barrier();
            cb_rstd.pop_front(onetile);
        }

        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; inner_idx += block_size) {
            // output (N, C, H, W)
            cb_output.wait_front(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                output_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + r;
                noc.async_write(
                    cb_output,
                    output_addrg,
                    output_tile_bytes,
                    {.offset_bytes = r * output_tile_bytes},
                    {.page_id = output_tile_idx});
            }
            noc.async_write_barrier();
            cb_output.pop_front(block_size);
        }  // inner_idx loop
    }  // outer_idx loop

}  // void kernel_main()
