// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto tile_offset = get_arg(args::tile_offset);
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto num_inner_tiles = get_arg(args::num_inner_tiles);

    const auto num_groups = get_arg(args::num_groups);
    const auto block_size = get_arg(args::block_size);

    // output
    DataflowBuffer dfb_output(dfb::output);
    const uint32_t output_tile_bytes = dfb_output.get_tile_size();
    const auto output_addrg = TensorAccessor(tensor::output);

    // mean
#ifdef MEAN_HAS_VALUE
    DataflowBuffer dfb_mean(dfb::mean);
    const uint32_t mean_tile_bytes = dfb_mean.get_tile_size();
    const auto mean_addrg = TensorAccessor(tensor::mean);
#endif

    // rstd
#ifdef RSTD_HAS_VALUE
    DataflowBuffer dfb_rstd(dfb::rstd);
    const uint32_t rstd_tile_bytes = dfb_rstd.get_tile_size();
    const auto rstd_addrg = TensorAccessor(tensor::rstd);
#endif

    constexpr uint32_t onetile = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const auto start_mean_rstd_idx = tile_offset / num_inner_tiles;

    Noc noc;

    const auto output_l1_read_ptr = dfb_output.get_read_ptr();
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
#ifdef MEAN_HAS_VALUE
        {
            const auto mean_dtype_bytes = mean_tile_bytes / (TILE_H * TILE_W);
            const auto mean_l1_read_ptr = dfb_mean.get_read_ptr();
            dfb_mean.wait_front(onetile);
            if (tilized_mean_rstd_idx_in_tile != 0) {
                CoreLocalMem<uint16_t> mean_ptr(mean_l1_read_ptr);
                mean_ptr[tilized_mean_rstd_idx_in_tile] = mean_ptr[0];
            }
            noc.async_write(
                dfb_mean,
                mean_addrg,
                mean_dtype_bytes,
                {.offset_bytes = tilized_mean_rstd_idx_in_tile * mean_dtype_bytes},
                {.page_id = mean_rstd_tile_idx, .offset_bytes = tilized_mean_rstd_idx_in_tile * mean_dtype_bytes});
            noc.async_write_barrier();
            dfb_mean.pop_front(onetile);
        }
#endif

        // rstd (1, 1, N, num_groups)
#ifdef RSTD_HAS_VALUE
        {
            const auto rstd_dtype_bytes = rstd_tile_bytes / (TILE_H * TILE_W);
            const auto rstd_l1_read_ptr = dfb_rstd.get_read_ptr();
            dfb_rstd.wait_front(onetile);
            if (tilized_mean_rstd_idx_in_tile != 0) {
                CoreLocalMem<uint16_t> rstd_ptr(rstd_l1_read_ptr);
                rstd_ptr[tilized_mean_rstd_idx_in_tile] = rstd_ptr[0];
            }
            noc.async_write(
                dfb_rstd,
                rstd_addrg,
                rstd_dtype_bytes,
                {.offset_bytes = tilized_mean_rstd_idx_in_tile * rstd_dtype_bytes},
                {.page_id = mean_rstd_tile_idx, .offset_bytes = tilized_mean_rstd_idx_in_tile * rstd_dtype_bytes});
            noc.async_write_barrier();
            dfb_rstd.pop_front(onetile);
        }
#endif

        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; inner_idx += block_size) {
            // output (N, C, H, W)
            dfb_output.wait_front(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                output_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + r;
                noc.async_write(
                    dfb_output,
                    output_addrg,
                    output_tile_bytes,
                    {.offset_bytes = r * output_tile_bytes},
                    {.page_id = output_tile_idx});
            }
            noc.async_write_barrier();
            dfb_output.pop_front(block_size);
        }  // inner_idx loop
    }  // outer_idx loop

}  // void kernel_main()
