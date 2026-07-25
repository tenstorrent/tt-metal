// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <typename D, typename T>
void write_mean_rstd(
    const Noc& noc,
    D dfb,
    uint32_t tile_offset,
    uint32_t num_inner,
    uint32_t normalized_dims,
    uint32_t outer_idx,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t Ht,
    uint32_t Wt,
    T addrg) {
    using namespace tt::constants;
    constexpr uint32_t onetile = 1;

    const uint32_t cb_tile_bytes = dfb.get_tile_size();
    const auto cb_dtype_bytes = cb_tile_bytes / (TILE_HEIGHT * TILE_WIDTH);

    dfb.wait_front(onetile);

    uint32_t output_l1_write_addr = dfb.get_read_ptr();
    CoreLocalMem<volatile uint16_t> l1_ptr(output_l1_write_addr);

    uint32_t output_tile_offset = tile_offset / num_inner;

    if (normalized_dims == 1) {
        for (uint32_t src_h = 0; src_h < 2; src_h++) {
            auto output_tile_idx = output_tile_offset + outer_idx;

            auto wt = output_tile_idx % Wt;
            auto nh = output_tile_idx / Wt;
            auto h = nh % output_height;
            auto n = nh / output_height;

            auto w = src_h * FACE_HEIGHT;

            auto tilized_idx = get_tilized_idx(h % TILE_HEIGHT, w);

            auto ht = h / TILE_HEIGHT;
            auto noc_id = n * Ht * Wt + ht * Wt + wt;

            auto src_idx = get_tilized_idx(0, src_h * FACE_WIDTH);

            noc.async_write(
                dfb,
                addrg,
                cb_dtype_bytes * FACE_HEIGHT,
                {.offset_bytes = src_idx * cb_dtype_bytes},
                {.page_id = noc_id, .offset_bytes = tilized_idx * cb_dtype_bytes});
            noc.async_write_barrier();
        }
    } else {
        auto output_idx = output_tile_offset + outer_idx;

        auto w = output_idx % output_width;
        auto nh = output_idx / output_width;
        auto h = nh % output_height;
        auto n = nh / output_height;

        auto tilized_idx = get_tilized_idx(h % TILE_HEIGHT, w % TILE_WIDTH);

        auto wt = w / TILE_WIDTH;
        auto ht = h / TILE_HEIGHT;

        auto noc_id = n * Ht * Wt + ht * Wt + wt;

        if (output_idx != 0) {
            l1_ptr[tilized_idx] = l1_ptr[0];
        }

        noc.async_write(
            dfb,
            addrg,
            cb_dtype_bytes,
            {.offset_bytes = tilized_idx * cb_dtype_bytes},
            {.page_id = noc_id, .offset_bytes = tilized_idx * cb_dtype_bytes});
        noc.async_write_barrier();
    }

    dfb.pop_front(onetile);
}

void kernel_main() {
    using namespace tt::constants;
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto num_inner = get_arg(args::num_inner);
    const auto tile_offset = get_arg(args::tile_offset);
    const auto mean_rstd_height = get_arg(args::mean_rstd_height);
    const auto mean_rstd_width = get_arg(args::mean_rstd_width);
    const auto normalized_dims = get_arg(args::normalized_dims);

    constexpr auto block_size = get_arg(args::block_size);

    Noc noc;

    // output
    DataflowBuffer dfb_output(dfb::output);
    const uint32_t output_tile_bytes = dfb_output.get_tile_size();
    const auto output_addrg = TensorAccessor(tensor::output);

#ifdef MEAN_HAS_VALUE
    // mean
    DataflowBuffer dfb_mean(dfb::mean);
    const auto mean_addrg = TensorAccessor(tensor::mean);
#endif

#ifdef RSTD_HAS_VALUE
    // rstd
    DataflowBuffer dfb_rstd(dfb::rstd);
    const auto rstd_addrg = TensorAccessor(tensor::rstd);
#endif

    uint32_t offs = 0;
    constexpr uint32_t onetile = 1;

    uint32_t Wt = (mean_rstd_width + TILE_WIDTH - 1) / TILE_WIDTH;
    uint32_t Ht = (mean_rstd_height + TILE_HEIGHT - 1) / TILE_HEIGHT;

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
#ifdef MEAN_HAS_VALUE
        write_mean_rstd(
            noc,
            dfb_mean,
            tile_offset,
            num_inner,
            normalized_dims,
            outer_idx,
            mean_rstd_height,
            mean_rstd_width,
            Ht,
            Wt,
            mean_addrg);
#endif

#ifdef RSTD_HAS_VALUE
        write_mean_rstd(
            noc,
            dfb_rstd,
            tile_offset,
            num_inner,
            normalized_dims,
            outer_idx,
            mean_rstd_height,
            mean_rstd_width,
            Ht,
            Wt,
            rstd_addrg);
#endif

        // output
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            dfb_output.wait_front(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                noc.async_write(
                    dfb_output,
                    output_addrg,
                    output_tile_bytes,
                    {.offset_bytes = r * output_tile_bytes},
                    {.page_id = offs + inner_idx + r + tile_offset});
            }
            noc.async_write_barrier();
            dfb_output.pop_front(block_size);
        }  // num_inner loop

        offs += num_inner;
    }  // num_rows_per_core loop
}  // void kernel_main()
