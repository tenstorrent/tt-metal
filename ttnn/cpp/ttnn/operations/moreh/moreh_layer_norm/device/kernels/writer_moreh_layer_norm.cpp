// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"

template <typename T>
void write_mean_rstd(
    const experimental::Noc& noc,
    uint32_t cb_id,
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

    experimental::CircularBuffer cb(cb_id);
    const uint32_t cb_tile_bytes = get_tile_size(cb_id);
    const auto cb_dtype_bytes = cb_tile_bytes / (TILE_HEIGHT * TILE_WIDTH);

    cb.wait_front(onetile);

    uint32_t output_l1_write_addr = cb.get_read_ptr();
    experimental::CoreLocalMem<volatile uint16_t> l1_ptr(output_l1_write_addr);

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
                cb,
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
            cb,
            addrg,
            cb_dtype_bytes,
            {.offset_bytes = tilized_idx * cb_dtype_bytes},
            {.page_id = noc_id, .offset_bytes = tilized_idx * cb_dtype_bytes});
        noc.async_write_barrier();
    }

    cb.pop_front(onetile);
}

void kernel_main() {
    using namespace tt::constants;
    const auto output_addr = get_arg_val<uint32_t>(0);
    const auto mean_addr = get_arg_val<uint32_t>(1);
    const auto rstd_addr = get_arg_val<uint32_t>(2);
    const auto num_rows_per_core = get_arg_val<uint32_t>(3);
    const auto num_inner = get_arg_val<uint32_t>(4);
    const auto tile_offset = get_arg_val<uint32_t>(5);
    const auto mean_rstd_height = get_arg_val<uint32_t>(6);
    const auto mean_rstd_width = get_arg_val<uint32_t>(7);
    const auto normalized_dims = get_arg_val<uint32_t>(8);

    constexpr bool mean_has_value = get_compile_time_arg_val(0) == 1;
    constexpr bool rstd_has_value = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t block_size = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();
    constexpr auto mean_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto rstd_args = TensorAccessorArgs<mean_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_output = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_mean = tt::CBIndex::c_17;
    constexpr uint32_t cb_id_rstd = tt::CBIndex::c_18;

    // output
    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);
    const auto output_addrg = TensorAccessor(output_args, output_addr);

    // mean
    const auto mean_addrg = TensorAccessor(mean_args, mean_addr);

    // rstd
    const auto rstd_addrg = TensorAccessor(rstd_args, rstd_addr);

    uint32_t offs = 0;
    constexpr uint32_t onetile = 1;

    uint32_t Wt = (mean_rstd_width + TILE_WIDTH - 1) / TILE_WIDTH;
    uint32_t Ht = (mean_rstd_height + TILE_HEIGHT - 1) / TILE_HEIGHT;

    experimental::Noc noc;
    experimental::CircularBuffer cb_output(cb_id_output);

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        if (mean_has_value) {
            write_mean_rstd(
                noc,
                cb_id_mean,
                tile_offset,
                num_inner,
                normalized_dims,
                outer_idx,
                mean_rstd_height,
                mean_rstd_width,
                Ht,
                Wt,
                mean_addrg);
        }

        if (rstd_has_value) {
            write_mean_rstd(
                noc,
                cb_id_rstd,
                tile_offset,
                num_inner,
                normalized_dims,
                outer_idx,
                mean_rstd_height,
                mean_rstd_width,
                Ht,
                Wt,
                rstd_addrg);
        }

        // output
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            cb_output.wait_front(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                noc.async_write(
                    cb_output,
                    output_addrg,
                    output_tile_bytes,
                    {.offset_bytes = r * output_tile_bytes},
                    {.page_id = offs + inner_idx + r + tile_offset});
            }
            noc.async_write_barrier();
            cb_output.pop_front(block_size);
        }  // num_inner loop

        offs += num_inner;
    }  // num_rows_per_core loop
}  // void kernel_main()
