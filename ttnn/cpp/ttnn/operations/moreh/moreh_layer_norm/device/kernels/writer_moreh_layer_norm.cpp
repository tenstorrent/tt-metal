// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

template <typename T>
void write_mean_rstd(uint32_t cb_id, uint32_t tile_offset, uint32_t num_inner, uint32_t normalized_dims, uint32_t outer_idx, uint32_t output_height, uint32_t output_width, uint32_t Ht, uint32_t Wt, T addrg)
{
    constexpr uint32_t onetile = 1;

    const uint32_t cb_tile_bytes = get_tile_size(cb_id);
    const auto cb_dtype_bytes = cb_tile_bytes / (TILE_HEIGHT * TILE_WIDTH);

    cb_wait_front(cb_id, onetile);

    uint32_t output_l1_write_addr = get_read_ptr(cb_id);
    auto l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(output_l1_write_addr);

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

            auto dst_noc_addr = get_noc_addr(noc_id, addrg);
            noc_async_write(
                output_l1_write_addr + src_idx * cb_dtype_bytes,
                dst_noc_addr + tilized_idx * cb_dtype_bytes,
                cb_dtype_bytes * FACE_HEIGHT);
            noc_async_write_barrier();
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

        auto dst_noc_addr = get_noc_addr(noc_id, addrg);
        noc_async_write(
            output_l1_write_addr + tilized_idx * cb_dtype_bytes,
            dst_noc_addr + tilized_idx * cb_dtype_bytes,
            cb_dtype_bytes);
        noc_async_write_barrier();
    }

    cb_pop_front(cb_id, onetile);
}

void kernel_main() {
    const auto output_addr = get_arg_val<uint32_t>(0);
    const auto mean_addr = get_arg_val<uint32_t>(1);
    const auto rstd_addr = get_arg_val<uint32_t>(2);
    const auto num_rows_per_core = get_arg_val<uint32_t>(3);
    const auto num_inner = get_arg_val<uint32_t>(4);
    const auto tile_offset = get_arg_val<uint32_t>(5);
    const auto mean_rstd_height = get_arg_val<uint32_t>(6);
    const auto mean_rstd_width = get_arg_val<uint32_t>(7);
    const auto normalized_dims = get_arg_val<uint32_t>(8);

    constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool mean_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool rstd_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool mean_has_value = get_compile_time_arg_val(3) == 1;
    constexpr bool rstd_has_value = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t block_size = get_compile_time_arg_val(5);

    constexpr uint32_t cb_id_output = tt::CB::c_out0;
    constexpr uint32_t cb_id_mean = tt::CB::c_out1;
    constexpr uint32_t cb_id_rstd = tt::CB::c_out2;

    // output
    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);
    const auto output_data_format = get_dataformat(cb_id_output);

    const InterleavedAddrGenFast<output_is_dram> output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    // mean
    const uint32_t mean_tile_bytes = get_tile_size(cb_id_mean);
    const auto mean_data_format = get_dataformat(cb_id_mean);

    const InterleavedAddrGenFast<mean_is_dram> mean_addrg = {
        .bank_base_address = mean_addr, .page_size = mean_tile_bytes, .data_format = mean_data_format};

    // rstd
    const uint32_t rstd_tile_bytes = get_tile_size(cb_id_rstd);
    const auto rstd_data_format = get_dataformat(cb_id_rstd);

    const InterleavedAddrGenFast<rstd_is_dram> rstd_addrg = {
        .bank_base_address = rstd_addr, .page_size = rstd_tile_bytes, .data_format = rstd_data_format};

    uint32_t offs = 0;
    constexpr uint32_t onetile = 1;

    uint32_t Wt = (mean_rstd_width + TILE_WIDTH - 1) / TILE_WIDTH;
    uint32_t Ht = (mean_rstd_height + TILE_HEIGHT - 1) / TILE_HEIGHT;

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        if (mean_has_value) {
            write_mean_rstd(cb_id_mean, tile_offset, num_inner, normalized_dims, outer_idx, mean_rstd_height, mean_rstd_width, Ht, Wt, mean_addrg);
        }

        if (rstd_has_value) {
            write_mean_rstd(cb_id_rstd, tile_offset, num_inner, normalized_dims, outer_idx, mean_rstd_height, mean_rstd_width, Ht, Wt, rstd_addrg);
        }

        // output
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            cb_wait_front(cb_id_output, block_size);
            auto output_l1_read_addr = get_read_ptr(cb_id_output);
            for (uint32_t r = 0; r < block_size; r++) {
                noc_async_write_tile(offs + inner_idx + r + tile_offset, output_addrg, output_l1_read_addr);
                output_l1_read_addr += output_tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_id_output, block_size);
        }  // num_inner loop

        offs += num_inner;
    }  // num_rows_per_core loop
}  // void kernel_main()
