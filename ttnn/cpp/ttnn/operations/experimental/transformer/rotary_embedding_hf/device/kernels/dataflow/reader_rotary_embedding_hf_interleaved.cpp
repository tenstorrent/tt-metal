// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    uint32_t num_rows = get_arg(args::num_rows);
    uint32_t start_id = get_arg(args::start_id);
    uint32_t start_row_id = get_arg(args::start_row_id);
    uint32_t cos_sin_start_id = get_arg(args::cos_sin_start_id);

    constexpr uint16_t scalar_value = get_arg(args::scalar_value);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t HtWt = get_arg(args::HtWt);
    constexpr uint32_t half_Wt = get_arg(args::half_Wt);

    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_input(dfb::in);
    DataflowBuffer dfb_rotated_input(dfb::rotated_in);
    DataflowBuffer dfb_cos(dfb::cos);
    DataflowBuffer dfb_sin(dfb::sin);
    DataflowBuffer dfb_scalar(dfb::scalar);

    const uint32_t input_tile_bytes = dfb_input.get_tile_size();
    const auto s0 = TensorAccessor(tensor::src);

    const uint32_t cos_tile_bytes = dfb_cos.get_tile_size();
    const auto s1 = TensorAccessor(tensor::cos);

    const uint32_t sin_tile_bytes = dfb_sin.get_tile_size();
    const auto s2 = TensorAccessor(tensor::sin);

    // Fill tile with scalar value (-1)
    const uint32_t scalar_tile_bytes = dfb_scalar.get_tile_size();
    dfb_scalar.reserve_back(onetile);
    uint32_t l1_zeros_addr_in_scalar = dfb_scalar.get_write_ptr();
    volatile tt_l1_ptr uint16_t* scalar_buffer =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_zeros_addr_in_scalar);
    scalar_buffer[0] = scalar_value;
    dfb_scalar.push_back(onetile);

    uint32_t input_curr_id = start_id;
    uint32_t rotated_input_curr_id = start_id + half_Wt;
    uint32_t cos_sin_curr_id = cos_sin_start_id;
    uint32_t ht = start_row_id;

    // read a ublock of tiles from src to the DFB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            dfb_rotated_input.reserve_back(onetile);
            uint32_t rotated_input_l1_write_addr = dfb_rotated_input.get_write_ptr();
            noc.async_read(
                s0,
                CoreLocalMem<uint32_t>(rotated_input_l1_write_addr),
                input_tile_bytes,
                {.page_id = rotated_input_curr_id},
                {});
            noc.async_read_barrier();
            dfb_rotated_input.push_back(onetile);
            rotated_input_curr_id++;

            dfb_sin.reserve_back(onetile);
            uint32_t sin_l1_write_addr = dfb_sin.get_write_ptr();
            noc.async_read(
                s2, CoreLocalMem<uint32_t>(sin_l1_write_addr), sin_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
            dfb_sin.push_back(onetile);

            dfb_input.reserve_back(onetile);
            uint32_t input_l1_write_addr = dfb_input.get_write_ptr();
            noc.async_read(
                s0, CoreLocalMem<uint32_t>(input_l1_write_addr), input_tile_bytes, {.page_id = input_curr_id}, {});
            noc.async_read_barrier();
            dfb_input.push_back(onetile);
            input_curr_id++;

            dfb_cos.reserve_back(onetile);
            uint32_t cos_l1_write_addr = dfb_cos.get_write_ptr();
            noc.async_read(
                s1, CoreLocalMem<uint32_t>(cos_l1_write_addr), cos_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
            dfb_cos.push_back(onetile);
            cos_sin_curr_id++;

            if (j == half_Wt - 1) {
                rotated_input_curr_id -= Wt;
            }
        }
        rotated_input_curr_id += Wt;
        ht++;
        if (ht == Ht) {
            ht = 0;
            cos_sin_curr_id -= HtWt;
        }
    }
}
