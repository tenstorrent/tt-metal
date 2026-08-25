// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    uint32_t num_rows = get_arg(args::num_rows);
#ifndef DECODE_MODE
    // Unread in decode mode: the cos/sin row bookkeeping it seeds only runs in the prefill loop.
    uint32_t start_row_id = get_arg(args::start_row_id);
#endif
    uint32_t cos_sin_start_id = get_arg(args::cos_sin_start_id);

    constexpr uint16_t scalar_value = get_arg(args::scalar_value);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t HtWt = get_arg(args::HtWt);
    constexpr uint32_t half_Wt_size = get_arg(args::half_Wt_size);

    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_rotated_input(dfb::rotated_input);
    DataflowBuffer dfb_cos(dfb::cos);
    DataflowBuffer dfb_sin(dfb::sin);
    DataflowBuffer dfb_scalar(dfb::scalar);

    dfb_input.reserve_back(num_rows * Wt);
    dfb_input.push_back(num_rows * Wt);
    uint32_t input_l1_read_addr = dfb_input.get_read_ptr();

    const uint32_t cos_tile_bytes = dfb_cos.get_tile_size();
    const auto s1 = TensorAccessor(tensor::cos);

    const uint32_t sin_tile_bytes = dfb_sin.get_tile_size();
    const auto s2 = TensorAccessor(tensor::sin);

    // Fill tile with zeros
    const uint32_t scalar_tile_bytes = dfb_scalar.get_tile_size();
    dfb_scalar.reserve_back(onetile);
    uint32_t l1_zeros_addr_in_scalar = dfb_scalar.get_write_ptr();
    volatile tt_l1_ptr uint16_t* scalar_buffer =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_zeros_addr_in_scalar);
    scalar_buffer[0] = scalar_value;
    dfb_scalar.push_back(onetile);

    uint32_t cos_sin_curr_id = cos_sin_start_id;

#ifdef DECODE_MODE
    dfb_sin.reserve_back(Wt);
    dfb_cos.reserve_back(Wt);
    uint32_t sin_l1_write_addr = dfb_sin.get_write_ptr();
    uint32_t cos_l1_write_addr = dfb_cos.get_write_ptr();
    for (uint32_t i = 0; i < Wt; i++) {
        noc.async_read(s2, CoreLocalMem<uint32_t>(sin_l1_write_addr), sin_tile_bytes, {.page_id = cos_sin_curr_id}, {});
        noc.async_read(s1, CoreLocalMem<uint32_t>(cos_l1_write_addr), cos_tile_bytes, {.page_id = cos_sin_curr_id}, {});
        cos_sin_curr_id++;
        sin_l1_write_addr += sin_tile_bytes;
        cos_l1_write_addr += cos_tile_bytes;
    }
    noc.async_read_barrier();
    dfb_sin.push_back(Wt);
    dfb_cos.push_back(Wt);
#else
    uint32_t ht = start_row_id;
#endif

    uint32_t Wt_size = half_Wt_size + half_Wt_size;
    // read a ublock of tiles from src to the DFB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_rows; ++i) {
        dfb_rotated_input.reserve_back(Wt);
        uint32_t rotated_input_l1_write_addr = dfb_rotated_input.get_write_ptr();
        noc.async_read(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(rotated_input_l1_write_addr),
            half_Wt_size,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
             .addr = input_l1_read_addr + half_Wt_size},
            {});
        noc.async_read(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(rotated_input_l1_write_addr + half_Wt_size),
            half_Wt_size,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
             .addr = input_l1_read_addr},
            {});
        input_l1_read_addr += Wt_size;
        noc.async_read_barrier();
        dfb_rotated_input.push_back(Wt);

#ifndef DECODE_MODE
        for (uint32_t j = 0; j < Wt; ++j) {
            dfb_sin.reserve_back(onetile);
            uint32_t sin_l1_write_addr = dfb_sin.get_write_ptr();
            noc.async_read(
                s2, CoreLocalMem<uint32_t>(sin_l1_write_addr), sin_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
            dfb_sin.push_back(onetile);

            dfb_cos.reserve_back(onetile);
            uint32_t cos_l1_write_addr = dfb_cos.get_write_ptr();
            noc.async_read(
                s1, CoreLocalMem<uint32_t>(cos_l1_write_addr), cos_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
            dfb_cos.push_back(onetile);
            cos_sin_curr_id++;
        }
        ht++;
        if (ht == Ht) {
            ht = 0;
            cos_sin_curr_id -= HtWt;
        }
#endif
    }
}
