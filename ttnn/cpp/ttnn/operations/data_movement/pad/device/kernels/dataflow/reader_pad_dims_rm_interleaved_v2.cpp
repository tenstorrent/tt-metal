// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

inline __attribute__((always_inline)) void fill_pad_cb_with_val(
    const uint32_t cb_id, const uint32_t num_bytes, const uint32_t val) {
    DataflowBuffer dfb(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr());

    for (uint32_t i = 0; i < num_bytes / 2; ++i) {
        ptr[i] = val;
    }
}

template <typename StreamState>
inline __attribute__((always_inline)) void read_input_stick_into_l1(
    Noc& noc,
    const StreamState& s,
    uint32_t& i_page,
    uint32_t l1_write_addr,
    const uint32_t num_input_pages_in_row,
    const uint32_t stick_size_bytes) {
    if (num_input_pages_in_row == 1) {
        // Width fits in a single page: index the accessor with the flat page id directly.
        // `noc_async_read_sharded` derives pages-per-row from the (rank-squeezed) dspec shape,
        // which is wrong when an outer dim is sharded and the width is a single page (the
        // width-page dim gets squeezed away and an inner dim is mistaken for the row width).
        noc.async_read(
            s, CoreLocalMem<uint32_t>(l1_write_addr), stick_size_bytes, {.page_id = i_page, .offset_bytes = 0}, {});
    } else {
        const uint32_t stick_id = i_page / num_input_pages_in_row;
        tt::data_movement::common::noc_async_read_sharded(
            noc, l1_write_addr, s, stick_id, /*offset=*/0, /*size=*/stick_size_bytes);
    }
    i_page += num_input_pages_in_row;
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(1);
    uint32_t num_sticks_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t start_page_id = get_arg_val<uint32_t>(3);
    uint32_t front_pad_n = get_arg_val<uint32_t>(4);
    uint32_t front_pad_c = get_arg_val<uint32_t>(5);
    uint32_t front_pad_h = get_arg_val<uint32_t>(6);
    tt_l1_ptr uint32_t* start_dim_offset = (tt_l1_ptr uint32_t*)(get_arg_addr(7));

    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t N_padded = get_compile_time_arg_val(4);
    constexpr uint32_t H_padded = get_compile_time_arg_val(5);
    constexpr uint32_t C_padded = get_compile_time_arg_val(6);
    constexpr uint32_t stick_size_padded = get_compile_time_arg_val(7);
    constexpr uint32_t stick_size_padded_front = get_compile_time_arg_val(8);
    constexpr uint32_t stick_size_padded_end = get_compile_time_arg_val(9);
    constexpr uint32_t num_zero_pad_sticks_read = get_compile_time_arg_val(10);
    constexpr uint32_t last_zero_stick_size = get_compile_time_arg_val(11);
    constexpr uint32_t stick_size_padded_aligned = get_compile_time_arg_val(18);

    constexpr bool not_pad_by_zero = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t front_padding = get_compile_time_arg_val(8);
    constexpr bool unaligned = get_compile_time_arg_val(19) == 1;

    constexpr uint32_t num_input_pages_in_row = get_compile_time_arg_val(20);
    constexpr uint32_t accessor_page_size = get_compile_time_arg_val(21);
    constexpr auto src_args = TensorAccessorArgs<22>();

    uint32_t packed_pad_value = 0;
    if constexpr (not_pad_by_zero) {
        packed_pad_value = kernel_compile_time_args[13];
    }

    constexpr uint32_t dfb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_pad = tt::CBIndex::c_1;
    constexpr uint32_t dfb_pad_align = tt::CBIndex::c_2;
    DataflowBuffer dfb_in0_exp(dfb_in0);
    DataflowBuffer dfb_pad_exp(cb_pad);
    DataflowBuffer dfb_pad_align_exp(dfb_pad_align);

    const auto s = TensorAccessor(src_args, src_addr, accessor_page_size);
    Noc noc;

    const uint32_t pad_val_addr = dfb_pad_exp.get_read_ptr();
    const uint32_t pad_align_addr = dfb_pad_align_exp.get_read_ptr();

    fill_pad_cb_with_val(cb_pad, stick_size_padded, packed_pad_value);

    uint32_t i_page = start_page_id;
    uint32_t curr_c = start_dim_offset[2], curr_h = start_dim_offset[1], curr_n = start_dim_offset[3];
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        dfb_in0_exp.reserve_back(num_sticks_per_barrier);
        uint32_t l1_write_addr = dfb_in0_exp.get_write_ptr();

        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            bool read_stick = (curr_h >= front_pad_h and curr_h < H) and (curr_c >= front_pad_c and curr_c < C) and
                              (curr_n >= front_pad_n and curr_n < N);
            {
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(
                    UnicastEndpoint{},
                    dst,
                    stick_size_padded,
                    {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                     .addr = pad_val_addr},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
            }
            if (read_stick) {
                if constexpr (front_padding) {
                    uint32_t temp_addr = dfb_pad_align_exp.get_write_ptr();
                    read_input_stick_into_l1(noc, s, i_page, temp_addr, num_input_pages_in_row, stick_size_bytes);
                    noc.async_read_barrier();
                    memmove(
                        (void*)(l1_write_addr + stick_size_padded_front),
                        (void*)(dfb_pad_align_exp.get_read_ptr()),
                        (size_t)(stick_size_bytes));
                } else if constexpr (unaligned) {
                    uint32_t temp_addr = dfb_pad_align_exp.get_write_ptr();
                    read_input_stick_into_l1(noc, s, i_page, temp_addr, num_input_pages_in_row, stick_size_bytes);
                    noc.async_read_barrier();
                    CoreLocalMem<uint32_t> dst(l1_write_addr);
                    noc.async_read(
                        UnicastEndpoint{},
                        dst,
                        stick_size_bytes,
                        {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                         .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                         .addr = pad_align_addr},
                        {.offset_bytes = 0});
                } else {
                    read_input_stick_into_l1(noc, s, i_page, l1_write_addr, num_input_pages_in_row, stick_size_bytes);
                }
            }
            l1_write_addr += stick_size_padded_aligned;
            curr_h++;
            if (curr_h == H_padded) {
                curr_c++;
                curr_h = 0;
                if (curr_c == C_padded) {
                    curr_n++;
                    curr_c = 0;
                }
            }
        }
        noc.async_read_barrier();
        dfb_in0_exp.push_back(num_sticks_per_barrier);
    }
}
