// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of pad's RM multicore-default reader (private to
// PadRmReaderWriterMultiCoreDefaultProgramFactory). The device-side NoC + TensorAccessor logic is
// unchanged; only the resource access is migrated to the Metal 2.0 named handles (dfb::/tensor::/args::).
//   - c_0 input stream  -> dfb::cb_in0       (PRODUCER)
//   - c_1 pad scratch    -> dfb::cb_pad        (PRODUCER+CONSUMER self-loop)
//   - c_2 pad-align scratch -> dfb::cb_pad_align (PRODUCER+CONSUMER self-loop, only when HAS_PAD_ALIGN)
//   - the per-stick start dim offset array (legacy get_arg_addr(7)) is read by constant indices, so
//     it becomes three named scalar RTAs (start_dim_h / start_dim_c / start_dim_n).
#include <stdint.h>
#include <cstring>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// Fill a scratch DFB's backing L1 with the broadcast packed pad value.
inline __attribute__((always_inline)) void fill_pad_dfb_with_val(
    DataflowBuffer& cb, const uint32_t num_bytes, const uint32_t val) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    for (uint32_t i = 0; i < num_bytes / 2; ++i) {
        ptr[i] = val;
    }
}

// Read multiple input pages for a single stick into L1: (num_pages - 1) full pages followed by a
// final partially filled page; advances i_page accordingly.
template <typename StreamState>
inline __attribute__((always_inline)) void read_input_pages_into_l1(
    Noc& noc,
    const StreamState& s,
    uint32_t& i_page,
    uint32_t l1_write_addr,
    const uint32_t num_input_pages_in_row,
    const uint32_t input_page_size,
    const uint32_t size_of_valid_data_in_last_input_page_in_row) {
    uint32_t write_addr = l1_write_addr;
    for (uint32_t p = 0; p < num_input_pages_in_row - 1; ++p) {
        CoreLocalMem<uint32_t> dst(write_addr);
        noc.async_read(s, dst, input_page_size, {.page_id = i_page + p, .offset_bytes = 0}, {.offset_bytes = 0});
        write_addr += input_page_size;
    }
    CoreLocalMem<uint32_t> dst(write_addr);
    noc.async_read(
        s,
        dst,
        size_of_valid_data_in_last_input_page_in_row,
        {.page_id = i_page + num_input_pages_in_row - 1, .offset_bytes = 0},
        {.offset_bytes = 0});
    i_page += num_input_pages_in_row;
}

void kernel_main() {
    const uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const uint32_t num_sticks_per_barrier = get_arg(args::num_sticks_per_barrier);
    const uint32_t start_page_id = get_arg(args::start_page_id);
    const uint32_t front_pad_n = get_arg(args::front_pad_n);
    const uint32_t front_pad_c = get_arg(args::front_pad_c);
    const uint32_t front_pad_h = get_arg(args::front_pad_h);
    const uint32_t start_dim_h = get_arg(args::start_dim_h);
    const uint32_t start_dim_c = get_arg(args::start_dim_c);
    const uint32_t start_dim_n = get_arg(args::start_dim_n);

    constexpr uint32_t N = get_arg(args::N);
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t C = get_arg(args::C);
    constexpr uint32_t stick_size_bytes = get_arg(args::stick_size_bytes);
    constexpr uint32_t N_padded = get_arg(args::N_padded);
    constexpr uint32_t H_padded = get_arg(args::H_padded);
    constexpr uint32_t C_padded = get_arg(args::C_padded);
    constexpr uint32_t stick_size_padded = get_arg(args::stick_size_padded);
    constexpr uint32_t stick_size_padded_front = get_arg(args::stick_size_padded_front);
    constexpr uint32_t stick_size_padded_end = get_arg(args::stick_size_padded_end);
    constexpr uint32_t num_zero_pad_sticks_read = get_arg(args::num_zero_pad_sticks_read);
    constexpr uint32_t last_zero_stick_size = get_arg(args::last_zero_stick_size);
    constexpr uint32_t stick_size_padded_aligned = get_arg(args::stick_size_padded_aligned);

    constexpr bool not_pad_by_zero = get_arg(args::not_pad_by_zero) == 1;
    constexpr uint32_t front_padding = get_arg(args::stick_size_padded_front);
    constexpr bool unaligned = get_arg(args::unaligned) == 1;

    constexpr uint32_t num_input_pages_in_row = get_arg(args::num_input_pages_in_row);
    constexpr uint32_t input_page_size = get_arg(args::input_page_size);
    constexpr uint32_t size_of_valid_data_in_last_input_page_in_row =
        get_arg(args::size_of_valid_data_in_last_input_page_in_row);

    uint32_t packed_pad_value = 0;
    if constexpr (not_pad_by_zero) {
        packed_pad_value = get_arg(args::packed_pad_value);
    }

    const auto s = TensorAccessor(tensor::src);
    Noc noc;
    DataflowBuffer cb_in0(dfb::cb_in0);
    DataflowBuffer cb_pad(dfb::cb_pad);

    const uint32_t pad_val_addr = cb_pad.get_read_ptr();
#ifdef HAS_PAD_ALIGN
    DataflowBuffer cb_pad_align(dfb::cb_pad_align);
    const uint32_t pad_align_addr = cb_pad_align.get_read_ptr();
#endif

    fill_pad_dfb_with_val(cb_pad, stick_size_padded, packed_pad_value);

    uint32_t i_page = start_page_id;
    uint32_t curr_c = start_dim_c, curr_h = start_dim_h, curr_n = start_dim_n;
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        cb_in0.reserve_back(num_sticks_per_barrier);
        uint32_t l1_write_addr = cb_in0.get_write_ptr();

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
#ifdef HAS_PAD_ALIGN
                if constexpr (front_padding) {  // Read noc into cb_pad_align l1
                    uint32_t temp_addr = cb_pad_align.get_write_ptr();
                    read_input_pages_into_l1(
                        noc,
                        s,
                        i_page,
                        temp_addr,
                        num_input_pages_in_row,
                        input_page_size,
                        size_of_valid_data_in_last_input_page_in_row);
                    noc.async_read_barrier();
                    memmove(
                        (void*)(l1_write_addr + stick_size_padded_front),
                        (void*)(cb_pad_align.get_read_ptr()),
                        (size_t)(stick_size_bytes));
                } else if constexpr (unaligned) {
                    uint32_t temp_addr = cb_pad_align.get_write_ptr();
                    read_input_pages_into_l1(
                        noc,
                        s,
                        i_page,
                        temp_addr,
                        num_input_pages_in_row,
                        input_page_size,
                        size_of_valid_data_in_last_input_page_in_row);
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
                    read_input_pages_into_l1(
                        noc,
                        s,
                        i_page,
                        l1_write_addr,
                        num_input_pages_in_row,
                        input_page_size,
                        size_of_valid_data_in_last_input_page_in_row);
                }
#else
                read_input_pages_into_l1(
                    noc,
                    s,
                    i_page,
                    l1_write_addr,
                    num_input_pages_in_row,
                    input_page_size,
                    size_of_valid_data_in_last_input_page_in_row);
#endif
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
        cb_in0.push_back(num_sticks_per_barrier);
    }
}
