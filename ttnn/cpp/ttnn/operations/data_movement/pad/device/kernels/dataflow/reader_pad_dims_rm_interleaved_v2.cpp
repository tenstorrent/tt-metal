// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port (in place — used only by the PadRmReaderWriterMultiCoreDefault factory).
// Logic UNCHANGED; only the access mechanism moves to named bindings:
//   - input src tensor address       -> ta::src
//   - input CB c_0                    -> dfb::in0   (producer; reader fills, writer drains)
//   - pad-fill scratch CB c_1         -> dfb::pad   (self-loop: address-source scratch only)
//   - pad-align scratch CB c_2        -> dfb::pad_align (conditional self-loop; bound + referenced
//                                        only when FRONT_PAD_OR_UNALIGNED, mirroring the host's
//                                        conditional c_2 allocation)
//   - positional CTAs/RTAs            -> get_arg(args::...); start_dim_offset array -> varargs
// c_1 and c_2 are "fake" CBs: the kernel never produces/consumes them as a FIFO, it only grabs a
// base pointer (get_read_ptr / get_write_ptr) and reads/writes the L1 region directly. They are
// bound as self-loops (producer + consumer on this reader) to satisfy the DFB producer/consumer
// invariant — interim, until a Metal 2.0 kernel-scratchpad / local-TensorAccessor resource lands.

#include <stdint.h>
#include <cstring>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

inline __attribute__((always_inline)) void fill_pad_cb_with_val(
    const uint32_t cb_id, const uint32_t num_bytes, const uint32_t val) {
    DataflowBuffer cb(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());

    for (uint32_t i = 0; i < num_bytes / 2; ++i) {
        ptr[i] = val;
    }
}

// Helper to read multiple input pages for a single stick into L1.
// This encapsulates the common pattern of reading (num_pages - 1) full pages
// followed by a final partially filled page, and advances i_page accordingly.
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
    uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    uint32_t num_sticks_per_barrier = get_arg(args::num_sticks_per_barrier);
    uint32_t start_page_id = get_arg(args::start_page_id);
    uint32_t front_pad_n = get_arg(args::front_pad_n);
    uint32_t front_pad_c = get_arg(args::front_pad_c);
    uint32_t front_pad_h = get_arg(args::front_pad_h);
    // start_dim_offset[]: per-core starting dim indices (num_dims long), passed as varargs.
    uint32_t start_dim_offset[4];
    for (uint32_t d = 0; d < 4; ++d) {
        start_dim_offset[d] = get_vararg(d);
    }

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

    constexpr uint32_t cb_in0 = dfb::in0;
    constexpr uint32_t cb_pad = dfb::pad;
    DataflowBuffer cb_in0_exp(cb_in0);
    DataflowBuffer cb_pad_exp(cb_pad);
#if defined(FRONT_PAD_OR_UNALIGNED)
    constexpr uint32_t cb_pad_align = dfb::pad_align;
    DataflowBuffer cb_pad_align_exp(cb_pad_align);
#endif

    const auto s = TensorAccessor(ta::src);
    Noc noc;

    const uint32_t pad_val_addr = cb_pad_exp.get_read_ptr();
#if defined(FRONT_PAD_OR_UNALIGNED)
    const uint32_t pad_align_addr = cb_pad_align_exp.get_read_ptr();
#endif

    fill_pad_cb_with_val(cb_pad, stick_size_padded, packed_pad_value);

    uint32_t i_page = start_page_id;
    uint32_t curr_c = start_dim_offset[2], curr_h = start_dim_offset[1], curr_n = start_dim_offset[3];
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        cb_in0_exp.reserve_back(num_sticks_per_barrier);
        uint32_t l1_write_addr = cb_in0_exp.get_write_ptr();

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
#if defined(FRONT_PAD_OR_UNALIGNED)
                if constexpr (front_padding) {  // Read noc into cb_pad_align l1
                    uint32_t temp_addr = cb_pad_align_exp.get_write_ptr();
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
                        (void*)(cb_pad_align_exp.get_read_ptr()),
                        (size_t)(stick_size_bytes));
                } else if constexpr (unaligned) {
                    uint32_t temp_addr = cb_pad_align_exp.get_write_ptr();
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
                } else
#endif
                {
                    read_input_pages_into_l1(
                        noc,
                        s,
                        i_page,
                        l1_write_addr,
                        num_input_pages_in_row,
                        input_page_size,
                        size_of_valid_data_in_last_input_page_in_row);
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
        cb_in0_exp.push_back(num_sticks_per_barrier);
    }
}
