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
#include "ckernel.h"
#include "experimental/kernel_args.h"

inline __attribute__((always_inline)) void fill_pad_dfb_with_val(
    DataflowBuffer& dfb, const uint32_t num_bytes, const uint32_t val) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr());

    // Round up so a non-4-byte-aligned tail stick is fully filled (the loop-back read consumes all num_bytes).
    const uint32_t num_words = (num_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_words; ++i) {
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
    auto num_sticks_per_core = get_arg(args::num_sticks_per_core);
    auto num_sticks_per_barrier = get_arg(args::num_sticks_per_barrier);
    auto start_page_id = get_arg(args::start_page_id);
    auto front_pad_n = get_arg(args::front_pad_n);
    auto front_pad_c = get_arg(args::front_pad_c);
    auto front_pad_h = get_arg(args::front_pad_h);

    constexpr auto N = get_arg(args::N);
    constexpr auto H = get_arg(args::H);
    constexpr auto C = get_arg(args::C);
    constexpr auto stick_size_bytes = get_arg(args::stick_size_bytes);
    constexpr auto N_padded = get_arg(args::N_padded);
    constexpr auto H_padded = get_arg(args::H_padded);
    constexpr auto C_padded = get_arg(args::C_padded);
    constexpr auto stick_size_padded = get_arg(args::stick_size_padded);
    constexpr auto stick_size_padded_front = get_arg(args::stick_size_padded_front);
    constexpr auto stick_size_padded_aligned = get_arg(args::stick_size_padded_aligned);

    constexpr bool not_pad_by_zero = get_arg(args::not_pad_by_zero) == 1;
    constexpr uint32_t front_padding = stick_size_padded_front;
    constexpr bool unaligned = get_arg(args::unaligned) == 1;

    constexpr auto num_input_pages_in_row = get_arg(args::num_input_pages_in_row);

    uint32_t packed_pad_value = 0;
    if constexpr (not_pad_by_zero) {
        packed_pad_value = get_arg(args::packed_pad_value);
    }

    DataflowBuffer dfb_in0_exp(dfb::in0);
    DataflowBuffer dfb_pad_exp(dfb::pad);
    // The realignment staging buffer is bound only when the host allocated it (front padding, or
    // an unaligned padded stick). A kernel may not name a DFB it has not bound, and `if constexpr`
    // does not suppress that name lookup, so every reference to it is gated at the preprocessor.
#ifdef PAD_ALIGN_DFB
    DataflowBuffer dfb_pad_align_exp(dfb::pad_align);
#endif

    const auto s = TensorAccessor(tensor::src);
    Noc noc;

    const uint32_t pad_val_addr = dfb_pad_exp.get_read_ptr();
#ifdef PAD_ALIGN_DFB
    const uint32_t pad_align_addr = dfb_pad_align_exp.get_read_ptr();
#endif

    fill_pad_dfb_with_val(dfb_pad_exp, stick_size_padded, packed_pad_value);
    // The fill above is baby-RISCV stores; the per-stick loop below loop-back noc.async_read's the pad DFB as
    // its source. A baby-RISCV store can retire before its write-request lands in L1, and the RISCV core
    // and NoC are different L1 clients with no program-order guarantee between them
    // (WormholeB0/TensixTile/BabyRISCV/MemoryOrdering.md). load_blocking the last filled word (blocking
    // load + memory clobber) to force the fill to be processed before the first loop-back read is issued.
    // One-time cost, outside the per-stick loop.
    (void)ckernel::load_blocking(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pad_val_addr) + (stick_size_padded / sizeof(uint32_t)) - 1);

    uint32_t i_page = start_page_id;
    uint32_t curr_c = get_arg(args::start_dim_offset_c), curr_h = get_arg(args::start_dim_offset_h),
             curr_n = get_arg(args::start_dim_offset_n);
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
#ifdef PAD_ALIGN_DFB
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
                } else
#endif
                {
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
