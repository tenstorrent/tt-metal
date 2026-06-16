// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port (in place — used only by the PadRmShardedHeightOnly factory).
// Logic UNCHANGED; only the access mechanism moves to named bindings:
//   - pad-value scratch CB c_1 -> dfb::pad  (fake CB: address-source scratch only -> self-loop)
//   - output shard CB c_16     -> dfb::out0 (borrowed output shard; written by base pointer)
//   - positional CTAs/RTAs     -> get_arg(args::...); start_dim_offset array -> varargs.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

inline __attribute__((always_inline)) void fill_pad_cb_with_val(
    Noc& noc, const uint32_t cb_id, const uint32_t num_bytes_risc, uint32_t num_noc_transfer, const uint32_t val) {
    DataflowBuffer cb(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());

    for (uint32_t i = 0; i < num_bytes_risc / 2; ++i) {
        ptr[i] = val;
    }

    uint32_t pad_val_addr = cb.get_write_ptr();
    uint32_t l1_write_addr = pad_val_addr;

    for (uint32_t i = 0; i < num_noc_transfer; ++i) {
        CoreLocalMem<uint32_t> dst(l1_write_addr);
        noc.async_read(
            UnicastEndpoint{},
            dst,
            num_bytes_risc,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
             .addr = pad_val_addr},
            {.offset_bytes = 0});
        l1_write_addr += num_bytes_risc;
    }
    noc.async_read_barrier();
}

inline __attribute__((always_inline)) void fill_pad_cb_with_zero(
    Noc& noc, const uint32_t cb_id, const uint32_t num_bytes_risc, uint32_t num_noc_transfer) {
    DataflowBuffer cb(cb_id);
    noc.async_write_zeros(cb, num_bytes_risc * num_noc_transfer);
    noc.write_zeros_l1_barrier();
}

void kernel_main() {
    uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    uint32_t start_id = get_arg(args::start_id);
    uint32_t front_pad_n = get_arg(args::front_pad_n);
    uint32_t front_pad_c = get_arg(args::front_pad_c);
    uint32_t front_pad_h = get_arg(args::front_pad_h);
    // start_dim_offset[]: per-core starting dim indices, passed as varargs (kernel indexes [1],[2],[3]).
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
    constexpr uint32_t num_zero_pad_sticks_read = get_arg(args::num_zero_pad_sticks_read);
    constexpr uint32_t zero_pad_stick_size = get_arg(args::zero_pad_stick_size);

    constexpr bool not_pad_by_zero = get_arg(args::not_pad_by_zero) == 1;
    uint32_t packed_pad_value = 0;
    uint32_t row_major_min_bytes = 0;
    uint32_t num_sticks_padded_read = 0;
    if constexpr (not_pad_by_zero) {
        packed_pad_value = get_arg(args::packed_pad_value);
        row_major_min_bytes = get_arg(args::row_major_min_bytes);
        num_sticks_padded_read = get_arg(args::num_sticks_padded_read);
    }

    constexpr auto cb_pad = dfb::pad;
    constexpr auto cb_out0 = dfb::out0;
    DataflowBuffer cb_pad_exp(cb_pad);
    DataflowBuffer cb_out0_exp(cb_out0);

    Noc noc;

    const uint32_t pad_val_addr = cb_pad_exp.get_read_ptr();

    if constexpr (not_pad_by_zero) {
        fill_pad_cb_with_val(noc, cb_pad, row_major_min_bytes, num_sticks_padded_read, packed_pad_value);
    } else {
        fill_pad_cb_with_zero(noc, cb_pad, zero_pad_stick_size, num_zero_pad_sticks_read);
    }

    uint32_t l1_write_addr = cb_out0_exp.get_write_ptr();

    uint32_t i_stick = start_id;
    uint32_t curr_c = start_dim_offset[2], curr_h = start_dim_offset[1], curr_n = start_dim_offset[3];
    for (uint32_t iter = 0; iter < num_sticks_per_core; ++iter) {
        bool read_stick = (curr_h >= front_pad_h and curr_h < H) and (curr_c >= front_pad_c and curr_c < C) and
                          (curr_n >= front_pad_n and curr_n < N);

        if (read_stick) {
            l1_write_addr += stick_size_bytes;
            i_stick++;

        } else {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(
                UnicastEndpoint{},
                dst,
                stick_size_bytes,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = pad_val_addr},
                {.offset_bytes = 0});
            l1_write_addr += stick_size_bytes;
        }

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
}
