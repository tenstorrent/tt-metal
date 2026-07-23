// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of the height-only sharded pad writer (private to PadRmShardedHeightOnlyProgramFactory).
// Self-loop DFBs are no longer permitted on data-movement kernels, so cb_pad is now a CROSS-KERNEL DFB:
// the reader PRODUCES the pad-value stick (the fill logic moved there); this writer CONSUMES it (wait_front
// -> read its address -> broadcast pad sticks -> pop_front). The c_16 output shard is written in place
// via tensor::output (NOC_LOCAL_ADDR_OFFSET(s_out.get_noc_addr(0))) — no borrowed co-write DFB. start_dim_offset
// is read by constant indices so it is three named scalar RTAs.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t N = get_arg(args::N);
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t C = get_arg(args::C);
    constexpr uint32_t stick_size_bytes = get_arg(args::stick_size_bytes);
    constexpr uint32_t N_padded = get_arg(args::N_padded);
    constexpr uint32_t H_padded = get_arg(args::H_padded);
    constexpr uint32_t C_padded = get_arg(args::C_padded);

    const uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t front_pad_n = get_arg(args::front_pad_n);
    const uint32_t front_pad_c = get_arg(args::front_pad_c);
    const uint32_t front_pad_h = get_arg(args::front_pad_h);
    const uint32_t start_dim_h = get_arg(args::start_dim_h);
    const uint32_t start_dim_c = get_arg(args::start_dim_c);
    const uint32_t start_dim_n = get_arg(args::start_dim_n);

    DataflowBuffer cb_pad(dfb::cb_pad);
    Noc noc;

    // The pad-value stick is produced by the reader (cross-kernel DFB); wait for it and use its address.
    cb_pad.wait_front(1);
    const uint32_t pad_val_addr = cb_pad.get_read_ptr();

    // Output shard base from the resident output TensorAccessor (written in place; no borrowed
    // co-write DFB — the reader writes the gathered sticks, this writer writes the pad sticks).
    const auto s_out = TensorAccessor(tensor::output);
    uint32_t l1_write_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(s_out.get_noc_addr(0));

    uint32_t i_stick = start_id;
    uint32_t curr_c = start_dim_c, curr_h = start_dim_h, curr_n = start_dim_n;
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
    cb_pad.pop_front(1);
}
