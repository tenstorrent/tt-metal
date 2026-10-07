// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pad reader: RM HEIGHT_SHARDED input, back-padding only.
// NCRISC. Each core reads from its local L1 shard (no NOC for data sticks).
// Data sticks: L1 self-read from shard address. W-pad appended from pad CB.
// Pad sticks (H/C/N): fill entirely from pad CB.
//
// The host builder computes per-core: shard L1 addr, data stick count,
// output coordinates, and total output sticks for this core's range.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

template <uint32_t num_bytes>
inline __attribute__((always_inline)) void fill_with_val(uint32_t dst, uint32_t val) {
    static_assert(num_bytes % sizeof(uint16_t) == 0, "RM pad values are 2B or 4B scalars");
    CoreLocalMem<volatile uint32_t> ptr(dst);
    for (uint32_t i = 0; i < num_bytes / sizeof(uint32_t); ++i) {
        ptr[i] = val;
    }
    if constexpr (num_bytes % sizeof(uint32_t) != 0) {
        CoreLocalMem<volatile uint16_t> tail(dst + (num_bytes / sizeof(uint32_t)) * sizeof(uint32_t));
        *tail = static_cast<uint16_t>(val);
    }
}

void kernel_main() {
    // Runtime args
    uint32_t shard_l1_addr = get_arg_val<uint32_t>(0);  // L1 address of this core's input shard
    uint32_t num_sticks = get_arg_val<uint32_t>(1);     // total output sticks this core produces
    uint32_t start_h = get_arg_val<uint32_t>(2);
    uint32_t start_c = get_arg_val<uint32_t>(3);
    uint32_t start_n = get_arg_val<uint32_t>(4);

    // Compile-time args
    constexpr uint32_t H = get_compile_time_arg_val(0);  // input H
    constexpr uint32_t C = get_compile_time_arg_val(1);  // input C
    constexpr uint32_t N = get_compile_time_arg_val(2);  // input N
    constexpr uint32_t H_padded = get_compile_time_arg_val(3);
    constexpr uint32_t C_padded = get_compile_time_arg_val(4);
    constexpr uint32_t N_padded = get_compile_time_arg_val(5);
    constexpr uint32_t stick_size = get_compile_time_arg_val(6);      // input W * elem_size
    constexpr uint32_t stick_size_out = get_compile_time_arg_val(7);  // W_padded * elem_size
    constexpr uint32_t stick_size_out_aligned = get_compile_time_arg_val(8);
    constexpr uint32_t pad_w_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t packed_pad_val = get_compile_time_arg_val(10);
    constexpr uint32_t BATCH = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out = get_compile_time_arg_val(12);
    constexpr uint32_t cb_pad = get_compile_time_arg_val(13);
    constexpr uint32_t shard_stick_stride = get_compile_time_arg_val(14);  // bytes between sticks in shard

    Noc noc;
    CircularBuffer out_cb(cb_out);
    CircularBuffer pad_cb(cb_pad);

    // Fill pad constant buffer
    fill_with_val<stick_size_out>(pad_cb.get_write_ptr(), packed_pad_val);
    const uint32_t pad_l1_addr = pad_cb.get_read_ptr();
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];

    uint32_t curr_h = start_h;
    uint32_t curr_c = start_c;
    uint32_t curr_n = start_n;
    uint32_t local_data_idx = 0;  // index into this core's shard
    uint32_t sticks_left = num_sticks;

    while (sticks_left > 0) {
        uint32_t batch = (sticks_left < BATCH) ? sticks_left : BATCH;
        out_cb.reserve_back(batch);
        uint32_t l1_addr = out_cb.get_write_ptr();

        for (uint32_t t = 0; t < batch; t++) {
            bool is_data = (curr_h < H) && (curr_c < C) && (curr_n < N);

            if (is_data) {
                // L1 self-read from shard (fast — no NOC fabric traversal)
                uint32_t src_l1 = shard_l1_addr + local_data_idx * shard_stick_stride;
                CoreLocalMem<uint32_t> out_mem(l1_addr);
                noc.async_read(
                    self_ep,
                    out_mem,
                    stick_size,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = src_l1},
                    {.offset_bytes = 0});
                local_data_idx++;

                // Append W-pad if needed
                if constexpr (pad_w_bytes > 0) {
                    noc.async_read(
                        self_ep,
                        out_mem,
                        pad_w_bytes,
                        {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = pad_l1_addr},
                        {.offset_bytes = stick_size});
                }
            } else {
                // Full pad stick
                CoreLocalMem<uint32_t> out_mem(l1_addr);
                noc.async_read(
                    self_ep,
                    out_mem,
                    stick_size_out,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = pad_l1_addr},
                    {.offset_bytes = 0});
            }

            l1_addr += stick_size_out_aligned;

            // Advance output coordinates
            curr_h++;
            if (curr_h == H_padded) {
                curr_h = 0;
                curr_c++;
                if (curr_c == C_padded) {
                    curr_c = 0;
                    curr_n++;
                }
            }
        }
        noc.async_read_barrier();  // ONE barrier for entire batch
        out_cb.push_back(batch);
        sticks_left -= batch;
    }
}
