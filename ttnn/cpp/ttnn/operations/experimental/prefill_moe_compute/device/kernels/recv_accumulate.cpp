// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Receive accumulate kernel for fabric return.
//
// Runs on receive core (1, grid_y) as dm1 (RISCV_1).
//
// Waits for SEM_RECV to reach total_expected_remote (barrier), then
// processes all staging buffer slots: reads each scaled row from the
// staging buffer, reads the corresponding output row, element-wise
// adds, and writes back.
//
// Runtime args:
//   [0] total_expected_remote
//   [1] staging_addr          (DRAM address of recv_staging_buf)
//   [2] output_addr           (DRAM address of output buffer)
//   [3] D_bytes               (row size in bytes = D * 2)
//   [4..4+total_expected-1]   dest_token_ids (one per staging slot)
//
// Compile-time args:
//   TensorAccessorArgs<0>: staging buffer accessor
//   TensorAccessorArgs<1>: output buffer accessor
//
// Semaphores:
//   SEM_RECV (id=4): Incremented by remote return kernels via fabric.

#include "api/dataflow/dataflow_api.h"

inline void bf16_add_rows(uint32_t dst_addr, uint32_t src_addr, uint32_t n_values) {
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_addr);
    uint16_t* src = reinterpret_cast<uint16_t*>(src_addr);
    for (uint32_t i = 0; i < n_values; ++i) {
        union {
            uint32_t u;
            float f;
        } a, b, res;
        a.u = static_cast<uint32_t>(dst[i]) << 16;
        b.u = static_cast<uint32_t>(src[i]) << 16;
        res.f = a.f + b.f;
        dst[i] = static_cast<uint16_t>(res.u >> 16);
    }
}

void kernel_main() {
    const uint32_t total_expected = get_arg_val<uint32_t>(0);
    const uint32_t staging_addr = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t D_bytes = get_arg_val<uint32_t>(3);

    constexpr auto staging_args = TensorAccessorArgs<0>();
    constexpr auto output_args = TensorAccessorArgs<1>();

    constexpr uint32_t SEM_RECV = 4;
    constexpr uint32_t cb_slot = 4;
    constexpr uint32_t cb_accum = 5;

    if (total_expected == 0) {
        return;
    }

    const uint32_t D = D_bytes >> 1;

    const auto staging_acc = TensorAccessor(staging_args, staging_addr, D_bytes);
    const auto output_acc = TensorAccessor(output_args, output_addr, D_bytes);

    // Wait for all remote results to arrive
    volatile tt_l1_ptr uint32_t* recv_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_RECV));
    noc_semaphore_wait(recv_sem, total_expected);
    noc_semaphore_set(recv_sem, 0);

    // Reserve L1 buffers
    cb_reserve_back(cb_slot, 1);
    const uint32_t slot_l1 = get_write_ptr(cb_slot);
    cb_reserve_back(cb_accum, 1);
    const uint32_t accum_l1 = get_write_ptr(cb_accum);

    // Process all staging slots: accumulate into output
    for (uint32_t i = 0; i < total_expected; ++i) {
        uint32_t dest_token = get_arg_val<uint32_t>(4 + i);

        // Read staging slot i (scaled row from remote device)
        noc_async_read_page(i, staging_acc, slot_l1);
        noc_async_read_barrier();

        // Read current output[dest_token]
        noc_async_read_page(dest_token, output_acc, accum_l1);
        noc_async_read_barrier();

        // Accumulate: output[dest_token] += staged_row
        bf16_add_rows(accum_l1, slot_l1, D);

        // Write back
        noc_async_write_page(dest_token, output_acc, accum_l1);
        noc_async_write_barrier();
    }

    cb_push_back(cb_slot, 1);
    cb_pop_front(cb_slot, 1);
    cb_push_back(cb_accum, 1);
    cb_pop_front(cb_accum, 1);
}
