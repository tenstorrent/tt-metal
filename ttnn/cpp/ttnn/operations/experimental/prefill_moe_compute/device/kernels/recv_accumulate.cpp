// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Receive kernel for fabric return.
//
// Pure data movement — no floating-point math.
// Writes received rows directly to output slots (no accumulation).
//
// Runs on receive core (1, grid_y) as dm1 (RISCV_1).
//
// Output layout: [NUM_EXPERTS, 1, P, D] ROW_MAJOR
//   Page dest_page = e * P + dest_token (pre-computed by host)
//
// Runtime args:
//   [0] total_expected_remote
//   [1] staging_addr          (DRAM address of recv_staging_buf)
//   [2] output_addr           (DRAM address of output buffer)
//   [3] D_bytes               (row size in bytes = D * 2)
//   [4..4+total_expected-1]   dest_page_ids (one per staging slot)
//
// Compile-time args:
//   TensorAccessorArgs<0>: staging buffer accessor
//   TensorAccessorArgs<1>: output buffer accessor
//
// Semaphores:
//   SEM_RECV (id=4): Incremented by remote return kernels via fabric.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t total_expected = get_arg_val<uint32_t>(0);
    const uint32_t staging_addr = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t D_bytes = get_arg_val<uint32_t>(3);

    constexpr auto staging_args = TensorAccessorArgs<0>();
    constexpr auto output_args = TensorAccessorArgs<1>();

    constexpr uint32_t SEM_RECV = 4;
    constexpr uint32_t cb_slot = 4;

    if (total_expected == 0) {
        return;
    }

    const auto staging_acc = TensorAccessor(staging_args, staging_addr, D_bytes);
    const auto output_acc = TensorAccessor(output_args, output_addr, D_bytes);

    // Wait for all remote results to arrive
    volatile tt_l1_ptr uint32_t* recv_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_RECV));
    noc_semaphore_wait(recv_sem, total_expected);
    noc_semaphore_set(recv_sem, 0);

    // Reserve L1 buffer
    cb_reserve_back(cb_slot, 1);
    const uint32_t slot_l1 = get_write_ptr(cb_slot);

    // Process all staging slots: write directly to output (no accumulation)
    for (uint32_t i = 0; i < total_expected; ++i) {
        uint32_t dest_page = get_arg_val<uint32_t>(4 + i);

        // Read staging slot i (raw row from remote device)
        noc_async_read_page(i, staging_acc, slot_l1);
        noc_async_read_barrier();

        // Simple write to output[dest_page] — no read, no accumulate
        noc_async_write_page(dest_page, output_acc, slot_l1);
        noc_async_write_barrier();
    }

    cb_push_back(cb_slot, 1);
    cb_pop_front(cb_slot, 1);
}
