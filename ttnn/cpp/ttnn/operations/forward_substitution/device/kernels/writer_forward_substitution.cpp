// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t matrix_page_size = get_compile_time_arg_val(0);  // C * sizeof(float)
    constexpr uint32_t matrix_size = get_compile_time_arg_val(1);       // C (square dim)
    constexpr auto output_args = TensorAccessorArgs<2>();

    // Runtime args
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t start_batch = get_arg_val<uint32_t>(1);
    uint32_t end_batch = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_work = tt::CBIndex::c_1;  // Scratch: full C×C matrix
    constexpr uint32_t cb_temp = tt::CBIndex::c_2;  // Scratch: one row temp buffer
    constexpr uint32_t onepage = 1;

    const auto dst = TensorAccessor(output_args, output_buffer_address, matrix_page_size);

    const uint32_t C = matrix_size;

    // Reserve scratch buffers once (used as raw L1 memory, not producer/consumer)
    cb_reserve_back(cb_work, onepage);
    uint32_t work_addr = get_write_ptr(cb_work);
    volatile tt_l1_ptr float* A = reinterpret_cast<volatile tt_l1_ptr float*>(work_addr);

    cb_reserve_back(cb_temp, onepage);
    uint32_t temp_addr = get_write_ptr(cb_temp);
    volatile tt_l1_ptr float* temp_row = reinterpret_cast<volatile tt_l1_ptr float*>(temp_addr);

    for (uint32_t b = start_batch; b < end_batch; b++) {
        // --- Phase 1: Receive all C rows from reader into work buffer ---
        for (uint32_t row = 0; row < C; row++) {
            cb_wait_front(cb_in, onepage);
            uint32_t src_addr = get_read_ptr(cb_in);

            // Copy row from cb_in to work buffer at correct offset
            volatile tt_l1_ptr float* src_row = reinterpret_cast<volatile tt_l1_ptr float*>(src_addr);
            volatile tt_l1_ptr float* dst_row = A + row * C;
            for (uint32_t j = 0; j < C; j++) {
                dst_row[j] = src_row[j];
            }

            cb_pop_front(cb_in, onepage);
        }

        // NOTE: This O(C^3) computation runs on the data-movement RISC-V core (BRISC/NCRISC)
        // rather than a TRISC compute core. This is intentional for v1 — for the target use
        // case (C=64, batch/core ~5), the ~435K mul-adds complete in ~4ms, which is still
        // much faster than the ~15-25ms CPU roundtrip (sync + transfer + compute + transfer)
        // this kernel replaces. A TRISC compute kernel could improve throughput for larger C.

        // --- Phase 2: Forward substitution ---
        // Computes (I + A)^{-1} row by row.
        // Equivalent to: A[i, :i] += A[i, :i] @ A[:i, :i]   for i = 1..C-1
        //
        // IMPORTANT: We compute the full updated row into temp_row before writing
        // back, to avoid read-after-write hazard. The Python reference reads
        // all of A[i, :i] atomically before updating any element.
        for (uint32_t i = 1; i < C; i++) {
            // Compute temp_row[j] = A[i,j] + sum_k(A[i,k] * A[k,j]) for j in [0, i)
            for (uint32_t j = 0; j < i; j++) {
                float dot = 0.0f;
                for (uint32_t k = 0; k < i; k++) {
                    dot += A[i * C + k] * A[k * C + j];
                }
                temp_row[j] = A[i * C + j] + dot;
            }
            // Write back the full row atomically
            for (uint32_t j = 0; j < i; j++) {
                A[i * C + j] = temp_row[j];
            }
        }

        // --- Phase 3: Add identity matrix ---
        for (uint32_t i = 0; i < C; i++) {
            A[i * C + i] += 1.0f;
        }

        // --- Phase 4: Write result rows back to DRAM ---
        // Batch all NOC writes, single barrier at end (source addrs in work buffer stay valid)
        uint32_t base_row = b * C;
        for (uint32_t row = 0; row < C; row++) {
            uint32_t row_addr = work_addr + row * C * sizeof(float);
            uint64_t noc_addr = dst.get_noc_addr(base_row + row);
            noc_async_write(row_addr, noc_addr, matrix_page_size);
        }
        noc_async_write_barrier();
    }
}
