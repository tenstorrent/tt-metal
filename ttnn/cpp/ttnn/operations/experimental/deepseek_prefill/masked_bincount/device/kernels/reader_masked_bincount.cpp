// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Masked Bincount Kernel
//
// Counts how many tokens are routed to each expert, producing a per-expert
// histogram masked by which experts are present on this device.
//
// Inputs:
//   - input [sp_dim, topk_dim]: UINT16 height-sharded tensor of expert indices
//     selected for each token (one row per token, one column per top-k slot).
//   - expert_dispatch_table [n_routed_experts]: INT32 tensor mapping experts to
//     chip IDs. Negative (-1) means absent (skip), non-negative values (chip IDs)
//     mean present (count).
//
// Output:
//   - histogram [n_routed_experts]: UINT32 count of token assignments per expert.
//
// The same kernel source is compiled twice per core: once for BRISC
// (is_initializer = true) and once for NCRISC (is_initializer = false). They
// share a single output histogram buffer (cb_out) in L1 and cooperate through
// semaphores to parallelise the work. The kernel runs in three phases:
//
// Phase 1 — Parallel page reads:
//   Both RISCs read their assigned portion of the shard into separate
//   input CBs (cb_in_brisc / cb_in_ncrisc). The shard's rows are split roughly
//   in half: BRISC gets h_brisc rows starting at h_start, NCRISC gets h_ncrisc
//   rows starting at h_start + h_brisc. All reads are issued together and
//   overlap with phase-2 initialisation.
//
// Phase 2 — Local histogram counting:
//   BRISC (the initializer) zeroes the shared histogram buffer in cb_out,
//   fetches the expert mask into cb_mask, then signals NCRISC via init_sem.
//   NCRISC waits for init_sem before proceeding. Both RISCs then iterate their
//   input rows: for each UINT16 expert index that passes the bounds check
//   (< n_routed_experts) and the mask check (mask[expert_idx] != 0), the count
//   is incremented atomically using noc_semaphore_inc on the local L1 address.
//   This is safe because semaphore increments are atomic even when both RISCs
//   target the same word. After counting, each RISC increments done_sem and
//   waits for the atomic barrier.
//
// Phase 3 — Tree reduction (BRISC only):
//   After both RISCs on a core finish (done_sem reaches 2), BRISC participates
//   in a binary-tree reduction across cores. The tree is structured so that
//   core i receives from children at indices i + 2^L for successive levels L.
//   At each level, BRISC waits for the child's gather_sem signal, reads the
//   child's histogram from remote L1 into a temporary CB (cb_gather_tmp), and
//   element-wise adds it into the local histogram. After processing all
//   children, non-root cores signal their parent's gather_sem. The root core
//   (parent_noc_x == 0xFFFFFFFF) writes the final reduced histogram.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t mask_addr = get_arg_val<uint32_t>(2);
    uint32_t h_start = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t h_count = get_compile_time_arg_val(4);
    constexpr uint32_t num_experts_per_token = get_compile_time_arg_val(5);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(6);
    constexpr bool is_initializer = (bool)get_compile_time_arg_val(7);
    constexpr uint32_t init_sem_idx = get_compile_time_arg_val(8);
    constexpr uint32_t done_sem_idx = get_compile_time_arg_val(9);
    constexpr uint32_t gather_sem_idx = get_compile_time_arg_val(10);
    constexpr uint32_t cb_gather_tmp = get_compile_time_arg_val(11);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(15);

    constexpr uint32_t src_accessor_offset = 17;
    constexpr auto src_args = TensorAccessorArgs<src_accessor_offset>();
    const auto src_accessor = TensorAccessor(src_args, src_addr);

    constexpr uint32_t dst_accessor_offset = src_args.next_compile_time_args_offset();
    constexpr auto dst_args_ct = TensorAccessorArgs<dst_accessor_offset>();
    const auto dst_accessor = TensorAccessor(dst_args_ct, dst_addr);

    constexpr uint32_t mask_accessor_offset = dst_args_ct.next_compile_time_args_offset();
    constexpr auto mask_args_ct = TensorAccessorArgs<mask_accessor_offset>();
    const auto mask_accessor = TensorAccessor(mask_args_ct, mask_addr);

    uint32_t in_base_addr = get_write_ptr(cb_id_in);
    // BRISC counts into cb_out, NCRISC into cb_gather_tmp (each RISC its own histogram: plain L1 increments, no
    // atomics); BRISC then adds NCRISC's into cb_out, the core's published histogram
    uint32_t out_addr = get_write_ptr(is_initializer ? cb_id_out : cb_gather_tmp);
    uint32_t mask_l1_addr = get_write_ptr(cb_mask);

    // Phase 1: Read this core's shard pages
    for (uint32_t h = 0; h < h_count; h++) {
        noc_async_read_page(h_start + h, src_accessor, in_base_addr + h * input_page_size);
    }

    // Phase 2: local counting. BRISC fetches the mask and signals NCRISC (init_sem); each RISC zeroes and fills
    // its own histogram.
    uint32_t init_sem_addr = get_semaphore(init_sem_idx);
    volatile tt_l1_ptr uint32_t* init_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_sem_addr);
    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);
    for (uint32_t i = 0; i < n_routed_experts; i++) {
        counts[i] = 0;
    }
    if constexpr (is_initializer) {
        noc_async_read_page(0, mask_accessor, mask_l1_addr);
        noc_async_read_barrier();
        noc_semaphore_set(init_sem_ptr, 1);
    } else {
        noc_async_read_barrier();
        noc_semaphore_wait(init_sem_ptr, 1);
    }
    invalidate_l1_cache();

    volatile tt_l1_ptr int32_t* mask = reinterpret_cast<volatile tt_l1_ptr int32_t*>(mask_l1_addr);
    for (uint32_t h = 0; h < h_count; h++) {
        volatile tt_l1_ptr uint16_t* row =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_base_addr + h * input_page_size);
        for (uint32_t w = 0; w < num_experts_per_token; w++) {
            uint32_t expert_idx = row[w];
            if (expert_idx < n_routed_experts && mask[expert_idx] >= 0) {
                counts[expert_idx] += 1;
            }
        }
    }
    asm volatile("fence" ::: "memory");

    uint32_t done_sem_addr = get_semaphore(done_sem_idx);
    volatile tt_l1_ptr uint32_t* done_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem_addr);
    if constexpr (!is_initializer) {
        noc_semaphore_inc(get_noc_addr(done_sem_addr), 1);
        noc_async_atomic_barrier();
        return;
    }

    // Phase 3 (BRISC): fold NCRISC's histogram in, publish to every reducer core, then (reducers only) sum the
    // reducer's 16-expert column of every core's histogram and write it to the output.
    noc_semaphore_wait_min(done_sem_ptr, 1);
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* other =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_gather_tmp));
    for (uint32_t i = 0; i < n_routed_experts; i++) {
        counts[i] += other[i];
    }
    asm volatile("fence" ::: "memory");

    const uint32_t num_cores_rt = get_arg_val<uint32_t>(4);
    const uint32_t num_reducers = get_arg_val<uint32_t>(5);
    const uint32_t my_reducer = get_arg_val<uint32_t>(6);  // 0xFFFFFFFF: not a reducer
    constexpr uint32_t per_reducer = 16;                  // experts per reducer (64 bytes)
    const uint32_t gather_sem_addr = get_semaphore(gather_sem_idx);
    // core c's NoC xy at args 7 + 2c
    for (uint32_t r = 0; r < num_reducers; r++) {
        const uint32_t x = get_arg_val<uint32_t>(7 + 2 * r), y = get_arg_val<uint32_t>(7 + 2 * r + 1);
        noc_semaphore_inc(get_noc_addr(x, y, gather_sem_addr), 1);
    }
    noc_async_atomic_barrier();
    if (my_reducer == 0xFFFFFFFF) {
        return;
    }
    volatile tt_l1_ptr uint32_t* gather_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_sem_addr);
    noc_semaphore_wait_min(gather_sem_ptr, num_cores_rt);
    noc_semaphore_set(gather_sem_ptr, 0);
    // stage every core's 64-byte column in c_5 (num_cores x 64 bytes, reducer staging)
    const uint32_t stage = get_write_ptr(tt::CBIndex::c_5);
    const uint32_t col_off = my_reducer * per_reducer * sizeof(uint32_t);
    for (uint32_t c = 0; c < num_cores_rt; c++) {
        const uint32_t x = get_arg_val<uint32_t>(7 + 2 * c), y = get_arg_val<uint32_t>(7 + 2 * c + 1);
        noc_async_read(get_noc_addr(x, y, out_addr + col_off), stage + c * 64, 64);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t sum[per_reducer];
    for (uint32_t e = 0; e < per_reducer; e++) {
        sum[e] = 0;
    }
    for (uint32_t c = 0; c < num_cores_rt; c++) {
        volatile tt_l1_ptr uint32_t* col = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + c * 64);
        for (uint32_t e = 0; e < per_reducer; e++) {
            sum[e] += col[e];
        }
    }
    // stage the 16 sums in cb_gather_tmp (NCRISC's histogram, already folded in) and write them to the output
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_gather_tmp));
    for (uint32_t e = 0; e < per_reducer; e++) {
        res[e] = sum[e];
    }
    noc_async_write(get_write_ptr(cb_gather_tmp), dst_accessor.get_noc_addr(0) + col_off, 64);
    noc_async_write_barrier();
}
