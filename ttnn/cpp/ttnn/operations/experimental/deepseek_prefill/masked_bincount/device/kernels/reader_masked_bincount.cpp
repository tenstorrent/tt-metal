// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
    constexpr uint32_t mask_page_size = get_compile_time_arg_val(16);

    constexpr uint32_t src_accessor_offset = 17;
    constexpr auto src_args = TensorAccessorArgs<src_accessor_offset>();
    const auto src_accessor = TensorAccessor(src_args, src_addr, input_page_size);

    constexpr uint32_t dst_accessor_offset = src_args.next_compile_time_args_offset();
    constexpr auto dst_args_ct = TensorAccessorArgs<dst_accessor_offset>();
    const auto dst_accessor = TensorAccessor(dst_args_ct, dst_addr, output_page_size);

    constexpr uint32_t mask_accessor_offset = dst_args_ct.next_compile_time_args_offset();
    constexpr auto mask_args_ct = TensorAccessorArgs<mask_accessor_offset>();
    const auto mask_accessor = TensorAccessor(mask_args_ct, mask_addr, mask_page_size);

    uint32_t in_base_addr = get_write_ptr(cb_id_in);
    uint32_t out_addr = get_write_ptr(cb_id_out);
    uint32_t mask_l1_addr = get_write_ptr(cb_mask);

    // Phase 1: Read this core's shard pages
    for (uint32_t h = 0; h < h_count; h++) {
        noc_async_read_page(h_start + h, src_accessor, in_base_addr + h * input_page_size);
    }

    // Phase 2: Local histogram counting (BRISC/NCRISC cooperate on same core)
    uint32_t init_sem_addr = get_semaphore(init_sem_idx);
    volatile tt_l1_ptr uint32_t* init_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_sem_addr);

    if constexpr (is_initializer) {
        volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);
        for (uint32_t i = 0; i < n_routed_experts; i++) {
            counts[i] = 0;
        }
        noc_async_read_page(0, mask_accessor, mask_l1_addr);
        noc_async_read_barrier();
        noc_semaphore_set(init_sem_ptr, 1);
    } else {
        noc_async_read_barrier();
        noc_semaphore_wait(init_sem_ptr, 1);
    }

    volatile tt_l1_ptr int32_t* mask = reinterpret_cast<volatile tt_l1_ptr int32_t*>(mask_l1_addr);

    for (uint32_t h = 0; h < h_count; h++) {
        volatile tt_l1_ptr uint16_t* row =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_base_addr + h * input_page_size);
        for (uint32_t w = 0; w < num_experts_per_token; w++) {
            uint32_t expert_idx = row[w];
            if (expert_idx < n_routed_experts && mask[expert_idx] >= 0) {
                uint64_t noc_addr = get_noc_addr(out_addr + expert_idx * sizeof(uint32_t));
                noc_semaphore_inc(noc_addr, 1);
            }
        }
    }
    noc_async_atomic_barrier();

    uint32_t done_sem_addr = get_semaphore(done_sem_idx);
    uint64_t done_sem_noc_addr = get_noc_addr(done_sem_addr);
    noc_semaphore_inc(done_sem_noc_addr, 1);
    noc_async_atomic_barrier();

    // Phase 3: Tree reduction — BRISC only
    if constexpr (is_initializer) {
        volatile tt_l1_ptr uint32_t* done_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem_addr);
        noc_semaphore_wait_min(done_sem_ptr, 2);

        uint32_t num_receive = get_arg_val<uint32_t>(4);
        uint32_t parent_noc_x = get_arg_val<uint32_t>(5);
        uint32_t parent_noc_y = get_arg_val<uint32_t>(6);

        uint32_t gather_sem_addr = get_semaphore(gather_sem_idx);
        volatile tt_l1_ptr uint32_t* gather_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_sem_addr);

        uint32_t tmp_addr = get_write_ptr(cb_gather_tmp);
        volatile tt_l1_ptr uint32_t* local_hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);

        for (uint32_t level = 0; level < num_receive; level++) {
            noc_semaphore_wait_min(gather_sem_ptr, level + 1);

            uint32_t child_noc_x = get_arg_val<uint32_t>(7 + level * 2);
            uint32_t child_noc_y = get_arg_val<uint32_t>(7 + level * 2 + 1);

            uint64_t child_hist_noc = get_noc_addr(child_noc_x, child_noc_y, out_addr);
            noc_async_read(child_hist_noc, tmp_addr, output_page_size);
            noc_async_read_barrier();

            volatile tt_l1_ptr uint32_t* remote_hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tmp_addr);
            for (uint32_t i = 0; i < n_routed_experts; i++) {
                local_hist[i] += remote_hist[i];
            }
        }

        if (parent_noc_x != 0xFFFFFFFF) {
            uint64_t parent_gather_noc = get_noc_addr(parent_noc_x, parent_noc_y, gather_sem_addr);
            noc_semaphore_inc(parent_gather_noc, 1);
            noc_async_atomic_barrier();
        } else {
            uint64_t dst_noc_addr = dst_accessor.get_noc_addr(0);
            noc_async_write(out_addr, dst_noc_addr, output_page_size);
            noc_async_write_barrier();
        }
    }
}
