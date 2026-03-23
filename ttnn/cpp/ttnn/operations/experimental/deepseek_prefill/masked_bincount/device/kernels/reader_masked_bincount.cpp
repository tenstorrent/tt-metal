// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
    constexpr uint32_t W = get_compile_time_arg_val(5);
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

    volatile tt_l1_ptr uint32_t* mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_l1_addr);

    for (uint32_t h = 0; h < h_count; h++) {
        volatile tt_l1_ptr uint16_t* row =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_base_addr + h * input_page_size);
        for (uint32_t w = 0; w < W; w++) {
            uint32_t expert_idx = row[w];
            if (expert_idx < n_routed_experts && mask[expert_idx] != 0) {
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
