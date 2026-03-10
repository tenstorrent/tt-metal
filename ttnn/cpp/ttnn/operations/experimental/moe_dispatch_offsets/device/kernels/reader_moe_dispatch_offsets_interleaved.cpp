// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t core_id = get_arg_val<uint32_t>(2);
    uint32_t h_start = get_arg_val<uint32_t>(3);
    uint32_t h_count = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_partial = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr bool src_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t histogram_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t W = get_compile_time_arg_val(8);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(9);
    constexpr uint32_t input_element_size = get_compile_time_arg_val(10);
    constexpr uint32_t num_total_cores = get_compile_time_arg_val(11);
    constexpr uint32_t reduce_core_id = get_compile_time_arg_val(12);
    constexpr uint32_t reduce_core_x = get_compile_time_arg_val(13);
    constexpr uint32_t reduce_core_y = get_compile_time_arg_val(14);
    constexpr uint32_t done_sem_idx = get_compile_time_arg_val(15);

    constexpr uint32_t src_accessor_offset = 16;
    constexpr auto src_args = TensorAccessorArgs<src_accessor_offset>();
    const auto src_accessor = TensorAccessor(src_args, src_addr, input_page_size);

    constexpr uint32_t dst_args_offset = src_args.next_compile_time_args_offset();
    constexpr auto dst_args = TensorAccessorArgs<dst_args_offset>();
    const auto dst_accessor = TensorAccessor(dst_args, dst_addr, output_page_size);

    // --- Read all assigned input pages into L1 ---
    uint32_t in_base_addr = get_write_ptr(cb_id_in0);
    for (uint32_t h = 0; h < h_count; h++) {
        noc_async_read_page(h_start + h, src_accessor, in_base_addr + h * input_page_size);
    }
    noc_async_read_barrier();

    // --- Compute partial histogram ---
    uint32_t partial_base = get_write_ptr(cb_id_partial);
    uint32_t my_slot_addr = partial_base + core_id * histogram_page_size;
    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(my_slot_addr);

    for (uint32_t i = 0; i < n_routed_experts; i++) {
        counts[i] = 0;
    }

    for (uint32_t h = 0; h < h_count; h++) {
        if constexpr (input_element_size == 2) {
            volatile tt_l1_ptr uint16_t* stick =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_base_addr + h * input_page_size);
            for (uint32_t i = 0; i < W; i++) {
                uint32_t expert_idx = stick[i];
                if (expert_idx < n_routed_experts) {
                    counts[expert_idx]++;
                }
            }
        } else {
            volatile tt_l1_ptr uint32_t* stick =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_base_addr + h * input_page_size);
            for (uint32_t i = 0; i < W; i++) {
                uint32_t expert_idx = stick[i];
                if (expert_idx < n_routed_experts) {
                    counts[expert_idx]++;
                }
            }
        }
    }

    // --- Push partial histogram to reduce core's L1 ---
    if (core_id != reduce_core_id) {
        uint64_t target_noc_addr = get_noc_addr(reduce_core_x, reduce_core_y, my_slot_addr);
        noc_async_write(my_slot_addr, target_noc_addr, histogram_page_size);
        noc_async_write_barrier();
    }

    // --- Signal reduce core ---
    uint32_t done_sem_local_addr = get_semaphore(done_sem_idx);
    uint64_t done_sem_noc_addr = get_noc_addr(reduce_core_x, reduce_core_y, done_sem_local_addr);
    noc_semaphore_inc(done_sem_noc_addr, 1);
    noc_async_atomic_barrier();

    // --- Reduce core: accumulate all partials and write output ---
    if (core_id == reduce_core_id) {
        volatile tt_l1_ptr uint32_t* done_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem_local_addr);
        noc_semaphore_wait(done_sem_ptr, num_total_cores);

        uint32_t out_cb_addr = get_write_ptr(cb_id_out);
        volatile tt_l1_ptr uint32_t* total = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_cb_addr);

        for (uint32_t i = 0; i < n_routed_experts; i++) {
            total[i] = 0;
        }

        for (uint32_t c = 0; c < num_total_cores; c++) {
            volatile tt_l1_ptr uint32_t* slot =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(partial_base + c * histogram_page_size);
            for (uint32_t i = 0; i < n_routed_experts; i++) {
                total[i] += slot[i];
            }
        }

        uint64_t dst_noc_addr = dst_accessor.get_noc_addr(0);
        noc_async_write(out_cb_addr, dst_noc_addr, output_page_size);
        noc_async_write_barrier();
    }
}
