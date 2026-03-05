// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "matmul_wo_ring_common.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t reduce_semaphore_id = get_named_compile_time_arg_val("reduce_semaphore_id");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    // CBs
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;

    // Constants for the kernel
    constexpr uint32_t num_w_tiles_w = matmul_wo_ring::NUM_W_TILES_W;
    constexpr uint32_t num_n_tiles_per_iter = matmul_wo_ring::N_TILES_PER_ITER;
    constexpr uint32_t max_num_tiles_h = matmul_wo_ring::MAX_K_TILES_PER_CORE;
    constexpr uint32_t num_iters = num_w_tiles_w / num_n_tiles_per_iter;

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    uint32_t semaphore_addr = get_semaphore(reduce_semaphore_id);
    volatile tt_l1_ptr uint32_t* my_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    uint32_t semaphore_value = num_cores;

    for (uint32_t iter_id = 0; iter_id < num_iters; ++iter_id) {
        cb_reserve_back(cb_s2c_in2, num_cores);
        noc_semaphore_wait_min(my_semaphore_ptr, semaphore_value);
        cb_push_back(cb_s2c_in2, num_cores);
        semaphore_value += num_cores;
    }
}
