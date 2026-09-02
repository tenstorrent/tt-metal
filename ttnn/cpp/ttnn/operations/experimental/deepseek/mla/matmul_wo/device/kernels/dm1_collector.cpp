// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
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
    constexpr auto cb_s2c_in2_id = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out_id = tt::CBIndex::c_4;

    CircularBuffer cb_s2c_in2(cb_s2c_in2_id);

    // Constants for the kernel
    constexpr uint32_t num_w_tiles_w = matmul_wo_ring::NUM_W_TILES_W;
    constexpr uint32_t num_n_tiles_per_iter = matmul_wo_ring::N_TILES_PER_ITER;
    constexpr uint32_t max_num_tiles_h = matmul_wo_ring::MAX_K_TILES_PER_CORE;
    constexpr uint32_t num_iters = num_w_tiles_w / num_n_tiles_per_iter;

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    Semaphore<> my_semaphore(reduce_semaphore_id);
    uint32_t semaphore_value = num_cores;

    for (uint32_t iter_id = 0; iter_id < num_iters; ++iter_id) {
        cb_s2c_in2.reserve_back(num_cores);

        // Wait for all 12 cores to send their data to this core
        my_semaphore.wait_min(semaphore_value);
        cb_s2c_in2.push_back(num_cores);
        semaphore_value += num_cores;
    }
}
