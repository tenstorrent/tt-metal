// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pv_rescale_fusion bench writer: the op's startup_block (reduce scalers + key-tail mask tile, verbatim), then it
// drains each unit's output credit (cb_out is backed by the output L1 shard, so nothing moves).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(1);
    constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sum_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t mask_cols = get_compile_time_arg_val(4);
    const uint32_t n_units = get_arg_val<uint32_t>(0);
    const uint32_t credit = get_arg_val<uint32_t>(1);

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    if constexpr (mask_cols != 0) {
        Noc noc;
        CircularBuffer mask_cb(cb_mask);
        const uint32_t tile_bytes = get_tile_size(cb_mask);
        mask_cb.reserve_back(1);
        const uint32_t base = get_write_ptr(cb_mask);
        noc.async_write_zeros(mask_cb, tile_bytes, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        constexpr uint32_t FACE_BYTES = 16 * 16 * 2;
        constexpr uint32_t ROW_BYTES = 16 * 2;
        volatile tt_l1_ptr uint16_t* f0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
        volatile tt_l1_ptr uint16_t* f1 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + FACE_BYTES);
        for (uint32_t c = mask_cols; c < 32; ++c) {
            if (c < 16) {
                f0[c] = 0xFF80;
            } else {
                f1[c - 16] = 0xFF80;
            }
        }
        for (uint32_t r = 1; r < 16; ++r) {
            noc_async_read(get_noc_addr(base), base + r * ROW_BYTES, ROW_BYTES);
            noc_async_read(get_noc_addr(base + FACE_BYTES), base + FACE_BYTES + r * ROW_BYTES, ROW_BYTES);
        }
        noc_async_read_barrier();
        noc_async_read(get_noc_addr(base), base + 2 * FACE_BYTES, 2 * FACE_BYTES);
        noc_async_read_barrier();
        mask_cb.push_back(1);
    }
    for (uint32_t u = 0; u < n_units; ++u) {
        cb_wait_front(cb_out, credit);
        cb_pop_front(cb_out, credit);
    }
}
