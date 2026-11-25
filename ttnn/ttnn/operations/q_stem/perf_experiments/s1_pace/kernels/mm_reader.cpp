// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
// s1_pace matmul bench reader (NCRISC): q_b slice (K x cols, bfp8) resident in cb_qb; x: two slots of M row tiles
// (K tiles each) filled once from DRAM, then every block re-pushes slot j % 2 without rewriting it (the L1 content
// persists), so the per-block cost measured is the compute's.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_qb = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t cols = get_compile_time_arg_val(3);
    constexpr uint32_t M = get_compile_time_arg_val(4);
    constexpr uint32_t NB = get_compile_time_arg_val(5);
    constexpr uint32_t XR = get_compile_time_arg_val(6);  // x row tiles in the tensor
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto qb_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t qb_addr = get_arg_val<uint32_t>(1);
    const uint32_t xt = get_tile_size(cb_x), qt = get_tile_size(cb_qb);
    const auto xa = TensorAccessor(x_args, x_addr, xt);
    const auto qa = TensorAccessor(qb_args, qb_addr, qt);

    cb_reserve_back(cb_qb, K * cols);
    uint32_t l1 = get_write_ptr(cb_qb);
    for (uint32_t p = 0; p < K * cols; ++p) {
        noc_async_read(qa.get_noc_addr(p), l1, qt);
        l1 += qt;
    }
    noc_async_read_barrier();
    cb_push_back(cb_qb, K * cols);

    constexpr uint32_t blk = M * K;
    for (uint32_t j = 0; j < NB; ++j) {
        cb_reserve_back(cb_x, blk);
        if (j < 2) {
            uint32_t w = get_write_ptr(cb_x);
            for (uint32_t i = 0; i < M; ++i) {
                const uint32_t r = (j * M + i) % XR;
                for (uint32_t k = 0; k < K; ++k) {
                    noc_async_read(xa.get_noc_addr(r * K + k), w, xt);
                    w += xt;
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_x, blk);
    }
}
