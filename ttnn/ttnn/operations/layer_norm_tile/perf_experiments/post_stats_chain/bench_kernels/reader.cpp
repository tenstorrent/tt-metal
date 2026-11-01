// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// post_stats_chain bench reader: z (R * Wt fp32 tiles) and mean (R tiles) are read once; later iterations
// re-push the same pages without data (each CB holds exactly one iteration, so the ring lands on the same
// L1). gamma / beta (Wt bf16 tiles each) are pushed once and stay resident, as in the op.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_z = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mean = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(3);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(4);  // R * Wt
    constexpr uint32_t rows = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t iterations = get_compile_time_arg_val(7);
    constexpr uint32_t work_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t io_bytes = get_compile_time_arg_val(9);
    constexpr auto z_args = TensorAccessorArgs<10>();
    constexpr auto mean_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<mean_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const auto z_acc = TensorAccessor(z_args, get_arg_val<uint32_t>(0), work_bytes);
    const auto mean_acc = TensorAccessor(mean_args, get_arg_val<uint32_t>(1), work_bytes);
    const auto gamma_acc = TensorAccessor(gamma_args, get_arg_val<uint32_t>(2), io_bytes);
    const auto beta_acc = TensorAccessor(beta_args, get_arg_val<uint32_t>(3), io_bytes);

    DataflowBuffer z(cb_z), mean(cb_mean), gamma(cb_gamma), beta(cb_beta);
    gamma.reserve_back(Wt);
    beta.reserve_back(Wt);
    for (uint32_t t = 0; t < Wt; ++t) {
        noc_async_read(gamma_acc.get_noc_addr(t), gamma.get_write_ptr() + t * io_bytes, io_bytes);
        noc_async_read(beta_acc.get_noc_addr(t), beta.get_write_ptr() + t * io_bytes, io_bytes);
    }
    for (uint32_t it = 0; it < iterations; ++it) {
        z.reserve_back(row_tiles);
        mean.reserve_back(rows);
        if (it == 0) {
            for (uint32_t t = 0; t < row_tiles; ++t) {
                noc_async_read(z_acc.get_noc_addr(t), z.get_write_ptr() + t * work_bytes, work_bytes);
            }
            for (uint32_t r = 0; r < rows; ++r) {
                noc_async_read(mean_acc.get_noc_addr(r), mean.get_write_ptr() + r * work_bytes, work_bytes);
            }
            noc_async_read_barrier();
            gamma.push_back(Wt);
            beta.push_back(Wt);
        }
        z.push_back(row_tiles);
        mean.push_back(rows);
    }
}
