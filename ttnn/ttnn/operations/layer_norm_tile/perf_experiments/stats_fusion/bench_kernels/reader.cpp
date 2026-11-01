// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// stats_fusion bench reader: z (Wt tiles) and mean (1 tile) read once; later iterations re-push the
// same pages without data (each CB holds exactly one iteration, so the ring lands on the same L1).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_z = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mean = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t iterations = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr auto z_args = TensorAccessorArgs<5>();
    constexpr auto mean_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();

    const uint32_t z_addr = get_arg_val<uint32_t>(0);
    const uint32_t mean_addr = get_arg_val<uint32_t>(1);
    const auto z_acc = TensorAccessor(z_args, z_addr, tile_bytes);
    const auto mean_acc = TensorAccessor(mean_args, mean_addr, tile_bytes);

    DataflowBuffer z(cb_z);
    DataflowBuffer mean(cb_mean);
    for (uint32_t it = 0; it < iterations; ++it) {
        z.reserve_back(Wt);
        mean.reserve_back(1);
        if (it == 0) {
            for (uint32_t t = 0; t < Wt; ++t) {
                noc_async_read(z_acc.get_noc_addr(t), z.get_write_ptr() + t * tile_bytes, tile_bytes);
            }
            noc_async_read(mean_acc.get_noc_addr(0), mean.get_write_ptr(), tile_bytes);
            noc_async_read_barrier();
        }
        z.push_back(Wt);
        mean.push_back(1);
    }
}
