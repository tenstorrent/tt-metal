// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t batches = get_compile_time_arg_val(2);
    constexpr uint32_t output_group = get_compile_time_arg_val(3);
    constexpr bool reduce_row = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t cb_in = 0;
    using Auxiliary = ttnn::kernel_lib::ReduceAuxiliaryArgs<5>;
    constexpr auto src_args = TensorAccessorArgs<Auxiliary::next_compile_time_args_offset()>();
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();

    const auto accessor = TensorAccessor(src_args, src_addr, get_tile_size(cb_in));
    const auto read_tile = [&](uint32_t tile_id) {
        cb_reserve_back(cb_in, 1);
        noc_async_read_tile(start_id + tile_id, accessor, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    };
    if constexpr (reduce_row) {
        for (uint32_t tile = 0; tile < batches * Ht * Wt; ++tile) {
            read_tile(tile);
        }
    } else {
        // H reduction consumes one planned group of output columns at a time.
        for (uint32_t batch = 0; batch < batches; ++batch) {
            for (uint32_t col = 0; col < Wt; col += output_group) {
                const uint32_t end = col + output_group < Wt ? col + output_group : Wt;
                for (uint32_t row = 0; row < Ht; ++row) {
                    for (uint32_t wt = col; wt < end; ++wt) {
                        read_tile(batch * Ht * Wt + row * Wt + wt);
                    }
                }
            }
        }
    }
}
