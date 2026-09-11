// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

constexpr uint32_t reduce_call_count = get_compile_time_arg_val(0);
template <uint32_t I>
using ReduceCall =
    ttnn::kernel_lib::BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallAtT<1, I>, dfb::val, dfb::one, dfb::y, dfb::cal>;

void kernel_main() {
    const auto num_outputs = get_arg(args::num_cols_per_core);
    const auto axis_tiles = get_arg(args::Ht);
    constexpr uint32_t block_tiles = get_arg(args::reduce_block_tiles);
    constexpr uint32_t buffer_tiles = get_arg(args::reduce_buffer_tiles);
    const auto num_blocks = axis_tiles < block_tiles ? 1 : axis_tiles / block_tiles;
    DataflowBuffer input(dfb::x);
    DataflowBuffer values(dfb::val);
    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::y);

    const auto post_reduce = [](uint32_t dst) {
#ifdef MINUS_INF
        negative_tile_init();
        negative_tile(dst);
#endif
    };
    for (uint32_t output = 0; output < num_outputs; ++output) {
        for (uint32_t block = 0; block < num_blocks; ++block) {
            const auto remaining = axis_tiles - block * block_tiles;
            const auto current_tiles = block + 1 == num_blocks ? remaining : block_tiles;
            values.reserve_back(buffer_tiles);
            for (uint32_t tile = 0; tile < current_tiles; ++tile) {
                input.wait_front(1);
                tile_regs_acquire();
                reconfig_data_format_srca(dfb::x);
                copy_init(dfb::x);
                copy_tile(dfb::x, 0, 0);
#ifdef IS_ZERO
                unary_ne_tile_init();
                unary_ne_tile(0, 0);
#else
                abs_tile_init();
                abs_tile(0);
#endif
#ifdef MINUS_INF
                negative_tile_init();
                negative_tile(0);
#endif
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(dfb::val);
                pack_tile<true>(0, dfb::val, tile);
                tile_regs_release();
                input.pop_front(1);
            }
            // Keep the resident block aligned across outputs. Only the logical
            // tiles in the host call are read; unused tail entries need no data.
            values.push_back(buffer_tiles);
            if (block == 0) {
                compute_kernel_lib::reduce<ReduceCall<0>>(post_reduce);
            } else if constexpr (reduce_call_count > 1) {
                if (block + 1 == num_blocks) {
                    compute_kernel_lib::reduce<ReduceCall<reduce_call_count - 1>>(post_reduce);
                } else {
                    compute_kernel_lib::reduce<ReduceCall<1>>(post_reduce);
                }
            }
            values.pop_front(buffer_tiles);
        }
    }
    DataflowBuffer(dfb::one).pop_front(get_arg(args::reduce_auxiliary_tiles));
}
