// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(1);

    uint32_t rt_args_idx = 0;
    const uint32_t has_work = get_arg_val<uint32_t>(rt_args_idx++);
    if (has_work == 0) {
        return;
    }

    const uint32_t num_blocks = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(rt_args_idx++);

    // Hardware startup stays with the kernel; the helper owns only the summation.
    compute_kernel_hw_startup(cb_in0, cb_in0, cb_out0);

    // Sum the gathered per-device blocks resident in cb_in0 into one output block (cb_in0 is a
    // shell over the fabric-landed data and is deliberately never popped). Replaces the hand-rolled
    // pass whose DST capacity was a hardcoded `max_dst_tiles = 8` (wrong under fp32 dest-accum) and
    // whose odd-block branch was an empty "TODO: Future support" that paired blocks off the end of
    // the CB whenever num_blocks (= ring size) was odd.
    compute_kernel_lib::sum_blocks(cb_in0, cb_out0, num_blocks, block_num_tiles);
}
