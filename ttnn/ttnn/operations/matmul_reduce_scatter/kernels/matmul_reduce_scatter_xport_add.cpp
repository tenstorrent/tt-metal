// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — transport-core TRISC: relay_add_block / final_add_block.
//
// cb_xport_sum = cb_xport_partial + arrival A [+ arrival B], accumulated in DEST (the transport kernels always run
// with fp32_dest_acc_en=True) and packed bf16. A relay port has (partial, A); a final core has (partial, A, B), or
// one of the two arrivals at a line end. The three-input form is one eltwise_chain: BinaryFpu add, then a
// DEST-reuse add of the third input, one pack.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_xport_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_arrival_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_arrival_b = get_compile_time_arg_val(2);
    constexpr uint32_t cb_xport_sum = get_compile_time_arg_val(3);
    constexpr uint32_t has_a = get_compile_time_arg_val(4);
    constexpr uint32_t has_b = get_compile_time_arg_val(5);
    constexpr uint32_t add_block = get_compile_time_arg_val(6);  // tiles per CB handshake / DEST batch
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);         // segments of this core x seg_tiles

    constexpr uint32_t cb_second = has_a ? cb_arrival_a : cb_arrival_b;
    constexpr auto in_cfg = [](uint32_t cb) {
        return input(cb, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block);
    };
    const auto shape = IterationShape::tiles(num_tiles).block_size(add_block);
    compute_kernel_hw_startup(cb_xport_partial, cb_second, cb_xport_sum);
    if constexpr (has_a && has_b) {
        eltwise_chain(
            shape,
            BinaryFpu<BinaryFpuOp::Add, in_cfg(cb_xport_partial), in_cfg(cb_arrival_a)>{},
            DestReuseBinary<BinaryFpuOp::Add, in_cfg(cb_arrival_b), DestReuseType::DEST_TO_SRCA>{},
            PackTile<output(cb_xport_sum, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{});
    } else {
        eltwise_chain(
            shape,
            BinaryFpu<BinaryFpuOp::Add, in_cfg(cb_xport_partial), in_cfg(cb_second)>{},
            PackTile<output(cb_xport_sum, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{});
    }
}
