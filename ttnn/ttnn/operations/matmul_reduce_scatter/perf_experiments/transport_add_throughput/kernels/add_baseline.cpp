// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — transport-core TRISC: relay_add_block / final_add_block.
//
// cb_xport_sum = cb_xport_partial + arrival A [+ arrival B], accumulated in DEST (the transport kernels always run
// with fp32_dest_acc_en=True) and packed bf16. A relay port has (partial, A); a final core has (partial, A, B), or
// one of the two arrivals at a line end. The three-input form is one eltwise_chain: BinaryFpu add, then a
// DEST-reuse add of the third input, one pack.
// A ring port's list starts with entries that have no upstream (the chip's own partial starts the block's chain):
// their tiles are copied through partial -> sum first (cb_xport_sum keeps exactly one producer), then the relay
// entries are added.
// The walk is a grid of (segments x seg_tiles), blocked per row: every segment (the sender's unit) is completed on
// its own (ragged last block of the row), never held back for tiles of the next segment -- a cross-segment
// lookahead chains into the upstream chip and can close a wait cycle on short block lists. Segments never straddle
// the CB wrap (capacity is a multiple of seg_tiles), so neither does any block.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Stage zones (permanent; opt-in via KERNEL_PERF_ZONES): xadd_copy / xadd_add span the whole copy-through / add walk
// (the helper waits per block on the reader and reserves on the sender / writer, so occupancy, not payload).

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_xport_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_arrival_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_arrival_b = get_compile_time_arg_val(2);
    constexpr uint32_t cb_xport_sum = get_compile_time_arg_val(3);
    constexpr uint32_t has_a = get_compile_time_arg_val(4);
    constexpr uint32_t has_b = get_compile_time_arg_val(5);
    constexpr uint32_t add_block = get_compile_time_arg_val(6);  // tiles per CB handshake / DEST batch (<= seg_tiles)
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(7);
    const uint32_t num_copy_segs = get_arg_val<uint32_t>(0);  // upstream-less entries: copied through
    const uint32_t num_segs = get_arg_val<uint32_t>(1);       // relay entries (or the finals' block): added

    constexpr uint32_t cb_second = has_a ? cb_arrival_a : cb_arrival_b;
    constexpr auto in_cfg = [](uint32_t cb) {
        return input(cb, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block);
    };
    const auto shape = IterationShape::grid(num_segs, seg_tiles).block_size(add_block);
    compute_kernel_hw_startup(cb_xport_partial, cb_second, cb_xport_sum);
    if (num_copy_segs > 0) {
        MaybeDeviceZoneScope("xadd_copy");
        eltwise_chain(
            IterationShape::grid(num_copy_segs, seg_tiles).block_size(add_block),
            CopyTile<in_cfg(cb_xport_partial)>{},
            PackTile<output(cb_xport_sum, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{});
    }
    if (num_segs == 0) {
        return;
    }
    MaybeDeviceZoneScope("xadd_add");
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
