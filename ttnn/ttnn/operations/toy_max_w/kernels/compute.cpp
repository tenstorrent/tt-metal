// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for toy_max_w.
//
// Streams the input W axis in NUM_BLOCKS blocks of BLOCK_SIZE tiles, computing
// the per-row max via:
//   accumulate_reduce<MAX, REDUCE_ROW>(cb_in, cb_scaler, cb_max,
//                                      block_shape, NUM_BLOCKS, partial_scaler)
// Then drains cb_max into cb_out for the writer.
//
// Partial-scaler routing: applied only to the last block, where padded
// positions in the last W-tile must not contribute to the max. The helper
// owns this routing — caller just declares the scaler as if for the whole
// reduction.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace {
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_max = 3;
constexpr uint32_t cb_out = 16;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(3);
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(4) != 0;

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::of(Ht, BLOCK_SIZE, /*NC=*/1);

    constexpr auto partial_scaler =
        HAS_PARTIAL_W ? ckl::ReducePartialScaler::last_tile_at(1) : ckl::ReducePartialScaler::none();

    // ---------- Streaming max along W ----------
    ckl::accumulate_reduce<ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>(
        cb_in, cb_scaler, cb_max, reduce_block_shape, NUM_BLOCKS, partial_scaler);

    // ---------- Drain cb_max → cb_out ----------
    ckl::copy_tiles<ckl::CopyInputPolicy::WaitAndPop>(cb_max, cb_out, Ht);

    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
