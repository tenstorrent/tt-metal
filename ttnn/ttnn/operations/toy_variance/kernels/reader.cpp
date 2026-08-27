// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for toy_variance (interleaved input, single core).
//
// Variance via Var(x) = E[(x - E[x])^2] is a two-pass algorithm:
//   Pass 1: stream x  -> compute mean
//   Pass 2: stream x  -> compute (x - mean)^2 -> mean
//
// The reader streams the input tensor TWICE through dfb::in_tiles. Each pass pushes tiles in the
// order the streaming reduce expects:
//   for b in [0, num_blocks): for ht in [0, Ht):
//     for wt in [0, block_size): tile_id = ht*Wt + b*block_size + wt
//
// The scaler tile (1/N for SUM-reduce-as-mean) is pushed once at startup; reduce<> waits on it but
// never pops, so the same tile serves both passes.
//
// Metal 2.0: every name below (`dfb::`, `args::`, `tensor::`) is generated from the ProgramSpec, so
// this kernel spells no buffer index and no argument offset. The `dfb::` tokens go straight into the
// Gen1 helper library's buffer-id template parameters -- DFBBindingToken's conversion to uint32_t is
// constexpr, so no local alias is needed to bridge the two.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t BLOCK_SIZE = get_arg(args::block_size);
    constexpr uint32_t NUM_BLOCKS = get_arg(args::num_blocks);
    constexpr uint32_t scaler_bits = get_arg(args::scaler_bits);  // 1/N as fp32 bits
    constexpr bool HAS_PARTIAL_W = get_arg(args::has_partial_w) != 0;
    constexpr uint32_t partial_w = get_arg(args::partial_w);  // valid positions in last W-tile

    Noc noc;
    DataflowBuffer dfb_in(dfb::in_tiles);
    const auto acc_in = TensorAccessor(tensor::in);
    const uint32_t tile_bytes = dfb_in.get_tile_size();

    // Scaler = 1/N -> SUM reduce produces means directly. For non-tile-aligned W, also emit a
    // partial scaler tile that zeros out positions beyond partial_w; the compute kernel selects it
    // last block via ReduceScaler::with_partial().
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    if constexpr (HAS_PARTIAL_W) {
        dataflow_kernel_lib::prepare_partial_reduce_scalers<
            dfb::scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            partial_w>(scaler_f);
    } else {
        dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scaler_f);
    }

    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t wt = 0; wt < BLOCK_SIZE; ++wt) {
                    const uint32_t tile_id = ht * Wt + b * BLOCK_SIZE + wt;
                    dfb_in.reserve_back(1);
                    noc.async_read(acc_in, dfb_in, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
                    noc.async_read_barrier();
                    dfb_in.push_back(1);
                }
            }
        }
    }
}
