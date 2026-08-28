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
// The scaler tile is owned by the compute kernel (ReduceScaler::compute_managed on an AVG reduce),
// so the reader touches only dfb::in_tiles.
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

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t BLOCK_SIZE = get_arg(args::block_size);
    constexpr uint32_t NUM_BLOCKS = get_arg(args::num_blocks);

    Noc noc;
    DataflowBuffer dfb_in(dfb::in_tiles);
    const auto acc_in = TensorAccessor(tensor::in);
    const uint32_t tile_bytes = dfb_in.get_tile_size();

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
