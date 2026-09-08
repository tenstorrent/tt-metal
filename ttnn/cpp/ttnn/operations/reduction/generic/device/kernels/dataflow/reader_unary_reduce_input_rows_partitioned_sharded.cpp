// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Height-sharded W-reduce reader (mirror of the width-sharded H reader
// reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp, but for ReduceOpDim::W).
//
// For a HEIGHT_SHARDED input the whole per-core shard is already resident in this core's L1 as
// dfb::in1 (borrowed from the input tensor).  The shard is stored row-major as
// (shard_Ht tile-rows) x (Wt tile-cols), which is exactly the (row, col) order the unified W-reduce
// compute kernel (REDUCE_ROW, WaitAndPopPerTile) consumes tiles in.  So the reader simply streams the
// resident tiles sequentially into dfb::in0 via a local (loopback) NoC read - no cross-core traffic
// and no reordering - instead of gathering every tile through the generic interleaved TensorAccessor.
void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

#ifdef REDUCE_SCALER
    constexpr auto scaler_bits = get_arg(args::scaler_bits);
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, REDUCE_OP, REDUCE_DIM>(scaler_f);
#endif

    constexpr uint32_t onetile = 1;

    Noc noc;
    // dfb::in0 is the reduce input pipe: this kernel fills it, the compute kernel drains it.
    DataflowBuffer dfb_in0(dfb::in0);
    // dfb::in1 is a view onto the resident input shard (borrowed memory). This kernel is its only
    // toucher: it reserves the whole shard and then re-reads it as the NoC source below.
    DataflowBuffer dfb_in1(dfb::in1);
    uint32_t tile_bytes = dfb_in0.get_tile_size();

    dfb_in1.reserve_back(num_tiles);
    uint32_t base_l1_addr = dfb_in1.get_write_ptr();

    UnicastEndpoint src;
    uint32_t src_noc_x = my_x[noc_index];
    uint32_t src_noc_y = my_y[noc_index];

    for (uint32_t t = 0; t < num_tiles; ++t) {
        dfb_in0.reserve_back(onetile);
        noc.async_read(
            src,
            dfb_in0,
            tile_bytes,
            {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = base_l1_addr + t * tile_bytes},
            {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0.push_back(onetile);
    }
}
