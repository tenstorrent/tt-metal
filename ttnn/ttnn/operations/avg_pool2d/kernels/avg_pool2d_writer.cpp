// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// avg_pool2d writer (block operations `fill_scaler_pair` and `store_mean_block`, op_design.md).
//
// 1. Builds the scaler pair in this cluster's L1, once: tile 0 = 1/(H*W) on all 32 position lanes, tile 1 =
//    1/(H*W) on the first last_tile_valid_rows lanes only (the reduce's partial scaler for the last position
//    tile row; rule R1). The scaler ring is explicit sync (the helper reserves, writes and pushes), opted out of
//    implicit sync on the host. It runs while the reader's first reads are in flight.
// 2. Drains `out` to output pages ct_start .. ct_start + core_ct_tiles - 1 (the output's single tile row),
//    implicit sync: one tagged write per ring entry.
//
// prepare_reduce_scaler's write_zeros is an untagged transfer: the tag registers are set to the user id first.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    const uint32_t ct_start = get_arg(args::ct_start);
    const uint32_t core_ct_tiles = get_arg(args::core_ct_tiles);
    const uint32_t inv_hw_bits = get_arg(args::inv_hw_bits);
    const uint32_t last_tile_valid_rows = get_arg(args::last_tile_valid_rows);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);
    static_assert(W == 1 && C == 1, "avg_pool2d writer implements the (1, 1, 1) thread plan");

    Noc noc;
    noc_async_write_set_trid(0, noc.get_noc_id());
    noc_async_read_set_trid(0, noc.get_noc_id());

    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer out(dfb::out);
    const auto output = TensorAccessor(tensor::output);

    const float inv_hw = __builtin_bit_cast(float, inv_hw_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(
        scaler, inv_hw, tt::constants::TILE_HEIGHT);
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(
        scaler, inv_hw, last_tile_valid_rows);

    for (uint32_t j = 0; j < core_ct_tiles; ++j) {
        noc.async_write<NocOptions::TXN_ID>(out, output, {}, {.page_id = ct_start + j});
    }
    // Hand the hart back with the user id in its tag register (see the reader).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
