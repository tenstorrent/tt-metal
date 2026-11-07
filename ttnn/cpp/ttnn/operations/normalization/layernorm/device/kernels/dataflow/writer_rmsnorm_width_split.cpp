// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the width-split RMSNorm: generates the reduce scalers, reads this core's slice of gamma
// and beta, and writes the output block back. It runs on the NoC preferred for DRAM writes, since the
// output write closes the critical path; the reader streams the input in on the other NoC.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "interleaved_block_io.hpp"

namespace {

#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
// Reads this core's slice of an affine tensor, num_valid_tiles tile columns from `start`, into a
// block_w-tile buffer; columns past the width (the last block's overhang) are left unread. A
// row-major affine tensor has one 32-wide stick per page; the stick lands in row 0 of faces 0 and 1
// of its tile, which is all a row broadcast reads.
template <typename Accessor, typename DFB>
void read_affine_block(
    const Noc& noc, const Accessor& src, DFB& dfb, uint32_t block_w, uint32_t start, uint32_t num_valid_tiles) {
    const uint32_t tile_bytes = dfb.get_tile_size();
    dfb.reserve_back(block_w);
#ifdef AFFINE_ROW_MAJOR
    const uint32_t face_bytes = tile_bytes / 4;
    const uint32_t half_stick_bytes = face_bytes / 16;
    for (uint32_t w = 0; w < num_valid_tiles; ++w) {
        noc.async_read(src, dfb, 2 * half_stick_bytes, {.page_id = start + w}, {.offset_bytes = w * tile_bytes});
    }
    noc.async_read_barrier();
    // The stick's second half sits right after the first in face 0; move it to row 0 of face 1.
    const UnicastEndpoint local_ep;
    for (uint32_t w = 0; w < num_valid_tiles; ++w) {
        noc.async_read(
            local_ep,
            dfb,
            half_stick_bytes,
            {.noc_x = my_x[noc.get_noc_id()],
             .noc_y = my_y[noc.get_noc_id()],
             .addr = dfb.get_write_ptr() + w * tile_bytes + half_stick_bytes},
            {.offset_bytes = w * tile_bytes + face_bytes});
    }
#else
    for (uint32_t w = 0; w < num_valid_tiles; ++w) {
        noc.async_read(src, dfb, tile_bytes, {.page_id = start + w}, {.offset_bytes = w * tile_bytes});
    }
#endif
    noc.async_read_barrier();
    dfb.push_back(block_w);
}
#endif

}  // namespace

void kernel_main() {
    constexpr auto block_ht = get_arg(args::block_ht);
    constexpr auto block_w = get_arg(args::block_w);
    constexpr auto Kt = get_arg(args::Kt);
    const bool is_root = get_arg(args::is_root) == 1;
    const uint32_t width_tile_start = get_arg(args::width_tile_start);
    const uint32_t num_valid_tiles = get_arg(args::num_valid_tiles);

    const Noc noc;
    DataflowBuffer dfb_scaler(dfb::scaler);
    DataflowBuffer dfb_scaler_one(dfb::scaler_one);
    DataflowBuffer dfb_out(dfb::out);

    // Compute starts on the input, which the reader streams in, and this scaler. The root's combine
    // scaler and gamma follow: they are first read once the partials are being gathered.
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        dfb_scaler, 1.0f);

    if (is_root) {
        dataflow_kernel_lib::
            prepare_reduce_scaler<dfb::scaler_one, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(
                dfb_scaler_one, 1.0f);
    }

#ifdef FUSE_GAMMA
    DataflowBuffer dfb_gamma(dfb::gamma);
    read_affine_block(noc, TensorAccessor(tensor::gamma), dfb_gamma, block_w, width_tile_start, num_valid_tiles);
#endif
#ifdef FUSE_BETA
    DataflowBuffer dfb_beta(dfb::beta);
    read_affine_block(noc, TensorAccessor(tensor::beta), dfb_beta, block_w, width_tile_start, num_valid_tiles);
#endif

    write_block(noc, TensorAccessor(tensor::dst), dfb_out, block_ht, block_w, Kt, width_tile_start, num_valid_tiles);
}
