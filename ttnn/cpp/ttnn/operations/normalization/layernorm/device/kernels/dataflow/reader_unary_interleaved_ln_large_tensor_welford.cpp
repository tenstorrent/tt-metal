// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    uint32_t NCHt = get_arg(args::NCHt);
    uint32_t Wt = get_arg(args::Wt);
    uint32_t tile_offset = get_arg(args::reader_start);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in);
    // Welford-fp32 alias of dfb_in (non-fused). It shares L1 memory with dfb_in0 but has its own
    // semaphore. The compute kernel waits on dfb_x_welford for the welford section because that
    // buffer is configured with UnpackToDest. When the alias is inactive it is not bound at all,
    // compute reads dfb_in directly, and the gate below avoids double-counting.
#if defined(WELFORD_FP32_ALIAS) && !defined(FUSE_PRE_ADD)
    DataflowBuffer dfb_x_welford(dfb::x_welford);
#endif
#ifdef FUSE_PRE_ADD
    DataflowBuffer dfb_in1(dfb::inb);
#endif
#ifdef FUSE_GAMMA
    DataflowBuffer dfb_gamma(dfb::gamma);
#endif
#ifdef FUSE_BETA
    DataflowBuffer dfb_beta(dfb::beta);
#endif

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = dfb_in0.get_tile_size();

    constexpr auto blk = get_arg(args::block_size);  // needed for correctness of softmax/LN kernels
    [[maybe_unused]] constexpr auto W = get_arg(args::W);

    const auto src_a = TensorAccessor(tensor::src);
#ifdef FUSE_GAMMA
    const uint32_t gamma_tile_bytes = dfb_gamma.get_tile_size();
    const auto addrg = TensorAccessor(tensor::gamma);
#endif
#ifdef FUSE_BETA
    const uint32_t beta_tile_bytes = dfb_beta.get_tile_size();
    const auto addrb = TensorAccessor(tensor::beta);
#endif
#ifdef FUSE_PRE_ADD
    const uint32_t src1_tile_bytes = dfb_in1.get_tile_size();
    const auto src_b = TensorAccessor(tensor::src_b);
#endif

    const uint32_t eps = get_arg(args::eps);
    DataflowBuffer dfb_eps(dfb::eps);
    generate_bcast_col_scalar(dfb_eps, eps);

    // read a ublock of tiles from src to the input buffer, and then push the ublock to unpacker
    uint32_t offs = 0;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // First pass
        // Calculate E[x] and Var[x]
        for (auto block : generic::blocks(Wt, blk)) {
            layernorm_dataflow_utils::read_block_to_dfb(
                noc, dfb_in0, src_a, src0_tile_bytes, offs + block.start() + tile_offset, block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_dfb(
                noc, dfb_in1, src_b, src1_tile_bytes, offs + block.start() + tile_offset, block);
#else
            // Non-fused welford-fp32 alias: dfb_x_welford shares dfb_in0's memory but has its own
            // semaphore. After the data lands in dfb_in0 (and therefore in shared memory), push
            // dfb_x_welford by the same amount so compute can wait_front on the alias separately
            // for welford reads. Absent when no alias is active; the duplicate push would then
            // double-count dfb_in0's semaphore.
#ifdef WELFORD_FP32_ALIAS
            dfb_x_welford.reserve_back(block.full_block_size());
            dfb_x_welford.push_back(block.full_block_size());
#endif
#endif
        }  // wt loop

        // Second pass
        // Calculate final output
        for (auto block : generic::blocks(Wt, blk)) {
            layernorm_dataflow_utils::read_block_to_dfb(
                noc, dfb_in0, src_a, src0_tile_bytes, offs + block.start() + tile_offset, block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_dfb(
                noc, dfb_in1, src_b, src1_tile_bytes, offs + block.start() + tile_offset, block);
#else
            // Keep dfb_x_welford's fifo pointers in lockstep with dfb_in0's across passes.
            // dfb_x_welford and dfb_in0 share the same L1 allocation via aliasing; each has its
            // own (fifo_rd_ptr, fifo_wr_ptr, semaphore) state. dfb_in0 is pushed in
            // both passes (Wt+Wt tiles per NCHt) and popped in both (welford + eltwise). If
            // dfb_x_welford were pushed only in pass 1 and popped only in the welford section, its
            // pointers would drift relative to dfb_in0's by Wt tiles per NCHt iteration. Once the
            // buffer wraps (dfb_in0 holds only Wt_next_block_up = 28 tiles for fp32 vs Wt up to
            // 130+), the welford section of the NEXT NCHt would read stale L1 data from
            // dfb_x_welford's out-of-date rd_ptr. We don't actually need the alias's data here
            // (the eltwise pass reads dfb_in0 directly), but we push the semaphore so compute can
            // pop it in lockstep.
#ifdef WELFORD_FP32_ALIAS
            dfb_x_welford.reserve_back(block.full_block_size());
            dfb_x_welford.push_back(block.full_block_size());
#endif
#endif
#ifdef FUSE_GAMMA
            {
                layernorm_dataflow_utils::read_block_to_dfb(
                    noc, dfb_gamma, addrg, gamma_tile_bytes, block.start(), block);
            }
#endif

#ifdef FUSE_BETA
            {
                layernorm_dataflow_utils::read_block_to_dfb(
                    noc, dfb_beta, addrb, beta_tile_bytes, block.start(), block);
            }
#endif
        }  // wt loop
        offs += Wt;
    }  // ncht loop
}
