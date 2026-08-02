// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t NCHt = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t tile_offset = get_arg_val<uint32_t>(3);
    uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    uint32_t beta_addr = get_arg_val<uint32_t>(7);
    uint32_t b_addr = get_arg_val<uint32_t>(8);

    constexpr uint32_t dfb_id_in0 = get_named_compile_time_arg_val("cb_in"),
                       dfb_id_in1 = get_named_compile_time_arg_val("cb_inb");
    constexpr uint32_t dfb_id_gamma = get_named_compile_time_arg_val("cb_gamma");
    constexpr uint32_t dfb_id_beta = get_named_compile_time_arg_val("cb_beta");
    // Welford-fp32 alias of cb_in (non-fused) -- shares L1 memory with dfb_in0 but has its own
    // semaphore. The compute kernel waits on dfb_x_welford for the welford section because that
    // buffer index is configured with unpack_to_dest_mode=UnpackToDestFp32. When the alias is
    // inactive, dfb_x_welford == cb_in (non-fused) or cb_x (fused); the gate below avoids double-counting.
    constexpr uint32_t dfb_id_x_welford = get_named_compile_time_arg_val("cb_x_welford");
    constexpr bool welford_fp32_alias = get_named_compile_time_arg_val("welford_fp32_alias") != 0;

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);
    DataflowBuffer dfb_x_welford(dfb_id_x_welford);
#ifdef FUSE_PRE_ADD
    DataflowBuffer dfb_in1(dfb_id_in1);
#endif
#ifdef FUSE_GAMMA
    DataflowBuffer dfb_gamma(dfb_id_gamma);
#endif
#ifdef FUSE_BETA
    DataflowBuffer dfb_beta(dfb_id_beta);
#endif

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(dfb_id_in0);

    constexpr uint32_t blk = get_compile_time_arg_val(0);  // needed for correctness of softmax/LN kernels
    [[maybe_unused]] constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr auto src0_args = TensorAccessorArgs<2>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const auto src_a = TensorAccessor(src0_args, src_addr);
#ifdef FUSE_GAMMA
    const uint32_t gamma_tile_bytes = get_tile_size(dfb_id_gamma);
    const auto addrg = TensorAccessor(gamma_args, gamma_addr);
#endif
#ifdef FUSE_BETA
    const uint32_t beta_tile_bytes = get_tile_size(dfb_id_beta);
    const auto addrb = TensorAccessor(beta_args, beta_addr);
#endif
#ifdef FUSE_PRE_ADD
    const uint32_t src1_tile_bytes = get_tile_size(dfb_id_in1);
    const auto src_b = TensorAccessor(src1_args, b_addr);
#endif

    constexpr uint32_t eps_dfb_id = get_named_compile_time_arg_val("cb_eps");
    const uint32_t eps = get_arg_val<uint32_t>(5);
    generate_bcast_col_scalar(CircularBuffer(eps_dfb_id), eps);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t offs = 0;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // First pass
        // Calculate E[x] and Var[x]
        for (auto block : generic::blocks(Wt, blk)) {
            layernorm_dataflow_utils::read_block_to_cb(
                noc, dfb_in0, src_a, src0_tile_bytes, offs + block.start() + tile_offset, block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_cb(
                noc, dfb_in1, src_b, src1_tile_bytes, offs + block.start() + tile_offset, block);
#else
            // Non-fused welford-fp32 alias: dfb_x_welford shares dfb_in0's memory but has its own
            // semaphore. After the data lands in dfb_in0 (and therefore in shared memory), push
            // dfb_x_welford by the same amount so compute can wait_front on the alias separately
            // for welford reads. Skipped when no alias is active (dfb_x_welford == dfb_in0; the
            // duplicate push would double-count dfb_in0's semaphore).
            if constexpr (welford_fp32_alias) {
                dfb_x_welford.reserve_back(block.full_block_size());
                dfb_x_welford.push_back(block.full_block_size());
            }
#endif
        }  // wt loop

        // Second pass
        // Calculate final output
        for (auto block : generic::blocks(Wt, blk)) {
            layernorm_dataflow_utils::read_block_to_cb(
                noc, dfb_in0, src_a, src0_tile_bytes, offs + block.start() + tile_offset, block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_cb(
                noc, dfb_in1, src_b, src1_tile_bytes, offs + block.start() + tile_offset, block);
#else
            // Keep dfb_x_welford's fifo pointers in lockstep with dfb_in0's across passes.
            // dfb_x_welford and dfb_in0 share the same L1 allocation via multi-buffer-index aliasing;
            // each has its own (fifo_rd_ptr, fifo_wr_ptr, semaphore) state. dfb_in0 is pushed in
            // both passes (Wt+Wt tiles per NCHt) and popped in both (welford + eltwise). If
            // dfb_x_welford were pushed only in pass 1 and popped only in the welford section, its
            // pointers would drift relative to dfb_in0's by Wt tiles per NCHt iteration. Once the
            // CB wraps (dfb_in0 holds only Wt_next_block_up = 28 tiles for fp32 vs Wt up to 130+),
            // the welford section of the NEXT NCHt would read stale L1 data from dfb_x_welford's
            // out-of-date rd_ptr. We don't actually need the alias's data here (the eltwise pass
            // reads dfb_in0 directly), but we push the semaphore so compute can pop it in lockstep.
            if constexpr (welford_fp32_alias) {
                dfb_x_welford.reserve_back(block.full_block_size());
                dfb_x_welford.push_back(block.full_block_size());
            }
#endif
#ifdef FUSE_GAMMA
            {
                layernorm_dataflow_utils::read_block_to_cb(
                    noc, dfb_gamma, addrg, gamma_tile_bytes, block.start(), block);
            }
#endif

#ifdef FUSE_BETA
            {
                layernorm_dataflow_utils::read_block_to_cb(noc, dfb_beta, addrb, beta_tile_bytes, block.start(), block);
            }
#endif
        }  // wt loop
        offs += Wt;
    }  // ncht loop
}
