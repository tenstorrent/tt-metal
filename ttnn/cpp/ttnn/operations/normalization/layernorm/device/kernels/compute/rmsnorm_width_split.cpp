// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute side of the width-split RMSNorm. Each core holds block_ht tile rows of block_w width tiles.
//
// 1. Per tile row: x^2, reduced along the row into this block's share of sum(x^2), then transposed
//    in DEST from a column into a row so the reader can hand the root two face rows instead of a
//    whole tile.
// 2. Root only: the gathered rows (one per core) are reduced column-wise into sum(x^2), scaled by
//    1/W, turned into 1/sqrt(mean + eps), and transposed in DEST back into a column for the reader
//    to multicast.
// 3. x * gamma, which does not depend on the statistics, while they are being combined.
// 4. Every core: (x * gamma) * 1/sqrt(mean + eps), then beta.

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr auto block_ht = get_arg(args::block_ht);
    constexpr auto block_w = get_arg(args::block_w);
    constexpr auto subblock_w = get_arg(args::subblock_w);
    constexpr auto gather_tiles_per_row = get_arg(args::gather_tiles_per_row);
    constexpr auto inv_width_bits = get_arg(args::inv_width);
    constexpr auto eps_bits = get_arg(args::eps);
    const bool is_root = get_arg(args::is_root) == 1;
    const uint32_t num_valid_tiles = get_arg(args::num_valid_tiles);
    constexpr uint32_t block_tiles = block_ht * block_w;
    constexpr bool fp32_dest = ckl::get_fp32_dest_acc_enabled();

    constexpr uint32_t in_id = dfb::in;
    constexpr uint32_t scaler_id = dfb::scaler;
    constexpr uint32_t scaler_one_id = dfb::scaler_one;
    constexpr uint32_t xsq_id = dfb::xsq;
    constexpr uint32_t partial_t_id = dfb::partial_t;
    constexpr uint32_t gather_id = dfb::gather;
    constexpr uint32_t rstd_id = dfb::rstd;
    constexpr uint32_t global_id = dfb::global;
    constexpr uint32_t out_id = dfb::out;
#ifdef FUSE_GAMMA
    constexpr uint32_t gamma_id = dfb::gamma;
    constexpr uint32_t scaled_id = dfb::xg;  // x * gamma
#else
    constexpr uint32_t scaled_id = in_id;
#endif
#ifdef FUSE_BETA
    constexpr uint32_t beta_id = dfb::beta;
    constexpr uint32_t normed_id = dfb::fusion;  // normalised, before beta
#else
    constexpr uint32_t normed_id = out_id;
#endif

    DataflowBuffer dfb_in(in_id);
    DataflowBuffer dfb_scaler(scaler_id);
    DataflowBuffer dfb_scaler_one(scaler_one_id);
    DataflowBuffer dfb_xsq(xsq_id);
    DataflowBuffer dfb_partial_t(partial_t_id);
    DataflowBuffer dfb_gather(gather_id);
    DataflowBuffer dfb_rstd(rstd_id);
    DataflowBuffer dfb_global(global_id);
    DataflowBuffer dfb_normed(normed_id);
#ifdef FUSE_GAMMA
    DataflowBuffer dfb_gamma(gamma_id);
    DataflowBuffer dfb_scaled(scaled_id);
#else
    DataflowBuffer& dfb_scaled = dfb_in;
#endif
#ifdef FUSE_BETA
    DataflowBuffer dfb_beta(beta_id);
    DataflowBuffer dfb_out(out_id);
#endif

    compute_kernel_hw_startup(in_id, in_id, xsq_id);
    dfb_in.wait_front(block_tiles);

    // The elementwise phases run over every tile row of the block, so each is initialised once. The
    // reduces below re-initialise per tile row: each row's transpose needs the reduce state cleared.
    mul_init(in_id, in_id);
    dfb_xsq.reserve_back(block_tiles);
    for (uint32_t t0 = 0; t0 < block_tiles; t0 += subblock_w) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < subblock_w; ++j) {
            mul_tiles(in_id, in_id, t0 + j, t0 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < subblock_w; ++j) {
            pack_tile(j, xsq_id);
        }
        tile_regs_release();
    }
    dfb_xsq.push_back(block_tiles);

    // Tiles past the width (the last block's overhang) were squared with the rest but stay out of
    // the sum.
    dfb_xsq.wait_front(block_tiles);
    dfb_scaler.wait_front(1);
    dfb_partial_t.reserve_back(block_ht);
    for (uint32_t r = 0; r < block_ht; ++r) {
        // REDUCE_ROW takes the scaler through SrcA and the data through SrcB.
        reconfig_data_format(scaler_id, xsq_id);
        pack_reconfig_data_format(partial_t_id);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(xsq_id, scaler_id, partial_t_id);
        tile_regs_acquire();
        for (uint32_t w = 0; w < num_valid_tiles; ++w) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(xsq_id, scaler_id, r * block_w + w, 0, 0);
        }
        // The reduce leaves the packer masked to its result column; clear that before packing the
        // transposed row.
        reduce_uninit();
        transpose_dest_init<fp32_dest>();
        transpose_dest<fp32_dest>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, partial_t_id);
        tile_regs_release();
    }
    dfb_partial_t.push_back(block_ht);
    dfb_xsq.pop_front(block_tiles);

    // The root combines first: everything else waits on it, while its own gamma scaling has until its
    // statistics come back through the reader.
    if (is_root) {
        // Rows past the last core are zero, so summing every gathered row of a column is exact.
        dfb_gather.wait_front(gather_tiles_per_row * block_ht);
        dfb_scaler_one.wait_front(1);
        dfb_rstd.reserve_back(block_ht);
        for (uint32_t r = 0; r < block_ht; ++r) {
            reconfig_data_format(gather_id, scaler_one_id);
            pack_reconfig_data_format(rstd_id);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(gather_id, scaler_one_id, rstd_id);
            tile_regs_acquire();
            for (uint32_t g = 0; g < gather_tiles_per_row; ++g) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(
                    gather_id, scaler_one_id, r * gather_tiles_per_row + g, 0, 0);
            }
            reduce_uninit();
            // sum(x^2) comes out along row 0 of faces 0 and 1, so 1/W, eps and rsqrt run on those two
            // faces only (VectorMode::R). 1/W is applied by the SFPU at FP32 precision rather than as
            // the reduce scaler, which the FPU takes at bfloat16 precision; the result keeps that
            // precision only when DEST is FP32. The transpose then turns the row into the column the
            // normalisation broadcasts.
            binop_with_scalar_tile_init();
            mul_unary_tile<fp32_dest>(0, inv_width_bits, VectorMode::R);
            add_unary_tile<fp32_dest>(0, eps_bits, VectorMode::R);
            rsqrt_tile_init();
            rsqrt_tile<RsqrtMode::Default, fp32_dest>(0, VectorMode::R);
            transpose_dest_init<fp32_dest>();
            transpose_dest<fp32_dest>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, rstd_id);
            tile_regs_release();
        }
        dfb_rstd.push_back(block_ht);
        dfb_gather.pop_front(gather_tiles_per_row * block_ht);
        dfb_scaler_one.pop_front(1);
    }

#ifdef FUSE_GAMMA
    dfb_gamma.wait_front(block_w);
    reconfig_data_format(in_id, gamma_id);
    pack_reconfig_data_format(scaled_id);
    mul_bcast_rows_init(in_id, gamma_id);
    dfb_scaled.reserve_back(block_tiles);
    for (uint32_t t0 = 0; t0 < block_tiles; t0 += subblock_w) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < subblock_w; ++j) {
            mul_tiles_bcast_rows(in_id, gamma_id, t0 + j, (t0 + j) % block_w, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < subblock_w; ++j) {
            pack_tile(j, scaled_id);
        }
        tile_regs_release();
    }
    dfb_scaled.push_back(block_tiles);
    dfb_in.pop_front(block_tiles);
    dfb_gamma.pop_front(block_w);
    dfb_scaled.wait_front(block_tiles);
#endif

    dfb_global.wait_front(block_ht);
    reconfig_data_format(scaled_id, global_id);
    pack_reconfig_data_format(normed_id);
    mul_bcast_cols_init(scaled_id, global_id);
    dfb_normed.reserve_back(block_tiles);
    for (uint32_t r = 0; r < block_ht; ++r) {
        for (uint32_t w0 = 0; w0 < block_w; w0 += subblock_w) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < subblock_w; ++j) {
                mul_tiles_bcast_cols(scaled_id, global_id, r * block_w + w0 + j, r, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < subblock_w; ++j) {
                pack_tile(j, normed_id);
            }
            tile_regs_release();
        }
    }
    dfb_normed.push_back(block_tiles);
    dfb_global.pop_front(block_ht);
    dfb_scaled.pop_front(block_tiles);

#ifdef FUSE_BETA
    dfb_beta.wait_front(block_w);
    dfb_normed.wait_front(block_tiles);
    reconfig_data_format(normed_id, beta_id);
    pack_reconfig_data_format(out_id);
    add_bcast_rows_init(normed_id, beta_id);
    dfb_out.reserve_back(block_tiles);
    for (uint32_t t0 = 0; t0 < block_tiles; t0 += subblock_w) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < subblock_w; ++j) {
            add_tiles_bcast_rows(normed_id, beta_id, t0 + j, (t0 + j) % block_w, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < subblock_w; ++j) {
            pack_tile(j, out_id);
        }
        tile_regs_release();
    }
    dfb_out.push_back(block_tiles);
    dfb_normed.pop_front(block_tiles);
    dfb_beta.pop_front(block_w);
#endif

    dfb_scaler.pop_front(1);
}
