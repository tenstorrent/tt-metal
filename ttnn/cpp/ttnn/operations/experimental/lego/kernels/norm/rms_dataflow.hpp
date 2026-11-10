// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RMSNorm readers.
//
// StatsReader: port of layernorm_distributed/.../reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp (no
//   FUSE_PRE_ADD): fills the SUM reduce scaler, then streams NCHt rows of Wt tiles starting at tile_offset, rows
//   row_stride tiles apart (row_stride = Wt for whole rows; the full width when a core owns one part of each row).
//   CT (A::): cb_inp cb_reduce Wt row_stride cta_src     RT: src_addr NCHt tile_offset
//
// ApplyReader: port of reader_unary_interleaved_ln_rm_gb_post_allgather.cpp (no gamma / beta): fills the AVG reduce
//   scaler (1 / reduce_factor) and the epsilon column tile, then per row reads the stats tiles (unless
//   stats_from_cb: then a compute block on this core has already produced them into cb_stats) and streams the row's
//   Wt input tiles.
//   Rows are row_stride input tiles and stats_row_stride stats tiles apart.
//   CT (A::): cb_inp cb_stats cb_reduce cb_eps Wt row_stride stats_tiles_cols stats_row_stride reduce_factor
//     stats_from_cb cta_src cta_stats
//   RT: src_addr stats_addr NCHt tile_offset stats_tile_offset eps_bits

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"

namespace lego::norm {

template <typename A>
struct StatsReader {
    uint32_t src_addr, NCHt, tile_offset;

    void init() {
        src_addr = rt<A>(0);
        NCHt = rt<A>(1);
        tile_offset = rt<A>(2);
    }

    void run() {
        const Noc noc;
        DataflowBuffer dfb_inp_buf(A::cb_inp);
        const uint32_t src0_tile_bytes = dfb_inp_buf.get_tile_size();
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<A::cb_reduce, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
        constexpr auto src_args = TensorAccessorArgs<A::cta_src>();
        const auto src_a = TensorAccessor(src_args, src_addr);
        // One barrier per row part (the CB holds Wt tiles) instead of the original's one per tile.
        for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
            uint32_t inp_tile_idx = tile_offset + ncht * A::row_stride;
            dfb_inp_buf.reserve_back(A::Wt);
            for (uint32_t wt = 0; wt < A::Wt; wt++) {
                noc.async_read(
                    src_a,
                    dfb_inp_buf,
                    src0_tile_bytes,
                    {.page_id = inp_tile_idx},
                    {.offset_bytes = wt * src0_tile_bytes});
                inp_tile_idx++;
            }
            noc.async_read_barrier();
            dfb_inp_buf.push_back(A::Wt);
        }
    }

    void teardown() {}
};

template <typename A>
struct ApplyReader {
    uint32_t src_addr, stats_addr, NCHt, tile_offset, stats_tile_offset, eps;

    void init() {
        src_addr = rt<A>(0);
        stats_addr = rt<A>(1);
        NCHt = rt<A>(2);
        tile_offset = rt<A>(3);
        stats_tile_offset = rt<A>(4);
        eps = rt<A>(5);
    }

    void run() {
        const Noc noc;
        DataflowBuffer dfb_inp_buf(A::cb_inp);
        const uint32_t src0_tile_bytes = dfb_inp_buf.get_tile_size();
        constexpr auto src_args = TensorAccessorArgs<A::cta_src>();
        const auto src_a = TensorAccessor(src_args, src_addr);

        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            A::cb_reduce,
            ckernel::PoolType::AVG,
            ckernel::ReduceDim::REDUCE_ROW,
            A::reduce_factor>();
        generate_bcast_col_scalar(CircularBuffer(A::cb_eps), eps);

        for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
            uint32_t inp_tile_idx = tile_offset + ncht * A::row_stride;
            uint32_t stats_tile_idx = stats_tile_offset + ncht * A::stats_row_stride;
            if constexpr (!A::stats_from_cb) {
                constexpr auto stats_args = TensorAccessorArgs<A::cta_stats>();
                const auto src_stats = TensorAccessor(stats_args, stats_addr);
                DataflowBuffer dfb_stats_buf(A::cb_stats);
                const uint32_t stats_tile_bytes = dfb_stats_buf.get_tile_size();
                dfb_stats_buf.reserve_back(A::stats_tiles_cols);
                uint32_t off = 0;
                for (uint32_t st = 0; st < A::stats_tiles_cols; ++st) {
                    noc.async_read(
                        src_stats, dfb_stats_buf, stats_tile_bytes, {.page_id = stats_tile_idx}, {.offset_bytes = off});
                    off += stats_tile_bytes;
                    stats_tile_idx++;
                }
                noc.async_read_barrier();
                dfb_stats_buf.push_back(A::stats_tiles_cols);
            }
            for (uint32_t wt = 0; wt < A::Wt; wt++) {
                dfb_inp_buf.reserve_back(1);
                noc.async_read(src_a, dfb_inp_buf, src0_tile_bytes, {.page_id = inp_tile_idx}, {.offset_bytes = 0});
                inp_tile_idx++;
                noc.async_read_barrier();
                dfb_inp_buf.push_back(1);
            }
        }
    }

    void teardown() {}
};

}  // namespace lego::norm
