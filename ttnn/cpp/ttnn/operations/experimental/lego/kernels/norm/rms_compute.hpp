// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RMSNorm compute.
//
// Stats: port of rmsnorm_distributed/.../rmsnorm_pre_allgather.cpp (no FUSE_PRE_ADD): per row of Wt tiles, square
//   into cb_x2 and row-reduce with the reader's scaler into one stats tile (sum(x**2) in column 0) pushed to cb_out.
//   CT (A::): cb_inp cb_reduce cb_x2 cb_out Wt unpack_fp32_active hw_startup      RT: NCHt
//
// Apply: port of rmsnorm_distributed/.../rmsnorm_post_allgather.cpp (no gamma / beta): per row, reduce the
//   stats_tiles_cols stats tiles with the AVG scaler into var = E(x**2), add epsilon, rsqrt, and multiply the row's
//   Wt input tiles by it (column broadcast) into cb_out.
//   CT (A::): cb_inp cb_stats cb_reduce cb_eps cb_var cb_recip cb_out Wt blk stats_tiles_cols hw_startup
//   RT: NCHt

#pragma once

#include <cstdint>
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"

namespace lego::norm {

namespace ckl = compute_kernel_lib;

template <typename A>
struct Stats {
    uint32_t NCHt;
    void init() { NCHt = rt<A>(0); }

    void run() {
        constexpr bool unpack_fp32_active = A::unpack_fp32_active;
        constexpr auto reduce_type = unpack_fp32_active ? PoolType::SUM : PoolType::AVG;
        constexpr auto reduce_fp32_mode = unpack_fp32_active ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;
        DataflowBuffer dfb_inp(A::cb_inp);
        DataflowBuffer dfb_reduce(A::cb_reduce);
        DataflowBuffer dfb_x2(A::cb_x2);
        constexpr auto input_squared =
            ckl::input(A::cb_inp, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::None, ckl::InputTileMapping::Block);
        if constexpr (A::hw_startup) {
            compute_kernel_hw_startup(A::cb_inp, A::cb_reduce, A::cb_x2);
        } else {
            reconfig_data_format(A::cb_inp, A::cb_inp);
            pack_reconfig_data_format(A::cb_x2);
        }
        constexpr auto squaring_shape = ckl::IterationShape::tiles(A::Wt).block_size(1);
        for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
            if constexpr (unpack_fp32_active) {
                ckl::unary<
                    ckl::Square<>,
                    input_squared,
                    ckl::output(A::cb_x2, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                    squaring_shape, dfb_inp, dfb_x2);
            } else {
                ckl::square<
                    input_squared,
                    ckl::output(A::cb_x2, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                    squaring_shape, dfb_inp, dfb_x2);
            }
            compute_kernel_lib::reduce<
                reduce_type,
                ReduceDim::REDUCE_ROW,
                A::cb_x2,
                A::cb_reduce,
                A::cb_out,
                compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
                compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                reduce_fp32_mode>(compute_kernel_lib::ReduceInputBlockShape::row(A::Wt));
            dfb_inp.pop_front(A::Wt);
        }
        dfb_reduce.pop_front(1);
    }

    void teardown() {}
};

template <typename A>
struct Apply {
    uint32_t NCHt;
    void init() { NCHt = rt<A>(0); }

    void run() {
        CircularBuffer cb_reduce(A::cb_reduce);
        CircularBuffer cb_eps(A::cb_eps);
        DataflowBuffer cb_eps_dfb(A::cb_eps);
        DataflowBuffer cb_var(A::cb_var);
        DataflowBuffer cb_recip_sqrt_var(A::cb_recip);
        DataflowBuffer cb_norm_x_input(A::cb_inp);
        DataflowBuffer cb_out(A::cb_out);

        if constexpr (A::hw_startup) {
            compute_kernel_hw_startup(A::cb_inp, A::cb_inp, A::cb_var);
        }

        cb_reduce.wait_front(1);
        cb_eps.wait_front(1);
        for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
            compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, A::cb_stats, A::cb_reduce, A::cb_var>(
                compute_kernel_lib::ReduceInputBlockShape::row(A::stats_tiles_cols));
            // 1/sqrt(var + eps)
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Add,
                    ckl::input(A::cb_var),
                    ckl::input(A::cb_eps, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{cb_var, cb_eps_dfb},
                ckl::Rsqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(A::cb_recip)>{cb_recip_sqrt_var});
            // x * 1/sqrt(E[x**2] + eps)
            ckl::mul<
                ckl::input(
                    A::cb_inp,
                    ckl::WaitPolicy::PerBlockSize,
                    ckl::PopPolicy::PerBlockSize,
                    ckl::InputTileMapping::Block),
                ckl::input(A::cb_recip, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
                ckl::output(A::cb_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                ckl::IterationShape::tiles(A::Wt).block_size(A::blk), cb_norm_x_input, cb_recip_sqrt_var, cb_out);
        }
        cb_eps.pop_front(1);
        cb_reduce.pop_front(1);
    }

    void teardown() {}
};

}  // namespace lego::norm
