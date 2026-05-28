// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader dataflow kernel for the generic fused all_gather_rms_norm op.
//
// Generates the AVG reduce scalar (1/reduce_factor) and the epsilon constant, reads gamma/beta once
// (TILE layout), and streams the input row tiles into cb_input. The single compute kernel reuses the
// resident input row for both x^2 and the normalize step.
//
// Ported from the generic distributed reader:
//   layernorm_distributed/device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_{pre,post}_allgather.cpp
//
// STATUS: complete for the ring_size == 1 (single-device) path. Residual (FUSE_PRE_ADD) and the
// gathered-stats read (ring_size > 1) are not handled yet.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // Compile-time args (order must match all_gather_rms_norm_program_factory.cpp: reader_ct_args).
    constexpr uint32_t cb_inp = get_compile_time_arg_val(0);
    constexpr uint32_t cb_reduce = get_compile_time_arg_val(1);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(3);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t blk = get_compile_time_arg_val(6);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(7);
    constexpr uint32_t has_beta = get_compile_time_arg_val(8);
    constexpr uint32_t reduce_factor = get_compile_time_arg_val(9);

    constexpr auto src_args = TensorAccessorArgs<10>();
#if FUSE_GAMMA
    constexpr auto gamma_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
#endif
#if FUSE_BETA
#if FUSE_GAMMA
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
#else
    constexpr auto beta_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
#endif
#endif

    // Runtime args.
    uint32_t ai = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t NCHt = get_arg_val<uint32_t>(ai++);
    const uint32_t tile_offset = get_arg_val<uint32_t>(ai++);
    const uint32_t eps_packed = get_arg_val<uint32_t>(ai++);
#if FUSE_GAMMA
    const uint32_t gamma_addr = get_arg_val<uint32_t>(ai++);
#endif
#if FUSE_BETA
    const uint32_t beta_addr = get_arg_val<uint32_t>(ai++);
#endif

    const uint32_t src_tile_bytes = get_tile_size(cb_inp);

    // Reduce scalar (AVG over reduce_factor) + epsilon broadcast tile.
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_reduce,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        reduce_factor,
        /*compute_uses_reduce_tile=*/true>();
    generate_bcast_col_scalar(cb_eps, eps_packed);

    const auto src_a = TensorAccessor(src_args, src_addr);
    Noc noc;
    CircularBuffer cb_inp_buf(cb_inp);

    // Read gamma / beta once (TILE layout: each is a row of Wt tiles, reused across all NCHt rows).
#if FUSE_GAMMA
    {
        const auto src_gamma = TensorAccessor(gamma_args, gamma_addr);
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
        CircularBuffer cb_gamma_buf(cb_gamma);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            cb_gamma_buf.reserve_back(1);
            noc.async_read(src_gamma, cb_gamma_buf, gamma_tile_bytes, {.page_id = wt}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_gamma_buf.push_back(1);
        }
    }
#endif
#if FUSE_BETA
    {
        const auto src_beta = TensorAccessor(beta_args, beta_addr);
        const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
        CircularBuffer cb_beta_buf(cb_beta);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            cb_beta_buf.reserve_back(1);
            noc.async_read(src_beta, cb_beta_buf, beta_tile_bytes, {.page_id = wt}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_beta_buf.push_back(1);
        }
    }
#endif

    // Stream the input rows assigned to this worker.
    uint32_t inp_tile_idx = tile_offset;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            for (uint32_t r = 0; r < blk; r++) {
                cb_inp_buf.reserve_back(1);
                noc.async_read(src_a, cb_inp_buf, src_tile_bytes, {.page_id = inp_tile_idx}, {.offset_bytes = 0});
                inp_tile_idx++;
                noc.async_read_barrier();
                cb_inp_buf.push_back(1);
            }
        }
    }
}
