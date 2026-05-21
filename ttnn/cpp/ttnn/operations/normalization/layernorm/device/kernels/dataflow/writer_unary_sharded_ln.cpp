// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "reshard_writer.hpp"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr bool is_all_to_all_worker = get_arg(args::is_all_to_all_worker) == 1;
    constexpr bool fuse_gamma = get_arg(args::do_gamma) == 1;
    constexpr bool fuse_beta = get_arg(args::do_beta) == 1;
    constexpr uint32_t block_w = get_arg(args::block_wt);
    constexpr bool use_welford = get_arg(args::use_welford) == 1;

    constexpr uint32_t worker_core_stride_w_bytes = get_arg(args::block_wt_bytes);
    constexpr uint32_t storage_core_stride_w_bytes = get_arg(args::block_wt_resharded_bytes);
    constexpr uint32_t block_ht = get_arg(args::block_ht);

    const uint32_t gamma_addr = get_arg(args::gamma_addr);
    const uint32_t beta_addr = get_arg(args::beta_addr);
    const uint32_t gamma_tile_start_id = get_arg(args::gamma_tile_start);
    const uint32_t beta_tile_start_id = get_arg(args::beta_tile_start);

    // Write-back: varargs carry per-core multi-segment list
#ifndef SKIP_WRITE_BACK
    const uint32_t num_segments_to_write_back = get_vararg(1);
    const uint32_t storage_core_start_offset = get_vararg(0);
    // segments start at vararg index 2; each segment is 3 words (bytes, noc_x, noc_y)
    // The reshard_writer helper expects an `tt_l1_ptr uint32_t*` pointing at the segment array.
    // We build a stack array of varargs to satisfy that contract.
    uint32_t segment_args_buf[64];  // upper-bound, matches host num_runtime_varargs
    for (uint32_t i = 0; i < 64; ++i) {
        segment_args_buf[i] = get_vararg(2 + i);
    }
    tt_l1_ptr uint32_t* segment_args = reinterpret_cast<tt_l1_ptr uint32_t*>(segment_args_buf);
#endif

    constexpr uint32_t cb_gamma = dfb::cb_gamma;
    constexpr uint32_t cb_beta = dfb::cb_beta;
    constexpr uint32_t cb_out = dfb::cb_out;
    constexpr uint32_t cb_out_resharded = dfb::cb_out_resharded;

    Noc noc;
    DataflowBuffer cb_gamma_obj(cb_gamma);
    DataflowBuffer cb_beta_obj(cb_beta);
    DataflowBuffer cb_out_obj(cb_out);
    DataflowBuffer cb_out_resharded_obj(cb_out_resharded);

    const uint32_t out_single_tile_size_bytes = get_tile_size(cb_out);

    if constexpr (!use_welford) {
        constexpr uint32_t cb_in_2 = dfb::cb_scaler;  // host bound scaler under accessor name "cb_in_2"
        const uint32_t scalar_w_bits = get_arg(args::packed_winv);
        float scalar_w_f = __builtin_bit_cast(float, scalar_w_bits);
        dataflow_kernel_lib::prepare_reduce_scaler<cb_in_2, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scalar_w_f);

        constexpr uint32_t eps_cb_id = dfb::cb_eps;
        const uint32_t eps = get_arg(args::eps_u);
        generate_bcast_col_scalar(eps_cb_id, eps);

        if constexpr (is_all_to_all_worker) {
            constexpr uint32_t cb_in_4 = dfb::cb_scaler_global;
            const uint32_t scalar_c_bits = get_arg(args::packed_cinv);
            float scalar_c_f = __builtin_bit_cast(float, scalar_c_bits);
            dataflow_kernel_lib::prepare_reduce_scaler<
                cb_in_4,
                ckernel::PoolType::AVG,
                ckernel::ReduceDim::REDUCE_ROW,
                /*compute_uses_reduce_tile=*/true>(scalar_c_f);
        }
    }

    if constexpr (fuse_gamma) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
        const auto gamma = TensorAccessor(ta::gamma);

        cb_gamma_obj.reserve_back(block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = gamma_tile_start_id + w;
            noc.async_read(
                gamma, cb_gamma_obj, gamma_tile_bytes, {.page_id = tile_id}, {.offset_bytes = w * gamma_tile_bytes});
        }
        noc.async_read_barrier();
        cb_gamma_obj.push_back(block_w);
    }

    if constexpr (fuse_beta) {
        const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
        const auto beta = TensorAccessor(ta::beta);

        cb_beta_obj.reserve_back(block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = beta_tile_start_id + w;
            noc.async_read(
                beta, cb_beta_obj, beta_tile_bytes, {.page_id = tile_id}, {.offset_bytes = w * beta_tile_bytes});
        }
        noc.async_read_barrier();
        cb_beta_obj.push_back(block_w);
    }

#ifndef SKIP_WRITE_BACK
    write_resharded_data(
        noc,
        cb_out_obj,
        cb_out_resharded_obj,
        num_segments_to_write_back,
        storage_core_start_offset,
        segment_args,
        worker_core_stride_w_bytes,
        storage_core_stride_w_bytes,
        block_ht);
#endif
}
