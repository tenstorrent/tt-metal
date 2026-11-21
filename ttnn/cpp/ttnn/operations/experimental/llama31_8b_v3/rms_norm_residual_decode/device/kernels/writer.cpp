// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "ttnn/operations/normalization/layernorm/device/kernels/dataflow/reshard_writer.hpp"
#include "api/tensor/noc_traits.h"
#include "internal/tensor/transfer_noc_addr.h"
#ifdef DO_COL_MASK
#include "ttnn/operations/normalization/layernorm/device/kernels/dataflow/col_mask_dataflow.h"
#endif

void kernel_main() {
#ifndef IDLE_CORE

    constexpr bool is_all_to_all_worker = get_arg(args::is_all_to_all_worker) == 1;
    constexpr auto block_w = get_arg(args::block_w);

    const uint32_t width_shard_tile_start_id = get_arg(args::width_shard_tile_start_id);

#ifndef SKIP_WRITE_BACK
    constexpr auto worker_core_stride_w_bytes = get_arg(args::worker_core_stride_w_bytes);
    constexpr auto storage_core_stride_w_bytes = get_arg(args::storage_core_stride_w_bytes);
    constexpr auto block_ht = get_arg(args::block_ht);

    const uint32_t num_segments_to_write_back = get_arg(args::num_segments_to_write_back);
    const uint32_t storage_core_start_offset = get_arg(args::storage_core_start_offset);
    constexpr auto max_write_back_segments = get_arg(args::max_write_back_segments);
    uint32_t segment_values[3 * max_write_back_segments];
    for (uint32_t i = 0; i < 3 * num_segments_to_write_back; ++i) {
        segment_values[i] = get_vararg(i);
    }
    tt_l1_ptr uint32_t* segment_args = reinterpret_cast<tt_l1_ptr uint32_t*>(segment_values);
#endif

    const Noc noc;
    constexpr uint32_t in_tile_bytes = get_arg(args::in_tile_bytes);
    const auto res_src = TensorAccessor(tensor::res_src);
    DataflowBuffer dfb_in0_obj(dfb::in0);
    DataflowBuffer dfb_in1_obj(dfb::in1);
    DataflowBuffer dfb_hout_obj(dfb::hout);
    dfb_in0_obj.reserve_back(block_w);
    dfb_in1_obj.reserve_back(block_w);
#ifndef INPUTS_IN_PLACE
    const auto y_src = TensorAccessor(tensor::y_src);
    for (uint32_t w = 0; w < block_w; w++) {
        const uint32_t tile_id = width_shard_tile_start_id + w;
        noc.async_read(y_src, dfb_in0_obj, in_tile_bytes, {.page_id = tile_id}, {.offset_bytes = w * in_tile_bytes});
        noc.async_read(res_src, dfb_in1_obj, in_tile_bytes, {.page_id = tile_id}, {.offset_bytes = w * in_tile_bytes});
    }
    noc.async_read_barrier();
#endif
    dfb_in0_obj.push_back(block_w);
    dfb_in1_obj.push_back(block_w);
#ifdef FUSE_GAMMA
    DataflowBuffer dfb_gamma_obj(dfb::gamma);
#endif
#ifdef FUSE_BETA
    DataflowBuffer dfb_beta_obj(dfb::beta);
#endif
#ifndef SKIP_WRITE_BACK
    DataflowBuffer dfb_out_obj(dfb::out);
#endif

#ifndef USE_WELFORD
    DataflowBuffer dfb_scaler_obj(dfb::scaler);
    DataflowBuffer dfb_eps_obj(dfb::eps);
    DataflowBuffer dfb_scaler_global_obj(dfb::scaler_global);
#ifdef DO_COL_MASK
    DataflowBuffer dfb_col_mask_obj(dfb::col_mask);
#endif
    {
        const uint32_t scalar_w_bits = get_arg(args::scalar_w);
        const float scalar_w_f = __builtin_bit_cast(float, scalar_w_bits);
        dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            dfb_scaler_obj, scalar_w_f);
#ifdef SCALER_SELF_LOOP
        dfb_scaler_obj.wait_front(1);
        dfb_scaler_obj.pop_front(1);
#endif

        const uint32_t eps = get_arg(args::eps);
        generate_bcast_col_scalar(dfb_eps_obj, eps);

#ifdef DO_COL_MASK
        generate_col_mask(dfb_col_mask_obj, block_w, get_arg(args::logical_K), width_shard_tile_start_id);
#endif

        if constexpr (is_all_to_all_worker) {
            const uint32_t scalar_c_bits = get_arg(args::scalar_c);
            const float scalar_c_f = __builtin_bit_cast(float, scalar_c_bits);
            dataflow_kernel_lib::
                prepare_reduce_scaler<dfb::scaler_global, ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW>(
                    dfb_scaler_global_obj, scalar_c_f);
        }
    }
#endif

#ifdef FUSE_GAMMA
    {
        const uint32_t gamma_tile_bytes = dfb_gamma_obj.get_tile_size();
        const auto gamma = TensorAccessor(tensor::gamma);

        dfb_gamma_obj.reserve_back(block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            const uint32_t tile_id = width_shard_tile_start_id + w;
            noc.async_read(
                gamma, dfb_gamma_obj, gamma_tile_bytes, {.page_id = tile_id}, {.offset_bytes = w * gamma_tile_bytes});
        }
        noc.async_read_barrier();
        dfb_gamma_obj.push_back(block_w);
    }
#endif

#ifdef FUSE_BETA
    {
        const uint32_t beta_tile_bytes = dfb_beta_obj.get_tile_size();
        const auto beta = TensorAccessor(tensor::beta);

        dfb_beta_obj.reserve_back(block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = width_shard_tile_start_id + w;
            noc.async_read(
                beta, dfb_beta_obj, beta_tile_bytes, {.page_id = tile_id}, {.offset_bytes = w * beta_tile_bytes});
        }
        noc.async_read_barrier();
        dfb_beta_obj.push_back(block_w);
    }
#endif

    dfb_hout_obj.wait_front(block_w);
    for (uint32_t w = 0; w < block_w; w++) {
        const uint32_t tile_id = width_shard_tile_start_id + w;
        noc.async_write(
            dfb_hout_obj, res_src, in_tile_bytes, {.offset_bytes = w * in_tile_bytes}, {.page_id = tile_id});
    }
    noc.async_write_barrier();
    dfb_hout_obj.pop_front(block_w);

#ifndef SKIP_WRITE_BACK
    const uint32_t output_base_addr = tensor_accessor::detail::TransferAccess::bank_base(TensorAccessor(tensor::dst));
    write_resharded_data(
        noc,
        dfb_out_obj,
        output_base_addr,
        num_segments_to_write_back,
        storage_core_start_offset,
        segment_args,
        worker_core_stride_w_bytes,
        storage_core_stride_w_bytes,
        block_ht);
#endif

#endif
}
