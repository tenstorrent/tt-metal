// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "mask_padded_tile.hpp"

#include <cstdint>

void kernel_main() {
    constexpr auto dfb_max_scaler = dfb::max_scaler;
    constexpr auto dfb_sum_scaler = dfb::sum_scaler;

#ifdef MASK_PADDED_DATA
    // The last dimension is not a multiple of the tile width: the compute adds this -inf tile to the last tile of
    // every row so the padding columns stay out of the row max and sum.
    constexpr std::uint32_t num_datum_padded = get_arg(args::num_datum_padded);
    constexpr std::uint32_t tile_hw = get_arg(args::tile_hw);
    DataflowBuffer dfb_mask_padded_obj(dfb::mask_padded);
#ifdef FUSED_SCALE_MASK
    push_mask_padded_tile<num_datum_padded, tile_hw>(dfb_mask_padded_obj);
#endif
#endif

#ifdef FUSED_SCALE_MASK
    const Noc noc;

    constexpr auto dfb_fused_scale = dfb::fused_scale;
    const std::uint32_t pre_scale = get_arg(args::pre_scale);
    generate_bcast_unary_scalar(CircularBuffer(dfb_fused_scale), pre_scale);

#ifndef SHARDED_CAUSAL_MASK
    // When the mask is interleaved (not sharded-resident) the reader streams it into dfb_attn (c_3).
    // When the mask is sharded (SHARDED_CAUSAL_MASK) c_3 is a borrowed-memory DFB the compute reads
    // directly, so the reader neither binds nor touches it.
    constexpr std::uint32_t block_wt = get_arg(args::block_w);
    const std::uint32_t mask_start_tile_id = get_arg(args::mask_start_tile_id);

    constexpr auto dfb_attn = dfb::fused_attn;
    DataflowBuffer dfb_attn_obj(dfb_attn);
    const std::uint32_t mask_tile_bytes = dfb_attn_obj.get_entry_size();
    std::uint32_t mask_id = mask_start_tile_id;

    const auto addr_mask = TensorAccessor(tensor::mask);
#endif

#if defined(CAUSAL_MASK) && !defined(SHARDED_CAUSAL_MASK)
    // The compute consumes one mask row of block_wt tiles per input row. Walk the core's rows: the mask
    // repeats every mask_Ht rows within a batch, and moves to the next batch's mask where the input does. A
    // core can start at any row of a head and its shard can span batches.
    constexpr std::uint32_t block_ht = get_arg(args::block_ht);
    constexpr std::uint32_t mask_Ht = get_arg(args::mask_Ht);
    constexpr std::uint32_t rows_per_batch = get_arg(args::rows_per_batch);
    constexpr std::uint32_t mask_batch_stride = get_arg(args::mask_batch_stride);
    std::uint32_t mask_ht = get_arg(args::mask_start_ht);
    std::uint32_t rows_left_in_batch = get_arg(args::mask_rows_left_in_batch);
    std::uint32_t mask_batch_base = mask_start_tile_id;

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    for (std::uint32_t h = 0; h < block_ht; h++) {
        mask_id = mask_batch_base + mask_ht * block_wt;
        dfb_attn_obj.reserve_back(block_wt);
        std::uint32_t write_offset = 0;
        for (std::uint32_t w = 0; w < block_wt; w++) {
            noc.async_read(
                addr_mask, dfb_attn_obj, mask_tile_bytes, {.page_id = mask_id}, {.offset_bytes = write_offset});
            write_offset += mask_tile_bytes;
            ++mask_id;
        }
        noc.async_read_barrier();
        dfb_attn_obj.push_back(block_wt);

        if (++mask_ht == mask_Ht) {
            mask_ht = 0;
        }
        if (--rows_left_in_batch == 0) {
            rows_left_in_batch = rows_per_batch;
            mask_ht = 0;
            mask_batch_base += mask_batch_stride;
        }
    }
#elif defined(CAUSAL_MASK) && defined(SHARDED_CAUSAL_MASK)
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
#elif defined(MASK_PER_ROW)
    // The shard does not sit inside one batch, so its rows need different batches' masks. Push the mask row of
    // each input row's own batch; the compute pops it after every row.
    constexpr std::uint32_t block_ht = get_arg(args::block_ht);
    constexpr std::uint32_t rows_per_batch = get_arg(args::rows_per_batch);
    constexpr std::uint32_t mask_batch_stride = get_arg(args::mask_batch_stride);
    std::uint32_t rows_left_in_batch = get_arg(args::mask_rows_left_in_batch);
    std::uint32_t mask_batch_base = mask_start_tile_id;

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    for (std::uint32_t h = 0; h < block_ht; h++) {
        mask_id = mask_batch_base;
        dfb_attn_obj.reserve_back(block_wt);
        std::uint32_t write_offset = 0;
        for (std::uint32_t w = 0; w < block_wt; w++) {
            noc.async_read(
                addr_mask, dfb_attn_obj, mask_tile_bytes, {.page_id = mask_id}, {.offset_bytes = write_offset});
            write_offset += mask_tile_bytes;
            ++mask_id;
        }
        noc.async_read_barrier();
        dfb_attn_obj.push_back(block_wt);

        if (--rows_left_in_batch == 0) {
            rows_left_in_batch = rows_per_batch;
            mask_batch_base += mask_batch_stride;
        }
    }
#else
    dfb_attn_obj.reserve_back(block_wt);
    std::uint32_t write_offset = 0;
    for (std::uint32_t w = 0; w < block_wt; w++) {
        noc.async_read(addr_mask, dfb_attn_obj, mask_tile_bytes, {.page_id = mask_id}, {.offset_bytes = write_offset});
        write_offset += mask_tile_bytes;
        ++mask_id;
    }
    noc.async_read_barrier();
    dfb_attn_obj.push_back(block_wt);

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
#endif

#else
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
#ifdef MASK_PADDED_DATA
    // The padding mask goes after the reduce scalers.
    push_mask_padded_tile<num_datum_padded, tile_hw, true>(dfb_mask_padded_obj);
#endif
#endif
}
