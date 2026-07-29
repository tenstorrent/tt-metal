// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

// HW-bcast scale for fused scale-attn-softmax
FORCE_INLINE void generate_inv_sqrt_hw_bcast_tile() {
    constexpr auto dfb_fused_scale = tt::CBIndex::c_2;
    DataflowBuffer dfb_fused_scale_obj(dfb_fused_scale);
    uint32_t u = get_arg_val<uint32_t>(0);
    dfb_fused_scale_obj.reserve_back(1);
    auto ptr = reinterpret_cast<uint16_t*>(dfb_fused_scale_obj.get_write_ptr());
    ptr[0] = u >> 16;
    dfb_fused_scale_obj.push_back(1);
}

void kernel_main() {
    constexpr uint32_t dfb_max_scaler = tt::CBIndex::c_1;
    constexpr uint32_t dfb_sum_scaler = tt::CBIndex::c_13;

#if FUSED_SCALE_MASK
    constexpr uint32_t block_wt = get_compile_time_arg_val(0);
    constexpr auto mask_args = TensorAccessorArgs<1>();
    const uint32_t mask_addr = get_arg_val<uint32_t>(1);
    const uint32_t mask_start_tile_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t dfb_attn = tt::CBIndex::c_3;
    uint32_t mask_tile_bytes = get_tile_size(dfb_attn);
    uint32_t mask_id = mask_start_tile_id;

    const auto addr_mask = TensorAccessor(mask_args, mask_addr);

    Noc noc;
    DataflowBuffer dfb_attn_obj(dfb_attn);

    constexpr auto dfb_fused_scale = tt::CBIndex::c_2;
    const uint32_t pre_scale = get_arg_val<uint32_t>(0);
    generate_bcast_unary_scalar(CircularBuffer(dfb_fused_scale), pre_scale);

#if defined(CAUSAL_MASK) && !defined(SHARDED_CAUSAL_MASK)

    constexpr uint32_t fused_head = get_compile_time_arg_val(mask_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t mask_block_ht = get_compile_time_arg_val(mask_args.next_compile_time_args_offset() + 4);

    for (uint32_t f = 0; f < fused_head; f++) {
        mask_id = mask_start_tile_id;

        for (uint32_t h = 0; h < mask_block_ht; h++) {
            dfb_attn_obj.reserve_back(block_wt);
            uint32_t write_offset = 0;
            for (uint32_t w = 0; w < block_wt; w++) {
                noc.async_read(
                    addr_mask, dfb_attn_obj, mask_tile_bytes, {.page_id = mask_id}, {.offset_bytes = write_offset});
                write_offset += mask_tile_bytes;
                ++mask_id;
            }
            noc.async_read_barrier();
            dfb_attn_obj.push_back(block_wt);

            if (f == 0 && h == 0) {
                dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                    dfb_max_scaler,
                    ckernel::PoolType::MAX,
                    ckernel::ReduceDim::REDUCE_ROW>();
                dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                    dfb_sum_scaler,
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW>();
            }
        }
    }
#elif defined(CAUSAL_MASK) && defined(SHARDED_CAUSAL_MASK)
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
#else
    dfb_attn_obj.reserve_back(block_wt);
    uint32_t write_offset = 0;
    for (uint32_t w = 0; w < block_wt; w++) {
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
#endif
}
