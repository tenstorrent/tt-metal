// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const std::uint32_t blk = get_arg(args::blk);
    const std::uint32_t NCht = get_arg(args::num_rows);
    const std::uint32_t tile_offset = get_arg(args::tile_offset);
    const std::uint32_t Wt = get_arg(args::Wt);
    // Capacity shared by the streamed CBs; a partial last pass is padded up to it.
    constexpr std::uint32_t dfb_length_t = get_arg(args::dfb_length);

    constexpr std::uint32_t dfb_id_in0 = dfb::in0;

    // ublocks size defined in tiles
    constexpr std::uint32_t onetile = 1;
    DataflowBuffer dfb_id_in0_obj(dfb_id_in0);
    std::uint32_t src0_tile_bytes = dfb_id_in0_obj.get_entry_size();

#if FUSED_SCALE_MASK
    const std::uint32_t pre_scale = get_arg(args::pre_scale);
    std::uint32_t Ht = get_arg(args::Ht);
    std::uint32_t start_ht = get_arg(args::start_ht);
    std::uint32_t start_mask_id = get_arg(args::start_mask_id);
#if CAUSAL_MASK
    std::uint32_t mask_start_ht = get_arg(args::mask_start_ht);
    std::uint32_t mask_offset = get_arg(args::mask_offset);
#endif

    constexpr std::uint32_t dfb_id_attn = dfb::fused_attn;
    DataflowBuffer dfb_id_attn_obj(dfb_id_attn);
    std::uint32_t mask_tile_bytes = dfb_id_attn_obj.get_entry_size();

    const auto addr_mask = TensorAccessor(tensor::mask);

#if CAUSAL_MASK
    constexpr std::uint32_t num_tiles_causal_mask = get_arg(args::num_tiles_causal_mask);

    std::uint32_t mask_ht = mask_start_ht;
#endif

    std::uint32_t ht = start_ht;
    std::uint32_t mask_id = start_mask_id;
    bool read_mask = true;
    constexpr auto dfb_fused_scale = dfb::fused_scale;
    generate_bcast_unary_scalar(CircularBuffer(dfb_fused_scale), pre_scale);
#endif

    const auto src_a = TensorAccessor(tensor::src);

    {
        constexpr std::uint32_t dfb_max_scaler = dfb::max_scaler;
        constexpr std::uint32_t dfb_sum_scaler = dfb::sum_scaler;
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            dfb_max_scaler,
            ckernel::PoolType::MAX,
            ckernel::ReduceDim::REDUCE_ROW>();
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            dfb_sum_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW>();
    }

    Noc noc;

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
#if NUMERIC_STABLE
    // We need an extra pass to get numeric stable
    constexpr std::uint32_t total_passes = 3;
#else
    constexpr std::uint32_t total_passes = 2;
#endif
#if FUSED_SCALE_MASK
    std::uint32_t mask_id_offset = mask_id;
    std::uint32_t mask_index = mask_id;
#endif

    for (std::uint32_t ncht = 0; ncht < NCht; ncht++) {
        // We need to pass once in order to calculate the sum and then to calculate the final value.
        for (std::uint32_t cur_pass = 0; cur_pass < total_passes; cur_pass++) {
            // We want to fill up the CB for input, and do so in chunks of blk
            std::uint32_t tile_index = tile_offset + (ncht * Wt);
#if FUSED_SCALE_MASK
            mask_index = mask_id_offset;
#endif
            for (std::uint32_t wt = 0; wt < Wt; wt += blk) {
                std::uint32_t rem = (wt + blk > Wt) ? (Wt - wt) : blk;  // clamped final block
                dfb_id_in0_obj.reserve_back(rem);
                std::uint32_t write_offset = 0;
#if FUSED_SCALE_MASK
                dfb_id_attn_obj.reserve_back(rem);
                std::uint32_t mask_write_offset = 0;
#endif
                for (std::uint32_t regs = 0; regs < rem; regs++) {
                    noc.async_read(
                        src_a,
                        dfb_id_in0_obj,
                        src0_tile_bytes,
                        {.page_id = tile_index},
                        {.offset_bytes = write_offset});
                    tile_index++;
                    write_offset += src0_tile_bytes;
#if FUSED_SCALE_MASK
                    noc.async_read(
                        addr_mask,
                        dfb_id_attn_obj,
                        mask_tile_bytes,
                        {.page_id = mask_index},
                        {.offset_bytes = mask_write_offset});
                    mask_index++;
                    mask_write_offset += mask_tile_bytes;
#endif
                }
                noc.async_read_barrier();
                dfb_id_in0_obj.push_back(rem);
#if FUSED_SCALE_MASK
                dfb_id_attn_obj.push_back(rem);

#endif
            }
            // Complete the CB cycle after a partial last Wt pass so compute can realign to fifo base.
            // Pad tiles are discarded by compute; contents are unused.
            const std::uint32_t dfb_align_pad = (dfb_length_t - (Wt % dfb_length_t)) % dfb_length_t;
            if (dfb_align_pad > 0) {
                dfb_id_in0_obj.reserve_back(dfb_align_pad);
                dfb_id_in0_obj.push_back(dfb_align_pad);
#if FUSED_SCALE_MASK
                dfb_id_attn_obj.reserve_back(dfb_align_pad);
                dfb_id_attn_obj.push_back(dfb_align_pad);
#endif
            }
        }
#if CAUSAL_MASK
        ++ht;
        ++mask_ht;
        if (ht == Ht) {
            ht = 0;
            mask_ht = 0;
            mask_id_offset += num_tiles_causal_mask;
        } else if (mask_ht == Wt) {
            mask_ht = 0;
            mask_id = mask_id_offset;
        }
#elif FUSED_SCALE_MASK
        ht++;
        if (ht != Ht) {
            mask_index = mask_id_offset;
        } else {
            ht = 0;
            mask_id_offset = mask_index;
        }

#endif
    }
}
