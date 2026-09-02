// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const std::uint32_t blk = get_arg(args::blk);
    const std::uint32_t num_blks = get_arg(args::num_rows);
    const std::uint32_t tile_offset = get_arg(args::tile_offset);
    const std::uint32_t Wt = get_arg(args::Wt);
    // in0's DFB capacity; pad finishes the fifo cycle between rows.
    constexpr std::uint32_t in0_t = get_arg(args::in0_t);
    // Uniform blocks tile every CB capacity, so a row that blk divides already ends on the base.
    const bool pad_to_fifo_base = Wt > 0 && blk > 0 && (Wt % blk) != 0;
    const std::uint32_t in0_pad = pad_to_fifo_base ? ((in0_t - (Wt % in0_t)) % in0_t) : 0;
    // fused_attn is sized in4_t = round_up(Wt, blk) but only Wt tiles are read per row/batch; same deal.
    const std::uint32_t attn_pad = pad_to_fifo_base ? ((blk - (Wt % blk)) % blk) : 0;

    constexpr std::uint32_t dfb_id_in0 = dfb::in0;

    // ublocks size defined in tiles
    constexpr std::uint32_t onetile = 1;
    DataflowBuffer dfb_id_in0_obj(dfb_id_in0);
    std::uint32_t src0_tile_bytes = dfb_id_in0_obj.get_entry_size();

#if FUSED_SCALE_MASK
    std::uint32_t Ht = get_arg(args::Ht);
    std::uint32_t start_ht = get_arg(args::start_ht);
    std::uint32_t start_mask_id = get_arg(args::start_mask_id);

    constexpr std::uint32_t dfb_id_attn = dfb::fused_attn;
    DataflowBuffer dfb_id_attn_obj(dfb_id_attn);
    std::uint32_t mask_tile_bytes = dfb_id_attn_obj.get_entry_size();

    const auto addr_mask = TensorAccessor(tensor::mask);

#if CAUSAL_MASK
    constexpr std::uint32_t num_tiles_causal_mask = get_arg(args::num_tiles_causal_mask);
    std::uint32_t mask_start_ht = get_arg(args::mask_start_ht);
    std::uint32_t mask_offset = get_arg(args::mask_offset);

    std::uint32_t mask_id_offset = mask_offset;
    std::uint32_t mask_ht = mask_start_ht;
#endif

    std::uint32_t ht = start_ht;
    std::uint32_t mask_id = start_mask_id;
    bool read_mask = true;
    constexpr auto dfb_fused_scale = dfb::fused_scale;
    const std::uint32_t pre_scale = get_arg(args::pre_scale);
    generate_bcast_unary_scalar(CircularBuffer(dfb_fused_scale), pre_scale);
#endif

    const auto src_a = TensorAccessor(tensor::src);

    Noc noc;

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

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    std::uint32_t i_tile = 0;
    std::uint32_t curr_tile = tile_offset;
    for (std::uint32_t i = 0; i < num_blks; ++i) {
        for (std::uint32_t j = 0; j < Wt; j += blk) {
            std::uint32_t rem = (j + blk > Wt) ? (Wt - j) : blk;  // clamped final block
            dfb_id_in0_obj.reserve_back(rem);
            std::uint32_t write_offset = 0;
            for (std::uint32_t r = 0; r < rem; ++r) {
                noc.async_read(
                    src_a, dfb_id_in0_obj, src0_tile_bytes, {.page_id = curr_tile}, {.offset_bytes = write_offset});
                curr_tile++;
                write_offset += src0_tile_bytes;
            }
            noc.async_read_barrier();
            dfb_id_in0_obj.push_back(rem);
        }
        if (in0_pad > 0) {
            dfb_id_in0_obj.reserve_back(in0_pad);
            dfb_id_in0_obj.push_back(in0_pad);
        }

#if FUSED_SCALE_MASK
// Recall that the total attention tensor size in tiles is NC,1,Wt
// For fused scale-mask softmax we write Wt attention tiles for every partHt*Wt
// of slice of tensor that was assigned to our core, then we skip to next batch
#if CAUSAL_MASK
        for (std::uint32_t j = 0; j < Wt; j += blk) {
            std::uint32_t rem = (j + blk > Wt) ? (Wt - j) : blk;  // clamped final block
            dfb_id_attn_obj.reserve_back(rem);
            std::uint32_t mask_write_offset = 0;
            for (std::uint32_t wb = 0; wb < rem; ++wb) {
                noc.async_read(
                    addr_mask,
                    dfb_id_attn_obj,
                    mask_tile_bytes,
                    {.page_id = mask_id},
                    {.offset_bytes = mask_write_offset});
                mask_write_offset += mask_tile_bytes;
                ++mask_id;
            }
            noc.async_read_barrier();
            dfb_id_attn_obj.push_back(rem);
        }
        if (attn_pad > 0) {
            dfb_id_attn_obj.reserve_back(attn_pad);
            dfb_id_attn_obj.push_back(attn_pad);
        }
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
#else
        if (read_mask) {
            for (std::uint32_t j = 0; j < Wt; j += blk) {
                // This is only executed every blk wts
                std::uint32_t rem = (j + blk > Wt) ? (Wt - j) : blk;  // clamped final block
                dfb_id_attn_obj.reserve_back(rem);
                std::uint32_t mask_write_offset = 0;
                for (std::uint32_t wb = 0; wb < rem; ++wb) {
                    noc.async_read(
                        addr_mask,
                        dfb_id_attn_obj,
                        mask_tile_bytes,
                        {.page_id = mask_id},
                        {.offset_bytes = mask_write_offset});
                    mask_write_offset += mask_tile_bytes;
                    ++mask_id;
                }
                noc.async_read_barrier();
                dfb_id_attn_obj.push_back(rem);
            }
            if (attn_pad > 0) {
                dfb_id_attn_obj.reserve_back(attn_pad);
                dfb_id_attn_obj.push_back(attn_pad);
            }
            read_mask = false;
        }
        ++ht;
        if (ht == Ht) {
            ht = 0;
            read_mask = true;
        }
#endif  // CAUSAL_MASK

#endif  // FUSED_SCALE_MASK
    }
}
