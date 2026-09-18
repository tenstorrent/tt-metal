// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "reshard_writer.hpp"
#include "api/tensor/noc_traits.h"
#ifdef DO_COL_MASK
#include "col_mask_dataflow.h"
#endif

void kernel_main() {
    // An idle core sits in a hole of a non-rectangular shard grid. It carries this program's dataflow
    // buffers and semaphores so the reduction's multicast has somewhere to land, and does no work of
    // its own, so its whole body is compiled out.
#ifndef IDLE_CORE

    constexpr bool is_all_to_all_worker = get_arg(args::is_all_to_all_worker) == 1;
    constexpr auto block_w = get_arg(args::block_w);

    // This core's first tile index along the width (the normalized dimension): width_index * block_w,
    // the start of this core's width shard. Used by the gamma read, the beta read, and the column mask,
    // which all index off this same per-core width offset.
    const uint32_t width_shard_tile_start_id = get_arg(args::width_shard_tile_start_id);

    // Reshard writer
#ifndef SKIP_WRITE_BACK
    constexpr auto worker_core_stride_w_bytes = get_arg(args::worker_core_stride_w_bytes);
    constexpr auto storage_core_stride_w_bytes = get_arg(args::storage_core_stride_w_bytes);
    constexpr auto block_ht = get_arg(args::block_ht);

    const uint32_t num_segments_to_write_back = get_arg(args::num_segments_to_write_back);
    const uint32_t storage_core_start_offset = get_arg(args::storage_core_start_offset);
    // One segment per storage core this block spans, three values each: the byte count and the
    // destination node's coordinates. The count varies per core, so the block is positional and the
    // local copy is sized by the longest block any core was given.
    constexpr auto max_write_back_segments = get_arg(args::max_write_back_segments);
    uint32_t segment_values[3 * max_write_back_segments];
    for (uint32_t i = 0; i < 3 * num_segments_to_write_back; ++i) {
        segment_values[i] = get_vararg(i);
    }
    tt_l1_ptr uint32_t* segment_args = (tt_l1_ptr uint32_t*)segment_values;
#endif

    Noc noc;
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
    {
        const uint32_t scalar_w_bits = get_arg(args::scalar_w);
        float scalar_w_f = __builtin_bit_cast(float, scalar_w_bits);
        dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scalar_w_f);

        const uint32_t eps = get_arg(args::eps);
        DataflowBuffer dfb_eps_obj(dfb::eps);
        generate_bcast_col_scalar(dfb_eps_obj, eps);

#ifdef DO_COL_MASK
        generate_col_mask(dfb::col_mask, block_w, get_arg(args::logical_K), width_shard_tile_start_id);
#endif

        if constexpr (is_all_to_all_worker) {
            const uint32_t scalar_c_bits = get_arg(args::scalar_c);
            float scalar_c_f = __builtin_bit_cast(float, scalar_c_bits);
            dataflow_kernel_lib::
                prepare_reduce_scaler<dfb::scaler_global, ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW>(
                    scalar_c_f);
        }
    }
#endif

#ifdef FUSE_GAMMA
    {
        const uint32_t gamma_tile_bytes = dfb_gamma_obj.get_tile_size();
        const auto gamma = TensorAccessor(tensor::gamma);

        dfb_gamma_obj.reserve_back(block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = width_shard_tile_start_id + w;
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

#ifndef SKIP_WRITE_BACK
    // The output tensor's shard sits at the same address on every core it is sharded across, so its
    // base address doubles as the destination address on each storage core. No data moves through a
    // buffer here: the writes go straight from this core's output buffer to the storage cores.
    const uint32_t output_base_addr = TensorAccessor(tensor::dst).get_bank_base_address();
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

#endif  // IDLE_CORE
}
