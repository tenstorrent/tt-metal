// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "welford_combine.h"
#include "noc_parameters.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr std::uint32_t reduce_receiver_semaphore_id =
        get_named_compile_time_arg_val("reduce_receiver_semaphore_id");
    constexpr std::uint32_t reduce_sender_semaphore_id = get_named_compile_time_arg_val("reduce_sender_semaphore_id");

    constexpr std::uint32_t num_batch_group = get_named_compile_time_arg_val("num_batch_group");
    constexpr std::uint32_t num_batches = get_named_compile_time_arg_val("num_batches");
    constexpr std::uint32_t num_groups = num_batch_group / num_batches;

    constexpr std::uint32_t per_core_N = get_named_compile_time_arg_val("per_core_N");
    const std::uint32_t per_core_N_bytes = get_named_compile_time_arg_val("per_core_N_bytes");
    const std::uint32_t per_core_N_bytes_with_stride = get_named_compile_time_arg_val("per_core_N_bytes_with_stride");
    constexpr std::uint32_t per_core_M = get_named_compile_time_arg_val("per_core_M");
    constexpr std::uint32_t tile_height = get_named_compile_time_arg_val("TILE_HEIGHT");
    constexpr std::uint32_t tile_width = get_named_compile_time_arg_val("TILE_WIDTH");

    constexpr std::uint32_t block_h = get_named_compile_time_arg_val("block_h");
    constexpr std::uint32_t block_w = get_named_compile_time_arg_val("block_w");

    constexpr std::uint32_t num_tiles_per_batch = get_named_compile_time_arg_val("num_tiles_per_batch");

    constexpr std::uint32_t num_out_blocks = get_named_compile_time_arg_val("num_out_blocks");
    // These are numbers in absolute terms, on a per batch, per group, per core basis without tiling
    constexpr std::uint32_t num_channels_per_group = get_named_compile_time_arg_val("num_channels_per_group");
    constexpr std::uint32_t num_rows_per_group = get_named_compile_time_arg_val("num_rows_per_group");

    constexpr auto src0_args = TensorAccessorArgs<0>();

    const std::uint32_t src_addr = get_arg_val<std::uint32_t>(0);
    const std::uint32_t start_id = get_arg_val<std::uint32_t>(2);
    const std::uint32_t num_channels_tiles = get_arg_val<std::uint32_t>(4);

    const std::uint32_t mcast_sender_noc_x = get_arg_val<std::uint32_t>(5);
    const std::uint32_t mcast_sender_noc_y = get_arg_val<std::uint32_t>(6);

    constexpr std::uint32_t dfb_ex_partial_id = tt::CBIndex::c_8;
    constexpr std::uint32_t dfb_ex_global_id = tt::CBIndex::c_15;
    constexpr std::uint32_t dfb_in0_id = tt::CBIndex::c_0;
    constexpr std::uint32_t dfb_repack_id = tt::CBIndex::c_26;
    constexpr std::uint32_t dfb_repack_out_id = tt::CBIndex::c_31;
    constexpr std::uint32_t dfb_out0_id = tt::CBIndex::c_16;
    // Welford-fp32 alias for dfb_in0. Shares SRAM with dfb_in0 but has its own buffer index
    // configured with UnpackToDestFp32, plus its own read/write pointers.
    // The Welford section of compute reads the alias to get full fp32 into DEST, while later
    // FPU consumers read dfb_in0 directly. When welford_fp32_alias is false, cb_in0_welford_id
    // == cb_in0_id and the gated pushes below are skipped.
    constexpr std::uint32_t dfb_in0_welford_id = get_named_compile_time_arg_val("cb_in0_welford");
    constexpr bool welford_fp32_alias = get_named_compile_time_arg_val("welford_fp32_alias") != 0;
    // When set, stats CBs hold fp32; the Welford combine reads/writes them as float not bf16.
    constexpr bool stats_is_fp32 = get_named_compile_time_arg_val("stats_is_fp32") != 0;
    constexpr bool sfpu_two_pass_l1_replay = get_named_compile_time_arg_val("sfpu_two_pass_l1_replay") != 0;

    Noc noc;
    Semaphore<> reduce_receiver_sem(reduce_receiver_semaphore_id);
    Semaphore<> reduce_sender_sem(reduce_sender_semaphore_id);
    DataflowBuffer dfb_ex_partial(dfb_ex_partial_id);
    DataflowBuffer dfb_ex_global(dfb_ex_global_id);
    DataflowBuffer dfb_in0(dfb_in0_id);
    DataflowBuffer dfb_in0_welford(dfb_in0_welford_id);
    DataflowBuffer dfb_repack(dfb_repack_id);
    DataflowBuffer dfb_repack_out(dfb_repack_out_id);
    DataflowBuffer dfb_out0(dfb_out0_id);

    constexpr std::uint32_t single_tile_size_bytes = get_tile_size(dfb_ex_partial_id);
    constexpr std::uint32_t src0_tile_bytes = get_tile_size(dfb_in0_id);

    // This is the stride between two consecutive local means/variances in the dfb_ex_partial
    constexpr std::uint32_t local_stride = 2;
    constexpr std::uint32_t single_row_size_bytes = single_tile_size_bytes / tile_height;
    constexpr std::uint32_t local_stride_per_group = local_stride * single_row_size_bytes;

    // Combine overload picked by pointer type: const float* -> fp32 combine, volatile uint16_t* -> bf16.
    using stats_read_t = std::conditional_t<stats_is_fp32, const float, volatile uint16_t>;
    using stats_write_t = std::conditional_t<stats_is_fp32, float, uint16_t>;

    const auto src_a = TensorAccessor(src0_args, src_addr);

#if defined(READER_REPACK) and defined(TILIZE_IN)
    std::uint32_t in0_l1_read_addr = dfb_in0.get_read_ptr();
    std::uint32_t src_addr_in0 = in0_l1_read_addr;
    UnicastEndpoint self_ep;
    for (std::uint32_t m = 0; m < per_core_M; ++m) {
        dfb_repack.reserve_back(per_core_N);
        std::uint32_t l1_write_addr_repack = dfb_repack.get_write_ptr();
        for (std::uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                CoreLocalMem<std::uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc.async_read_barrier();
        dfb_repack.push_back(per_core_N);
    }
#endif

    constexpr std::uint32_t out_block_h_normal = block_h / num_out_blocks;
    constexpr std::uint32_t input_cb_tiles = out_block_h_normal * block_w;
    constexpr std::uint32_t max_read_batch = input_cb_tiles < 8 ? input_cb_tiles : 8;
    std::uint32_t num_out_blocks_padded = num_out_blocks;
    std::uint32_t extra_out_block = false;
    std::uint32_t out_block_h_last = out_block_h_normal;
    if constexpr (block_h % num_out_blocks != 0) {
        extra_out_block = true;
        num_out_blocks_padded++;
        out_block_h_last = block_h % num_out_blocks;
    }

    std::uint32_t index_b_offset = 0;
    for (std::uint32_t b = 0; b < num_batches; ++b) {
        std::uint32_t mt_offset = 0;
        constexpr std::uint32_t num_stats_passes = sfpu_two_pass_l1_replay ? 1 : 2;
        for (std::uint32_t stats_pass = 0; stats_pass < num_stats_passes; ++stats_pass) {
            mt_offset = 0;
            for (std::uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                std::uint32_t out_block_h_actual;
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    out_block_h_actual = out_block_h_last;
                } else {
                    out_block_h_actual = out_block_h_normal;
                }

#if !defined(READER_REPACK) or !defined(TILIZE_IN)
                for (std::uint32_t mt = 0; mt < out_block_h_actual; ++mt) {
#include "groupnorm_read_input_row.inc"
                    mt_offset += num_channels_tiles;
                }
#endif
            }
        }

        dfb_ex_partial.wait_front(2);
        auto local_means_ptr = dfb_ex_partial.get_read_ptr();
        auto local_vars_ptr = local_means_ptr + single_tile_size_bytes;

        dfb_ex_global.reserve_back(2 * num_groups);
        auto global_means_ptr = dfb_ex_global.get_write_ptr();
        auto global_vars_ptr = global_means_ptr + single_tile_size_bytes;

        for (std::uint32_t m = 0; m < num_groups; ++m) {
            // Read mean and variance arrays from dfb_ex_partial, then combine using Welford
            auto p_local_means = reinterpret_cast<stats_read_t*>(local_means_ptr);
            auto p_local_vars = reinterpret_cast<stats_read_t*>(local_vars_ptr);

#ifdef WELFORD_SFPU_LOCAL_COMBINE
            const WelfordStats<std::remove_cv_t<stats_read_t>> local_result = {
                .mean = p_local_means[0],
                .variance = p_local_vars[0],
                .count = num_channels_per_group * num_rows_per_group,
            };
#else
            const auto local_result = combine_welford_stats<
                tile_width,
                num_channels_per_group * num_rows_per_group / tile_width,
                local_stride>(p_local_means, p_local_vars);
#endif

            // Write this to dfb_ex_global
            auto p_global_means = reinterpret_cast<volatile stats_write_t*>(global_means_ptr);
            auto p_global_vars = reinterpret_cast<volatile stats_write_t*>(global_vars_ptr);
            p_global_means[0] = local_result.mean;
            p_global_vars[0] = local_result.variance;

#ifndef GN_DISTRIBUTED_AG
            // Signal to sender that our partial data is ready
            reduce_receiver_sem.up(noc, mcast_sender_noc_x, mcast_sender_noc_y, 1);

            // Wait for sender to signal that it has sent the global data
            reduce_sender_sem.wait(VALID);
            reduce_sender_sem.set(INVALID);
#endif

            local_means_ptr += local_stride_per_group;
            local_vars_ptr += local_stride_per_group;
            global_means_ptr += 2 * single_tile_size_bytes;
            global_vars_ptr += 2 * single_tile_size_bytes;
        }

#ifdef GN_DISTRIBUTED_AG
        // Batched handshake: signal the master ONCE that all groups' partials are ready, then wait
        // ONCE for its single batched mcast-back of the GLOBAL (mean, var) for every group. The
        // master defers its mcast-back until after the cross-device fabric exchange, so the
        // per-group lock-step above would deadlock that single exchange.
        reduce_receiver_sem.up(noc, mcast_sender_noc_x, mcast_sender_noc_y, 1);
        reduce_sender_sem.wait(VALID);
        reduce_sender_sem.set(INVALID);
#endif

        dfb_ex_partial.pop_front(2);
        dfb_ex_global.push_back(2 * num_groups);

        if constexpr (!sfpu_two_pass_l1_replay) {
            mt_offset = 0;
            for (std::uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                std::uint32_t out_block_h_actual;
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    out_block_h_actual = out_block_h_last;
                } else {
                    out_block_h_actual = out_block_h_normal;
                }
#if !defined(READER_REPACK) or !defined(TILIZE_IN)
                for (std::uint32_t mt = 0; mt < out_block_h_actual; ++mt) {
#include "groupnorm_read_input_row.inc"
                    mt_offset += num_channels_tiles;
                }
#endif
            }
        }
        index_b_offset += num_tiles_per_batch;
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    std::uint32_t l1_write_addr_repack = dfb_out0.get_write_ptr();
    for (std::uint32_t m = 0; m < per_core_M; ++m) {
        dfb_repack_out.wait_front(per_core_N);
        std::uint32_t in0_l1_read_addr = dfb_repack_out.get_read_ptr();
        std::uint32_t src_addr_in0 = in0_l1_read_addr;
        UnicastEndpoint self_ep;
        for (std::uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                CoreLocalMem<std::uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc.async_read_barrier();
        dfb_repack_out.pop_front(per_core_N);
    }
#endif
}
