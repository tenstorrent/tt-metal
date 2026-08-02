// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Mcast-group RECEIVER reader for the fused distributed GroupNorm. Derived verbatim from the stock
// welford_reader_mcast_receiver_unary_gn.cpp, with two differences: (1) positional compile-time args
// (the fused Program path uses CreateKernel + positional args, not named args); (2) a BATCHED
// intra-device handshake — the receiver combines every group's local partial, signals the master
// ONCE, then waits ONCE for the master's single batched mcast-back of the GLOBAL stat. This matches
// the master reader, which defers its mcast-back until after the cross-device fabric exchange; the
// stock per-group signal/wait lock-step would deadlock that single exchange. The arithmetic is
// identical to stock.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_combine.h"
#include "noc_parameters.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // Scalars [0..18] mirror the master reader's welford GN args (dit_gn_welford_reader.cpp); the
    // src0 TensorAccessor block starts at index 19. Indices the receiver does not use (e.g. 2,
    // num_cores_per_mcast_group) are kept for a shared host arg layout.
    constexpr uint32_t reduce_receiver_semaphore_id = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_sender_semaphore_id = get_compile_time_arg_val(1);

    constexpr uint32_t num_batch_group = get_compile_time_arg_val(3);
    constexpr uint32_t num_batches = get_compile_time_arg_val(4);
    constexpr uint32_t num_groups = num_batch_group / num_batches;

    constexpr uint32_t per_core_N = get_compile_time_arg_val(5);
    const uint32_t per_core_N_bytes = get_compile_time_arg_val(6);
    const uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(7);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(8);
    constexpr uint32_t tile_height = get_compile_time_arg_val(9);
    constexpr uint32_t tile_width = get_compile_time_arg_val(10);

    constexpr uint32_t block_h = get_compile_time_arg_val(11);
    constexpr uint32_t block_w = get_compile_time_arg_val(12);

    constexpr uint32_t num_tiles_per_batch = get_compile_time_arg_val(13);

    constexpr uint32_t num_out_blocks = get_compile_time_arg_val(14);
    constexpr uint32_t num_channels_per_group = get_compile_time_arg_val(15);
    constexpr uint32_t num_rows_per_group = get_compile_time_arg_val(16);

    constexpr uint32_t cb_in0_welford_arg = get_compile_time_arg_val(17);
    constexpr bool welford_fp32_alias_arg = get_compile_time_arg_val(18) != 0;

    constexpr auto src0_args = TensorAccessorArgs<19>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t num_channels_tiles = get_arg_val<uint32_t>(4);

    const uint32_t mcast_sender_noc_x = get_arg_val<uint32_t>(5);
    const uint32_t mcast_sender_noc_y = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_repack_id = tt::CBIndex::c_26;
    constexpr uint32_t cb_repack_out_id = tt::CBIndex::c_31;
    constexpr uint32_t cb_out0_id = tt::CBIndex::c_16;
    constexpr uint32_t cb_in0_welford_id = cb_in0_welford_arg;
    constexpr bool welford_fp32_alias = welford_fp32_alias_arg;

    Noc noc;
    Semaphore<> reduce_receiver_sem(reduce_receiver_semaphore_id);
    Semaphore<> reduce_sender_sem(reduce_sender_semaphore_id);
    CircularBuffer cb_ex_partial(cb_ex_partial_id);
    CircularBuffer cb_ex_global(cb_ex_global_id);
    CircularBuffer cb_in0(cb_in0_id);
    CircularBuffer cb_in0_welford(cb_in0_welford_id);
    CircularBuffer cb_repack(cb_repack_id);
    CircularBuffer cb_repack_out(cb_repack_out_id);
    CircularBuffer cb_out0(cb_out0_id);

    constexpr uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial_id);
    constexpr uint32_t src0_tile_bytes = get_tile_size(cb_in0_id);

    constexpr uint32_t local_stride = 2;
    constexpr uint32_t single_row_size_bytes = single_tile_size_bytes / tile_height;
    constexpr uint32_t local_stride_per_group = local_stride * single_row_size_bytes;

    const auto src_a = TensorAccessor(src0_args, src_addr);

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = cb_in0.get_read_ptr();
    uint32_t src_addr_in0 = in0_l1_read_addr;
    UnicastEndpoint self_ep;
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack.reserve_back(per_core_N);
        uint32_t l1_write_addr_repack = cb_repack.get_write_ptr();
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                CoreLocalMem<uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc.async_read_barrier();
        cb_repack.push_back(per_core_N);
    }
#endif

    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    uint32_t num_out_blocks_padded = num_out_blocks;
    uint32_t extra_out_block = false;
    uint32_t out_block_h_last = out_block_h_normal;
    if constexpr (block_h % num_out_blocks != 0) {
        extra_out_block = true;
        num_out_blocks_padded++;
        out_block_h_last = block_h % num_out_blocks;
    }

    uint32_t index_b_offset = 0;
    for (uint32_t b = 0; b < num_batches; ++b) {
        uint32_t mt_offset = 0;
        for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
            uint32_t out_block_h_actual;
            if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                out_block_h_actual = out_block_h_last;
            } else {
                out_block_h_actual = out_block_h_normal;
            }

#if !defined(READER_REPACK) or !defined(TILIZE_IN)
            for (uint32_t mt = 0; mt < out_block_h_actual; ++mt) {
                for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                    cb_in0.reserve_back(1);
                    const uint32_t l1_write_addr = cb_in0.get_write_ptr();
                    noc.async_read(
                        src_a,
                        CoreLocalMem<uint32_t>(l1_write_addr),
                        src0_tile_bytes,
                        {.page_id = start_id + index_b_offset + mt_offset + nt},
                        {});
                    noc.async_read_barrier();
                    cb_in0.push_back(1);
                    if constexpr (welford_fp32_alias) {
                        cb_in0_welford.reserve_back(1);
                        cb_in0_welford.push_back(1);
                    }
                }
                mt_offset += num_channels_tiles;
            }
#endif
        }

        cb_ex_partial.wait_front(2);
        auto local_means_ptr = cb_ex_partial.get_read_ptr();
        auto local_vars_ptr = local_means_ptr + single_tile_size_bytes;

        cb_ex_global.reserve_back(2 * num_groups);
        auto global_means_ptr = cb_ex_global.get_write_ptr();
        auto global_vars_ptr = global_means_ptr + single_tile_size_bytes;

        // Combine every group's per-core partial into cb_ex_global (the master reads these via NoC).
        for (uint32_t m = 0; m < num_groups; ++m) {
            auto p_local_means = reinterpret_cast<volatile uint16_t*>(local_means_ptr);
            auto p_local_vars = reinterpret_cast<volatile uint16_t*>(local_vars_ptr);

            auto local_result = combine_welford_stats<
                tile_width,
                num_channels_per_group * num_rows_per_group / tile_width,
                local_stride>(p_local_means, p_local_vars);

            auto p_global_means = reinterpret_cast<volatile uint16_t*>(global_means_ptr);
            auto p_global_vars = reinterpret_cast<volatile uint16_t*>(global_vars_ptr);
            p_global_means[0] = local_result.mean;
            p_global_vars[0] = local_result.variance;

            local_means_ptr += local_stride_per_group;
            local_vars_ptr += local_stride_per_group;
            global_means_ptr += 2 * single_tile_size_bytes;
            global_vars_ptr += 2 * single_tile_size_bytes;
        }

        // Batched handshake: signal the master ONCE that all groups' partials are ready, then wait
        // ONCE for its single batched mcast-back of the GLOBAL (mean, var) for every group.
        reduce_receiver_sem.up(noc, mcast_sender_noc_x, mcast_sender_noc_y, 1);
        reduce_sender_sem.wait(VALID);
        reduce_sender_sem.set(INVALID);

        cb_ex_partial.pop_front(2);
        cb_ex_global.push_back(2 * num_groups);

        mt_offset = 0;
        for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
            uint32_t out_block_h_actual;
            if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                out_block_h_actual = out_block_h_last;
            } else {
                out_block_h_actual = out_block_h_normal;
            }
#if !defined(READER_REPACK) or !defined(TILIZE_IN)
            for (uint32_t mt = 0; mt < out_block_h_actual; ++mt) {
                for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                    cb_in0.reserve_back(1);
                    const uint32_t l1_write_addr = cb_in0.get_write_ptr();
                    noc.async_read(
                        src_a,
                        CoreLocalMem<uint32_t>(l1_write_addr),
                        src0_tile_bytes,
                        {.page_id = start_id + index_b_offset + mt_offset + nt},
                        {});
                    noc.async_read_barrier();
                    cb_in0.push_back(1);
                    if constexpr (welford_fp32_alias) {
                        cb_in0_welford.reserve_back(1);
                        cb_in0_welford.push_back(1);
                    }
                }
                mt_offset += num_channels_tiles;
            }
#endif
        }
        index_b_offset += num_tiles_per_batch;
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = cb_out0.get_write_ptr();
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_repack_out.wait_front(per_core_N);
        uint32_t in0_l1_read_addr = cb_repack_out.get_read_ptr();
        uint32_t src_addr_in0 = in0_l1_read_addr;
        UnicastEndpoint self_ep;
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                CoreLocalMem<uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc.async_read_barrier();
        cb_repack_out.pop_front(per_core_N);
    }
#endif
}
