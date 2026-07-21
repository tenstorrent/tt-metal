// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "welford_combine.h"
#include "ckernel.h"
#include "noc_parameters.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t reduce_receiver_semaphore_id = get_named_compile_time_arg_val("reduce_receiver_semaphore_id");
    constexpr uint32_t reduce_sender_semaphore_id = get_named_compile_time_arg_val("reduce_sender_semaphore_id");

    constexpr uint32_t num_batch_group = get_named_compile_time_arg_val("num_batch_group");
    constexpr uint32_t num_batches = get_named_compile_time_arg_val("num_batches");
    constexpr uint32_t num_groups = num_batch_group / num_batches;

    constexpr uint32_t per_core_N = get_named_compile_time_arg_val("per_core_N");
    const uint32_t per_core_N_bytes = get_named_compile_time_arg_val("per_core_N_bytes");
    const uint32_t per_core_N_bytes_with_stride = get_named_compile_time_arg_val("per_core_N_bytes_with_stride");
    constexpr uint32_t per_core_M = get_named_compile_time_arg_val("per_core_M");
    constexpr uint32_t tile_height = get_named_compile_time_arg_val("TILE_HEIGHT");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("TILE_WIDTH");

    constexpr uint32_t block_h = get_named_compile_time_arg_val("block_h");
    constexpr uint32_t block_w = get_named_compile_time_arg_val("block_w");

    constexpr uint32_t num_tiles_per_batch = get_named_compile_time_arg_val("num_tiles_per_batch");

    constexpr uint32_t num_out_blocks = get_named_compile_time_arg_val("num_out_blocks");
    // These are numbers in absolute terms, on a per batch, per group, per core basis without tiling
    constexpr uint32_t num_channels_per_group = get_named_compile_time_arg_val("num_channels_per_group");
    constexpr uint32_t num_rows_per_group = get_named_compile_time_arg_val("num_rows_per_group");

    constexpr auto src0_args = TensorAccessorArgs<0>();

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
    // Welford-fp32 alias for cb_in0. Shares SRAM with cb_in0 but has its own buffer index
    // configured with UnpackToDestFp32, plus its own read/write pointers.
    // The Welford section of compute reads the alias to get full fp32 into DEST, while later
    // FPU consumers read cb_in0 directly. When welford_fp32_alias is false, cb_in0_welford_id
    // == cb_in0_id and the gated pushes below are skipped.
    constexpr uint32_t cb_in0_welford_id = get_named_compile_time_arg_val("cb_in0_welford");
    constexpr bool welford_fp32_alias = get_named_compile_time_arg_val("welford_fp32_alias") != 0;

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

    // This is the stride between two consecutive local means/variances in the cb_ex_partial
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
                        // Mirror the cb_in0 push on the alias. They share SRAM (multi-buffer-index
                        // alias) so the noc.async_read above already filled both views; this is
                        // purely bookkeeping so compute's welford section can wait_front
                        // on cb_in0_welford independently of cb_in0.
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

        for (uint32_t m = 0; m < num_groups; ++m) {
            // Read mean and variance arrays from cb_ex_partial, then combine using Welford
            auto p_local_means = reinterpret_cast<volatile uint16_t*>(local_means_ptr);
            auto p_local_vars = reinterpret_cast<volatile uint16_t*>(local_vars_ptr);

            auto local_result = combine_welford_stats<
                tile_width,
                num_channels_per_group * num_rows_per_group / tile_width,
                local_stride>(p_local_means, p_local_vars);

            // Write this to cb_ex_global
            auto p_global_means = reinterpret_cast<volatile uint16_t*>(global_means_ptr);
            auto p_global_vars = reinterpret_cast<volatile uint16_t*>(global_vars_ptr);
            p_global_means[0] = local_result.mean;
            p_global_vars[0] = local_result.variance;

            // Force both L1 stores to be processed into their L1 banks before we signal the sender,
            // which immediately NoC-reads these exact addresses back. Without this, the baby-RISC
            // store can still be in flight when the sender's read-back reaches the L1 bank -> the
            // sender aggregates a stale partial. A blocking load of the same address orders after the
            // store and stalls the pipeline until the store has landed (see ckernel.h load_blocking /
            // MemoryOrdering.md). Both addresses are drained because they lie in different L1 words
            // (single_tile_size_bytes apart) and may map to different L1 banks.
            ckernel::load_blocking(p_global_means);
            ckernel::load_blocking(p_global_vars);

            // Signal to sender that our partial data is ready
            reduce_receiver_sem.up(noc, mcast_sender_noc_x, mcast_sender_noc_y, 1);

            // Wait for sender to signal that it has sent the global data
            reduce_sender_sem.wait(VALID);
            reduce_sender_sem.set(INVALID);

            local_means_ptr += local_stride_per_group;
            local_vars_ptr += local_stride_per_group;
            global_means_ptr += 2 * single_tile_size_bytes;
            global_vars_ptr += 2 * single_tile_size_bytes;
        }

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
                        // Mirror the cb_in0 push on the alias. They share SRAM (multi-buffer-index
                        // alias) so the noc.async_read above already filled both views; this is
                        // purely bookkeeping so compute's welford section can wait_front
                        // on cb_in0_welford independently of cb_in0.
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
