// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "matmul_dataflow_common_metal2.hpp"
#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp"

#if defined(DIR_SPLIT)
#define K_FORWARD_AT(n) ((n) < DIR_SPLIT)
#elif defined(DIR_PERIOD)
#define K_FORWARD_AT(n) (((((n) % DIR_PERIOD) / DIR_BLOCK) % 2) == 0)
#endif

#if defined(N_ROTATE) && defined(DIR_PERIOD)
#define N_NEXT_TILE n_next_tile
#define IN1_BULK_NBI n_block_rot
#else
#define N_NEXT_TILE (n_tile + N_block_tiles)
#define IN1_BULK_NBI n_block_iter
#endif
void kernel_main() {
    Noc noc;
    constexpr auto M_cores = get_arg(args::M_cores);
    constexpr auto K_tiles = get_arg(args::K_tiles);
    constexpr auto padded_K_tiles = get_arg(args::padded_K_tiles);
    constexpr auto N_tiles = get_arg(args::N_tiles);
    constexpr auto padded_N_tiles = get_arg(args::padded_N_tiles);
    constexpr auto M_block_tiles = get_arg(args::M_block_tiles);
    constexpr auto K_block_tiles = get_arg(args::K_block_tiles);
    constexpr auto N_block_tiles = get_arg(args::N_block_tiles);
    constexpr auto N_blocks_per_core = get_arg(args::N_blocks_per_core);
    constexpr auto in0_tile_size = get_arg(args::in0_tile_size);
    constexpr auto out_tile_size = get_arg(args::out_tile_size);
    constexpr auto in2_tile_size = get_arg(args::in2_tile_size);
    Semaphore in0_sender_semaphore(sem::in0_sender);
    Semaphore in0_receiver_semaphore(sem::in0_receiver);
    Semaphore in0_valid_semaphore(sem::in0_valid);
    constexpr auto is_injector_core = get_arg(args::is_injector_core);
    constexpr auto N_chunks = get_arg(args::N_chunks);
    constexpr auto N_tiles_per_chunk = get_arg(args::N_tiles_per_chunk);

    const auto is_sink_core = get_arg(args::is_sink_core);
    const auto in0_dest_noc_x = get_arg(args::in0_dest_noc_x);
    const auto in0_dest_noc_y = get_arg(args::in0_dest_noc_y);
    const auto in0_sender_noc_x = get_arg(args::in0_sender_noc_x);
    const auto in0_sender_noc_y = get_arg(args::in0_sender_noc_y);
    const auto in0_idx = get_arg(args::in0_idx);
    const auto N_start_tile = get_arg(args::N_start_tile);
    const auto N_end_tile = get_arg(args::N_end_tile);
    const auto defer_write_k_block = get_arg(args::defer_write_k_block);
    const auto max_defer_write_k_block = get_arg(args::max_defer_write_k_block);

#ifdef FUSE_TERNARY
    const auto broadcast_ternary_b = get_arg(args::broadcast_ternary_b);
#endif

    const uint32_t M_tiles = get_arg(args::M_tiles);
    const auto [M_tiles_per_core, M_blocks_per_core, padded_M_tiles, M_start_tile, M_end_tile] =
        ttnn::runtime_length::uniform_split(M_tiles, M_cores, in0_idx, M_block_tiles);
#if defined(N_ROTATE) && defined(DIR_PERIOD)
    uint32_t n_rot_start = 0;
    while (n_rot_start + 1 < N_blocks_per_core && !K_FORWARD_AT(N_start_tile + n_rot_start * N_block_tiles)) {
        n_rot_start++;
    }
#endif

    const auto in0_reader = TensorAccessor(tensor::in0);

    [[maybe_unused]] auto outputs_tuple = make_tensor_accessors(tensor::outputs);

#ifdef FUSE_BIAS
    const auto in2_reader = TensorAccessor(tensor::in2);
#endif

#ifdef FUSE_TERNARY
    const auto ternary_a_reader = TensorAccessor(tensor::ternary_a);
    const auto ternary_b_reader = TensorAccessor(tensor::ternary_b);
#endif

    const TensorShape2D in0_shape(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);
    const TensorShape2D out0_shape(M_tiles, N_tiles_per_chunk, padded_M_tiles, N_tiles_per_chunk);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

#ifdef FUSE_SWIGLU
    constexpr uint32_t out_N_block_tiles = N_block_tiles / 2;
    constexpr uint32_t out_block_num_tiles_swiglu = M_block_tiles * out_N_block_tiles;
    const TensorShape2D out_shape_swiglu(M_tiles, N_tiles / 2, padded_M_tiles, padded_N_tiles / 2);
    constexpr uint32_t out_N_tiles_per_chunk = N_tiles_per_chunk / 2;
    const TensorShape2D out0_shape_swiglu(M_tiles, out_N_tiles_per_chunk, padded_M_tiles, out_N_tiles_per_chunk);
#endif

    DataflowBuffer dfb_in0(dfb::in0);
#ifdef IS_OUTPUT_WRITER
    DataflowBuffer dfb_out(dfb::out);
#endif
#if defined(FUSE_BIAS) && !defined(IS_OUTPUT_WRITER)
    DataflowBuffer dfb_in2(dfb::in2);
#endif

#if defined(FUSE_TERNARY) && !defined(IS_OUTPUT_WRITER)
    DataflowBuffer dfb_ternary_a(dfb::ternary_a);
    DataflowBuffer dfb_ternary_b(dfb::ternary_b);
    constexpr uint32_t ternary_a_tile_size = get_tile_size(dfb::ternary_a);
    constexpr uint32_t ternary_b_tile_size = get_tile_size(dfb::ternary_b);
#endif

#ifdef FUSE_AG
    MinimalMatmulOpReceiver fused_op_receiver;
    uint32_t fused_op_rt_args_idx = out_addr_rt_arg_idx + N_chunks;
    uint32_t num_devices = get_arg_val<uint32_t>(fused_op_rt_args_idx);
    uint32_t num_k_blocks = get_arg_val<uint32_t>(fused_op_rt_args_idx + 1);
    uint8_t k_block_device_expected[num_k_blocks]{};
    uint8_t k_block_device_received[num_k_blocks]{};
    uint32_t device_k_block_counts[num_devices]{};
    uint32_t device_k_block_start_ids[num_devices]{};
    uint32_t forward_k_block_schedule[num_k_blocks]{};
    if constexpr (is_injector_core) {
        fused_op_receiver = MinimalMatmulOpReceiver(
            true,
            fused_op_rt_args_idx,
            k_block_device_expected,
            k_block_device_received,
            device_k_block_counts,
            device_k_block_start_ids,
            forward_k_block_schedule);
    }
#endif

#ifdef IN0_HAS_SECOND_SOURCE
    const auto in3_reader = TensorAccessor(tensor::in3);
#endif

#ifdef SRS_FUSE_OP_SIGNALER
    uint32_t srs_fuse_signaler_rt_args_idx = out_addr_rt_arg_idx + N_chunks;
#ifdef FUSE_AG
    srs_fuse_signaler_rt_args_idx += 12;
#endif
    OpSignaler srs_fuse_signaler;
    uint32_t mm_progress_counters_base = 0;
    if constexpr (is_output_writer) {
        srs_fuse_signaler = OpSignaler(srs_fuse_signaler_rt_args_idx);
        mm_progress_counters_base = get_arg_val<uint32_t>(srs_fuse_signaler_rt_args_idx++);
    }
#endif

    in0_valid_semaphore.set(VALID);

    bool k_forward = true;
    bool reuse_block = false;

    [[maybe_unused]] uint32_t defer_write_m_tile = 0;
    [[maybe_unused]] uint32_t defer_write_m_tile_end = 0;
    [[maybe_unused]] uint32_t defer_write_n_tile = 0;
    [[maybe_unused]] uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        uint32_t out_m_tile = m_tile;
        uint32_t out_m_tile_end = m_tile_end;
        uint32_t current_M_block_tiles = m_tile_end - m_tile;
        uint32_t current_block_bytes = current_M_block_tiles * K_block_tiles * in0_tile_size;
#ifdef FUSE_AG
        if constexpr (is_injector_core) {
            fused_op_receiver.reset();
        }
#endif

        reuse_block = false;
        k_forward = true;
#ifdef DIR_PERIOD
#if defined(N_ROTATE)
        k_forward = K_FORWARD_AT(N_start_tile + n_rot_start * N_block_tiles);
#else
        k_forward = K_FORWARD_AT(N_start_tile);
#endif
#endif
        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
#if defined(N_ROTATE) && defined(DIR_PERIOD)
            [[maybe_unused]] const uint32_t n_block_rot = (n_block_iter + n_rot_start) % N_blocks_per_core;
            uint32_t n_tile = N_start_tile + n_block_rot * N_block_tiles;
            [[maybe_unused]] const uint32_t n_next_tile =
                N_start_tile + ((n_block_iter + 1 + n_rot_start) % N_blocks_per_core) * N_block_tiles;
#else
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
#endif
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            bool is_last_block = (m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1));
            bool not_first_block = (n_block_iter > 0 || m_block_iter > 0);

            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
#ifdef IS_OUTPUT_WRITER
#ifdef FUSE_SWIGLU
                    dfb_out.wait_front(out_block_num_tiles_swiglu);
                    uint32_t out_read_ptr_swiglu = dfb_out.get_read_ptr();
                    if constexpr (N_chunks == 1) {
                        write_block_sync<M_block_tiles, out_N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape_swiglu,
                            out_read_ptr_swiglu,
                            out_tile_size,
                            defer_write_m_tile,
                            defer_write_m_tile_end,
                            defer_write_n_tile / 2,
                            defer_write_n_tile_end / 2);
                    } else {
                        write_block_sync_split<M_block_tiles, out_N_block_tiles, N_chunks, out_N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape_swiglu,
                            out_read_ptr_swiglu,
                            out_tile_size,
                            defer_write_m_tile,
                            defer_write_m_tile_end,
                            defer_write_n_tile / 2,
                            defer_write_n_tile_end / 2);
                    }
                    dfb_out.pop_front(out_block_num_tiles_swiglu);
#else
                    dfb_out.wait_front(out_block_num_tiles);
                    uint32_t out_read_ptr = dfb_out.get_read_ptr();

                    if constexpr (N_chunks == 1) {
                        write_block_sync<M_block_tiles, N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape,
                            out_read_ptr,
                            out_tile_size,
                            defer_write_m_tile,
                            defer_write_m_tile_end,
                            defer_write_n_tile,
                            defer_write_n_tile_end);
                    } else {
                        write_block_sync_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape,
                            out_read_ptr,
                            out_tile_size,
                            defer_write_m_tile,
                            defer_write_m_tile_end,
                            defer_write_n_tile,
                            defer_write_n_tile_end);
                    }
                    dfb_out.pop_front(out_block_num_tiles);
#endif
#endif
                }

#ifdef IN0_RESIDENT
                if (n_block_iter > 0) {
                    continue;
                }
#endif
                if (reuse_block && k_block_iter == 0) {
                    reuse_block = false;
                    continue;
                }
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
#ifdef IN0_RESIDENT
                k_block = k_block_iter;
#endif
                dfb_in0.reserve_back(in0_block_num_tiles);

                uint32_t in0_start_address = dfb_in0.get_write_ptr();
                if constexpr (is_injector_core) {
#ifdef FUSE_AG
                    if (is_injector_core) {
                        k_block =
                            fused_op_receiver.compute_actual_k_block_iter(n_block_iter == 0, k_block_iter, k_forward);
                    }
#endif
                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                        in0_reader,
                        in0_shape,
                        dfb_in0,
                        in0_tile_size,
#ifdef IN0_HAS_SECOND_SOURCE
#ifdef IN0_VIRTUAL_CONCAT
                        in3_reader,
                        IN0_K_SPLIT_TILES,
                        K_tiles - 1,
                        K_tiles - IN0_K_SPLIT_TILES,
                        IN0_K_SPLIT_TILES,
#else
                        in3_reader,
                        fused_op_receiver.local_k_start,
                        fused_op_receiver.local_k_end,
                        fused_op_receiver.input_tensor_Wt,
                        K_tiles,
#endif
#endif
                        m_tile,
                        m_tile_end,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles);
                } else {
                    in0_receiver_semaphore.set(INVALID);
                    in0_sender_semaphore.up(noc, in0_sender_noc_x, in0_sender_noc_y, 1);
                    in0_receiver_semaphore.wait(VALID);
                }

                dfb_in0.push_back(in0_block_num_tiles);

                if (!is_sink_core) {
                    in0_sender_semaphore.wait(1);
                    in0_sender_semaphore.set(0);

                    noc.async_write(
                        CoreLocalMem<uint32_t>(in0_start_address),
                        UnicastEndpoint{},
                        current_block_bytes,
                        {},
                        {.noc_x = in0_dest_noc_x, .noc_y = in0_dest_noc_y, .addr = in0_start_address});

#ifdef ARCH_BLACKHOLE
                    noc.async_writes_flushed();
#endif

                    in0_valid_semaphore.relay_unicast(noc, in0_receiver_semaphore, in0_dest_noc_x, in0_dest_noc_y);
                }
#ifdef SRS_FUSE_OP_SIGNALER
                if constexpr (is_output_writer) {
                    if (defer_write && not_first_block && k_block_iter == max_defer_write_k_block) {
                        noc.async_write_barrier();
                        srs_fuse_signaler.signal_op_per_core(mm_progress_counters_base);
                    }
                }
#endif
            }
#ifdef FUSE_BIAS
#ifndef IS_OUTPUT_WRITER
            dfb_in2.reserve_back(N_block_tiles);

            uint32_t l1_write_addr_in2 = dfb_in2.get_write_ptr();
            for (uint32_t n_tile_id = n_tile; n_tile_id < n_tile_end; n_tile_id++) {
                noc.async_read(
                    in2_reader, CoreLocalMem<uint32_t>(l1_write_addr_in2), in2_tile_size, {.page_id = n_tile_id}, {});
                l1_write_addr_in2 += in2_tile_size;
            }
            noc.async_read_barrier();

            dfb_in2.push_back(N_block_tiles);
#endif
#endif

#ifdef FUSE_TERNARY
#ifndef IS_OUTPUT_WRITER
            read_ternary_blocks_sync<M_block_tiles, N_block_tiles>(
                ternary_a_reader,
                ternary_b_reader,
                out_shape,
                dfb_ternary_a,
                dfb_ternary_b,
                ternary_a_tile_size,
                ternary_b_tile_size,
                broadcast_ternary_b,
                m_tile,
                m_tile_end,
                n_tile,
                n_tile_end);
#endif
#endif

#ifdef DIR_PERIOD
            {
                const bool next_forward = K_FORWARD_AT(N_NEXT_TILE);
                reuse_block = (next_forward != k_forward);
                k_forward = next_forward;
            }
#else
            k_forward = !k_forward;
            reuse_block = true;
#endif

            defer_write_m_tile = out_m_tile;
            defer_write_m_tile_end = out_m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;
#ifdef SRS_FUSE_OP_SIGNALER
            defer_write = false;
#else
            defer_write = !is_last_block;
#endif

            if (!defer_write) {
#ifdef IS_OUTPUT_WRITER
#ifdef FUSE_SWIGLU
                if constexpr (N_chunks == 1) {
                    write_block_sync_granular<M_block_tiles, out_N_block_tiles>(
                        std::get<0>(outputs_tuple),
                        out_shape_swiglu,
                        dfb_out,
                        out_tile_size,
                        out_m_tile,
                        out_m_tile_end,
                        n_tile / 2,
                        n_tile_end / 2);
                } else {
                    write_block_sync_granular_split<M_block_tiles, out_N_block_tiles, N_chunks, out_N_tiles_per_chunk>(
                        outputs_tuple,
                        out0_shape_swiglu,
                        dfb_out,
                        out_tile_size,
                        out_m_tile,
                        out_m_tile_end,
                        n_tile / 2,
                        n_tile_end / 2);
                }
#else
                if constexpr (N_chunks == 1) {
                    write_block_sync_granular<M_block_tiles, N_block_tiles>(
                        std::get<0>(outputs_tuple),
                        out_shape,
                        dfb_out,
                        out_tile_size,
                        out_m_tile,
                        out_m_tile_end,
                        n_tile,
                        n_tile_end);
                } else {
                    write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                        outputs_tuple,
                        out0_shape,
                        dfb_out,
                        out_tile_size,
                        out_m_tile,
                        out_m_tile_end,
                        n_tile,
                        n_tile_end);
                }
#endif
#ifdef SRS_FUSE_OP_SIGNALER
                noc.async_write_barrier();
                srs_fuse_signaler.signal_op_per_core(mm_progress_counters_base);
#endif
#endif
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
