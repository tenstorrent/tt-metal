// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of dm_in1_sender_out.cpp. Bound by MinimalMatmulDeviceOperation::ProgramFactory;
// the legacy original beside it still serves the fused-CCL emitter
// (minimal_matmul_factory_helper_common).

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

void kernel_main() {
    Noc noc;
    constexpr auto M_tiles = get_arg(args::M_tiles);
    constexpr auto padded_M_tiles = get_arg(args::padded_M_tiles);
    constexpr auto K_tiles = get_arg(args::K_tiles);
    constexpr auto padded_K_tiles = get_arg(args::padded_K_tiles);
    constexpr auto N_tiles = get_arg(args::N_tiles);
    constexpr auto padded_N_tiles = get_arg(args::padded_N_tiles);
    constexpr auto M_block_tiles = get_arg(args::M_block_tiles);
    constexpr auto K_block_tiles = get_arg(args::K_block_tiles);
    constexpr auto N_block_tiles = get_arg(args::N_block_tiles);
    constexpr auto M_blocks_per_core = get_arg(args::M_blocks_per_core);
    constexpr auto N_blocks_per_core = get_arg(args::N_blocks_per_core);
    constexpr auto in1_tile_size = get_arg(args::in1_tile_size);
    // Tile stride for the output L1 walks in write_block_sync*. (It also fed the legacy
    // TensorAccessor page-size 3rd argument, which the binding model now supplies itself.)
    constexpr auto out_tile_size = get_arg(args::out_tile_size);
    constexpr auto in2_tile_size = get_arg(args::in2_tile_size);
    Semaphore in1_sender_semaphore(sem::in1_sender);
    Semaphore in1_receiver_semaphore(sem::in1_receiver);
    Semaphore in1_valid_semaphore(sem::in1_valid);
    // is_output_writer was a compile-time arg; it is now a define, because it gates which
    // DataflowBuffers this kernel instance binds (dfb::out on the writer, dfb::in2 / dfb::ternary_*
    // on the non-writer) and `if constexpr` still name-looks-up the discarded branch.
    constexpr auto is_injector_core = get_arg(args::is_injector_core);
    constexpr auto N_chunks = get_arg(args::N_chunks);
    constexpr auto N_tiles_per_chunk = get_arg(args::N_tiles_per_chunk);

    // Range parameters. The buffer addresses that used to lead this list are TensorBindings now.
    const auto is_sink_core = get_arg(args::is_sink_core);
    const auto in1_dest_noc_x = get_arg(args::in1_dest_noc_x);
    const auto in1_dest_noc_y = get_arg(args::in1_dest_noc_y);
    const auto in1_sender_noc_x = get_arg(args::in1_sender_noc_x);
    const auto in1_sender_noc_y = get_arg(args::in1_sender_noc_y);
    const auto M_start_tile = get_arg(args::M_start_tile);
    const auto M_end_tile = get_arg(args::M_end_tile);
    const auto N_start_tile = get_arg(args::N_start_tile);
    const auto N_end_tile = get_arg(args::N_end_tile);
    const auto defer_write_k_block = get_arg(args::defer_write_k_block);
    const auto max_defer_write_k_block = get_arg(args::max_defer_write_k_block);

#ifdef FUSE_TERNARY
    const auto broadcast_ternary_b = get_arg(args::broadcast_ternary_b);
#endif  // FUSE_TERNARY

    const auto in1_reader = TensorAccessor(tensor::in1);

    // Tuple of output accessors, one per chunk. tensor::outputs is a TensorBindingSequence, which
    // is how a compile-time-variadic binding count is expressed: its size is N_chunks.
    [[maybe_unused]] auto outputs_tuple = make_tensor_accessors(tensor::outputs);

#ifdef FUSE_BIAS
    const auto in2_reader = TensorAccessor(tensor::in2);
#endif

#ifdef FUSE_TERNARY
    const auto ternary_a_reader = TensorAccessor(tensor::ternary_a);
    const auto ternary_b_reader = TensorAccessor(tensor::ternary_b);
#endif  // FUSE_TERNARY

    const TensorShape2D in1_shape(K_tiles, N_tiles, padded_K_tiles, padded_N_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);
    const TensorShape2D out0_shape(M_tiles, N_tiles_per_chunk, padded_M_tiles, N_tiles_per_chunk);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

#ifdef FUSE_SWIGLU
    // SwiGLU emits one output tile per interleaved gate/up pair, so the output along N
    // is half the matmul (weight) N. Compute the halved output geometry once here; the
    // weight-space n ranges (n_tile, N_tiles, ...) are halved at each write call site.
    constexpr uint32_t out_N_block_tiles = N_block_tiles / 2;
    constexpr uint32_t out_block_num_tiles_swiglu = M_block_tiles * out_N_block_tiles;
    const TensorShape2D out_shape_swiglu(M_tiles, N_tiles / 2, padded_M_tiles, padded_N_tiles / 2);
    // Split (chunks>1): each output chunk is half the weight per-chunk width.
    constexpr uint32_t out_N_tiles_per_chunk = N_tiles_per_chunk / 2;
    const TensorShape2D out0_shape_swiglu(M_tiles, out_N_tiles_per_chunk, padded_M_tiles, out_N_tiles_per_chunk);
#endif

    // in1 is bound on every instance; out only on the writer, in2 / ternary only on the non-writer.
    DataflowBuffer dfb_in1(dfb::in1);
#ifdef IS_OUTPUT_WRITER
    DataflowBuffer dfb_out(dfb::out);
#endif
#if defined(FUSE_BIAS) && !defined(IS_OUTPUT_WRITER)
    DataflowBuffer dfb_in2(dfb::in2);
#endif

#if defined(FUSE_TERNARY) && !defined(IS_OUTPUT_WRITER)
    DataflowBuffer dfb_ternary_a(dfb::ternary_a);
    DataflowBuffer dfb_ternary_b(dfb::ternary_b);
    // Legacy declared these constexpr, so they keep the free-function form with the binding token
    // (a member getter cannot produce a constant expression).
    constexpr uint32_t ternary_a_tile_size = get_tile_size(dfb::ternary_a);
    constexpr uint32_t ternary_b_tile_size = get_tile_size(dfb::ternary_b);
#endif

    in1_valid_semaphore.set(VALID);

    constexpr uint32_t full_N_tiles_bytes = N_block_tiles * in1_tile_size;

    bool k_forward = true;

    [[maybe_unused]] uint32_t defer_write_m_tile = 0;
    [[maybe_unused]] uint32_t defer_write_m_tile_end = 0;
    [[maybe_unused]] uint32_t defer_write_n_tile = 0;
    [[maybe_unused]] uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);

        k_forward = true;

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            uint32_t current_N_block_tiles = n_tile_end - n_tile;
            uint32_t current_N_tiles_bytes = current_N_block_tiles * in1_tile_size;
            bool is_last_block = (m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1));
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
#ifdef IS_OUTPUT_WRITER
#ifdef FUSE_SWIGLU
                    dfb_out.wait_front(out_block_num_tiles_swiglu);
                    uint32_t out_read_ptr = dfb_out.get_read_ptr();
                    if constexpr (N_chunks == 1) {
                        write_block_sync<M_block_tiles, out_N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape_swiglu,
                            out_read_ptr,
                            out_tile_size,
                            defer_write_m_tile,
                            defer_write_m_tile_end,
                            defer_write_n_tile / 2,
                            defer_write_n_tile_end / 2);
                    } else {
                        write_block_sync_split<M_block_tiles, out_N_block_tiles, N_chunks, out_N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape_swiglu,
                            out_read_ptr,
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

                    // write_block_sync_split is more generic (support multiple output tensors)
                    // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync should be faster
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
#endif  // FUSE_SWIGLU
#endif  // IS_OUTPUT_WRITER
                }

                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                dfb_in1.reserve_back(in1_block_num_tiles);

                uint32_t in1_start_address = dfb_in1.get_write_ptr();
                if constexpr (is_injector_core) {
                    read_in1_block_sync<K_block_tiles, N_block_tiles>(
                        in1_reader,
                        in1_shape,
                        dfb_in1,
                        in1_tile_size,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles,
                        n_tile,
                        n_tile_end);
                } else {
                    in1_receiver_semaphore.set(INVALID);
                    in1_sender_semaphore.up(noc, in1_sender_noc_x, in1_sender_noc_y, 1);
                    in1_receiver_semaphore.wait(VALID);
                }

                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                dfb_in1.push_back(in1_block_num_tiles);

                if (!is_sink_core) {
                    in1_sender_semaphore.wait(1);
                    in1_sender_semaphore.set(0);

                    /**
                     * in1 is K_block_tiles x N_block_tiles. When N block is partial, we don't need to write the
                     * padded tiles. For each tile in the K block, write only the non-padded N tiles. Use
                     * `current_N_tiles_bytes`.
                     */
                    for (uint32_t i = 0; i < K_block_tiles; i++) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(in1_start_address),
                            UnicastEndpoint{},
                            current_N_tiles_bytes,
                            {},
                            {.noc_x = in1_dest_noc_x, .noc_y = in1_dest_noc_y, .addr = in1_start_address});
                        in1_start_address += full_N_tiles_bytes;
                    }

#ifdef ARCH_BLACKHOLE
                    noc.async_writes_flushed();
#endif

                    in1_valid_semaphore.relay_unicast(noc, in1_receiver_semaphore, in1_dest_noc_x, in1_dest_noc_y);
                }
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
#endif  // !IS_OUTPUT_WRITER
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
#endif  // !IS_OUTPUT_WRITER
#endif  // FUSE_TERNARY

            k_forward = !k_forward;
            // We have an output block to write out

            defer_write_m_tile = m_tile;
            defer_write_m_tile_end = m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;
            /**
             * If this isn't the last output block, defer writing until the defer_k_write_block iteration
             * of the next output block.
             */
            defer_write = !is_last_block;
            // Injectors defer too: writing synchronously here would hold the next block's in1 / in0 injection (and
            // every core down its forwarding chain) behind this block's last subblock.

            if (!defer_write) {
#ifdef IS_OUTPUT_WRITER
#ifdef FUSE_SWIGLU
                if constexpr (N_chunks == 1) {
                    write_block_sync_granular<M_block_tiles, out_N_block_tiles>(
                        std::get<0>(outputs_tuple),
                        out_shape_swiglu,
                        dfb_out,
                        out_tile_size,
                        m_tile,
                        m_tile_end,
                        n_tile / 2,
                        n_tile_end / 2);
                } else {
                    write_block_sync_granular_split<M_block_tiles, out_N_block_tiles, N_chunks, out_N_tiles_per_chunk>(
                        outputs_tuple,
                        out0_shape_swiglu,
                        dfb_out,
                        out_tile_size,
                        m_tile,
                        m_tile_end,
                        n_tile / 2,
                        n_tile_end / 2);
                }
#else
                // write_block_sync_granular_split is more generic (support multiple output tensors)
                // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync_granular should be faster
                if constexpr (N_chunks == 1) {
                    write_block_sync_granular<M_block_tiles, N_block_tiles>(
                        std::get<0>(outputs_tuple),
                        out_shape,
                        dfb_out,
                        out_tile_size,
                        m_tile,
                        m_tile_end,
                        n_tile,
                        n_tile_end);
                } else {
                    write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                        outputs_tuple, out0_shape, dfb_out, out_tile_size, m_tile, m_tile_end, n_tile, n_tile_end);
                }
#endif  // FUSE_SWIGLU
#endif  // IS_OUTPUT_WRITER
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
