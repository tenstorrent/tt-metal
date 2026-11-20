// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matmul in1 (weight) reader / forwarder (+ output writer when A::is_output_writer).
// Port of minimal_matmul/device/kernels/dm_in1_sender_out.cpp; same scope cuts as mm/in0.hpp.
//
// Compile-time values (A::): as In0 with in1_tile_size instead of in0_tile_size, cb_in1 instead of cb_in0, cta_in1;
//   k_order (+ k_slices k_slice_blocks k_first): visit K blocks in In0's slice-major order (no waits: weights).
// Runtime args: in1_addr is_sink dest_noc_x dest_noc_y sender_noc_x sender_noc_y M_start M_end N_start N_end
//   defer_write_k_block max_defer_write_k_block out_addr [signal_n_blocks: counters num_targets x0 y0 ...]
// n_stride n_perm signal_n_blocks: N groups and per-block signals (mm/n_blocks.hpp).

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common.hpp"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"
#include "ttnn/operations/experimental/lego/kernels/mm/n_blocks.hpp"
#include "ttnn/operations/experimental/lego/kernels/mm/bank_pairs.hpp"

namespace lego::mm {

template <typename A>
struct In1 {
    static constexpr uint32_t K_num_blocks = A::padded_K_tiles / A::K_block_tiles;
    static constexpr uint32_t in1_block_num_tiles = A::K_block_tiles * A::N_block_tiles;
    static constexpr uint32_t out_N_block_tiles = A::fuse_swiglu ? A::N_block_tiles / 2 : A::N_block_tiles;
    static constexpr uint32_t out_written_block_tiles = A::M_block_tiles * out_N_block_tiles;

    uint32_t in1_addr, is_sink_core, in1_dest_noc_x, in1_dest_noc_y, in1_sender_noc_x, in1_sender_noc_y;
    uint32_t M_start_tile, M_end_tile, N_start_tile, N_end_tile, defer_write_k_block, max_defer_write_k_block;
    uint32_t out_addr;

    void init() {
        uint32_t i = 0;
        in1_addr = rt<A>(i++);
        is_sink_core = rt<A>(i++);
        in1_dest_noc_x = rt<A>(i++);
        in1_dest_noc_y = rt<A>(i++);
        in1_sender_noc_x = rt<A>(i++);
        in1_sender_noc_y = rt<A>(i++);
        M_start_tile = rt<A>(i++);
        M_end_tile = rt<A>(i++);
        N_start_tile = rt<A>(i++);
        N_end_tile = rt<A>(i++);
        defer_write_k_block = rt<A>(i++);
        max_defer_write_k_block = rt<A>(i++);
        out_addr = rt<A>(i++);
    }

    // read_in1_block_sync without its barriers (in1_read_ahead: the caller waits later).
    template <typename Acc>
    static void issue_in1_block(
        Noc& noc,
        const Acc& acc,
        const TensorShape2D& shape,
        CircularBuffer& cb,
        uint32_t write_ptr,
        uint32_t d0_start,
        uint32_t d0_end,
        uint32_t d1_start,
        uint32_t d1_end) {
        const uint32_t base = cb.get_write_ptr();
        for (uint32_t i = d0_start; i < d0_end; i++) {
            for (uint32_t j = d1_start; j < d1_end; j++) {
                if (j < shape.logical_d1) {
                    if (i < shape.logical_d0) {
                        noc.async_read(
                            acc,
                            CoreLocalMem<uint32_t>(write_ptr),
                            A::in1_tile_size,
                            {.page_id = i * shape.logical_d1 + j},
                            {});
                    } else {
                        fill_zeros_async(noc, cb, A::in1_tile_size, write_ptr - base);
                    }
                }
                write_ptr += A::in1_tile_size;
            }
            write_ptr += (A::N_block_tiles - (d1_end - d1_start)) * A::in1_tile_size;
        }
    }

    // K block k of columns [n0, n1) into the CB block at write_ptr, no barrier (bank-paired quads or tile by tile)
    template <typename Acc>
    static void issue_block(
        Noc& noc,
        const Acc& acc,
        const TensorShape2D& shape,
        CircularBuffer& cb,
        uint32_t write_ptr,
        uint32_t k,
        uint32_t n0,
        uint32_t n1) {
        if constexpr (A::in1_bank_pairs > 0) {
            issue_in1_block_bp<A::in1_bank_pairs, A::N_block_tiles, A::in1_tile_size>(
                noc, acc, shape.logical_d1, write_ptr, k * A::K_block_tiles, (k + 1) * A::K_block_tiles, n0, n1);
        } else {
            issue_in1_block(noc, acc, shape, cb, write_ptr, k * A::K_block_tiles, (k + 1) * A::K_block_tiles, n0, n1);
        }
    }

    static uint32_t actual_k_block(uint32_t k_block) {
        if constexpr (A::k_order) {
            const uint32_t i = k_block / A::k_slice_blocks;
            const uint32_t step = (i + 1) / 2;
            const uint32_t s = ((i % 2 == 1) ? A::k_first + step : A::k_first + A::k_slices - step) % A::k_slices;
            return s * A::k_slice_blocks + k_block % A::k_slice_blocks;
        } else {
            return k_block;
        }
    }

    void run() {
        Noc noc;
        Semaphore<> in1_sender_semaphore(A::sender_sem);
        Semaphore<> in1_receiver_semaphore(A::receiver_sem);
        Semaphore<> in1_valid_semaphore(A::valid_sem);

        constexpr auto in1_args = TensorAccessorArgs<A::cta_in1>();
        const auto in1_reader = TensorAccessor(in1_args, in1_addr);
        constexpr auto out_args = TensorAccessorArgs<A::cta_out>();
        const auto out_writer = TensorAccessor(out_args, out_addr, A::out_tile_size);

        const TensorShape2D in1_shape(A::K_tiles, A::N_tiles, A::padded_K_tiles, A::padded_N_tiles);
        constexpr uint32_t out_N = A::fuse_swiglu ? A::N_tiles / 2 : A::N_tiles;
        constexpr uint32_t out_padded_N = A::fuse_swiglu ? A::padded_N_tiles / 2 : A::padded_N_tiles;
        const TensorShape2D out_shape(A::M_tiles, out_N, A::padded_M_tiles, out_padded_N);
        constexpr uint32_t n_div = A::fuse_swiglu ? 2 : 1;

        CircularBuffer cb_in1(A::cb_in1);
        CircularBuffer cb_out(A::cb_out);

        in1_valid_semaphore.set(VALID);

        constexpr uint32_t full_N_tiles_bytes = A::N_block_tiles * A::in1_tile_size;
        bool k_forward = true;
        uint32_t defer_write_m_tile = 0, defer_write_m_tile_end = 0, defer_write_n_tile = 0, defer_write_n_tile_end = 0;
        bool defer_write = false;

        for (uint32_t m_block_iter = 0; m_block_iter < A::M_blocks_per_core; m_block_iter++) {
            uint32_t m_tile, m_tile_end;
            if constexpr (A::m_batches > 1) {  // block b: this core's rows of batch b
                m_tile = M_start_tile + m_block_iter * A::m_batch_tiles;
                m_tile_end = M_end_tile + m_block_iter * A::m_batch_tiles;
            } else {
                m_tile = M_start_tile + m_block_iter * A::M_block_tiles;
                m_tile_end = std::min(m_tile + A::M_block_tiles, M_end_tile);
            }
            k_forward = true;
            for (uint32_t n_block_iter = 0; n_block_iter < A::N_blocks_per_core; n_block_iter++) {
                uint32_t n_tile, n_tile_end;
                n_block_range<A>(N_start_tile, N_end_tile, n_block_iter, n_tile, n_tile_end);
                uint32_t current_N_tiles_bytes = (n_tile_end - n_tile) * A::in1_tile_size;
                bool is_last_block =
                    (m_block_iter == A::M_blocks_per_core - 1) && (n_block_iter == (A::N_blocks_per_core - 1));

                for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                    if (defer_write && k_block_iter == defer_write_k_block) {
                        if constexpr (A::is_output_writer) {
                            cb_out.wait_front(out_written_block_tiles);
                            write_block_bp<A::in1_bank_pairs, out_N_block_tiles>(
                                out_writer,
                                out_shape,
                                cb_out.get_read_ptr(),
                                A::out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile / n_div,
                                defer_write_n_tile_end / n_div);
                            cb_out.pop_front(out_written_block_tiles);
                        }
                    }
                    uint32_t k_block = actual_k_block(k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter);
                    if (!(A::in1_read_ahead && A::is_injector_core && k_block_iter > 0)) {
                        cb_in1.reserve_back(in1_block_num_tiles);  // read ahead: reserved last iteration
                    }
                    uint32_t in1_start_address = cb_in1.get_write_ptr();
                    if constexpr (A::is_injector_core && (A::in1_read_ahead || A::in1_bank_pairs > 0)) {
                        LEGO_ZONE("in1_rd");
                        if (k_block_iter == 0 || !A::in1_read_ahead) {
                            issue_block(
                                noc, in1_reader, in1_shape, cb_in1, in1_start_address, k_block, n_tile, n_tile_end);
                        }
                        noc.async_read_barrier();
                        noc.write_zeros_l1_barrier();
                    } else if constexpr (A::is_injector_core) {
                        LEGO_ZONE("in1_rd");
                        read_in1_block_sync<A::K_block_tiles, A::N_block_tiles>(
                            in1_reader,
                            in1_shape,
                            A::cb_in1,
                            A::in1_tile_size,
                            k_block * A::K_block_tiles,
                            (k_block + 1) * A::K_block_tiles,
                            n_tile,
                            n_tile_end);
                    } else {
                        LEGO_ZONE("in1_rx");
                        in1_receiver_semaphore.set(INVALID);
                        in1_sender_semaphore.up(noc, in1_sender_noc_x, in1_sender_noc_y, 1);
                        in1_receiver_semaphore.wait(VALID);
                    }
                    cb_in1.push_back(in1_block_num_tiles);
                    if constexpr (A::is_injector_core && A::in1_read_ahead) {
                        if (k_block_iter + 1 < K_num_blocks) {  // the next block's reads run during the forward
                            const uint32_t nk =
                                actual_k_block(k_forward ? k_block_iter + 1 : (K_num_blocks - 2) - k_block_iter);
                            cb_in1.reserve_back(in1_block_num_tiles);
                            issue_block(
                                noc, in1_reader, in1_shape, cb_in1, cb_in1.get_write_ptr(), nk, n_tile, n_tile_end);
                        }
                    }

                    if (!is_sink_core) {
                        LEGO_ZONE("in1_fw");
                        in1_sender_semaphore.wait(1);
                        in1_sender_semaphore.set(0);
                        for (uint32_t i = 0; i < A::K_block_tiles; i++) {
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
                k_forward = A::in0_resident || !k_forward;  // resident in0: K in order for every N block

                defer_write_m_tile = m_tile;
                defer_write_m_tile_end = m_tile_end;
                defer_write_n_tile = n_tile;
                defer_write_n_tile_end = n_tile_end;
                defer_write = !is_last_block && !A::is_injector_core && !A::signal_n_blocks;

                if (!defer_write) {
                    if constexpr (A::is_output_writer) {
                        write_block_granular_bp<A::in1_bank_pairs, A::M_block_tiles, out_N_block_tiles>(
                            out_writer,
                            out_shape,
                            A::cb_out,
                            A::out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile / n_div,
                            n_tile_end / n_div);
                        if constexpr (A::signal_n_blocks) {
                            signal_n_block<A>(13, n_block_iter);
                        }
                    }
                }
            }
        }
    }

    void teardown() {
        Noc noc;
        noc.async_write_barrier();
        noc.async_atomic_barrier();
    }
};

}  // namespace lego::mm
