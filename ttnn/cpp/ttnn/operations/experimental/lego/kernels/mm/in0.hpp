// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matmul in0 reader / forwarder (+ output writer when A::is_output_writer).
// Port of minimal_matmul/device/kernels/dm_in0_sender.cpp: same block loop, same forward chain (an injector core
// reads in0 blocks from the tensor, every core forwards them to the next core of its row), same deferred output
// writes. Not ported: bias, fused ternary, N-chunked outputs, M windowing, the FUSE_AG / SRS signalers (Lego wires
// stage-to-stage signalling with separate blocks).
//
// Added hook (row_scale), shaped like the original's bias read (FUSE_BIAS reads the bias row after the K loop): after
// the K loop of an M block's first output block, wait until local semaphore scale_sem reaches scale_count, then read
// the block's M_block_tiles row-scale tiles (tile i of a [M, 32] tensor scales output row tile i) into cb_scale for the
// compute epilogue.
//
// Added hook (k_order), from all_gather_minimal_matmul's fused receiver: K blocks are visited in a slice-major order
// starting at slice k_first (k_first, k_first+1, k_first-1, k_first+2, ... of k_slices slices of k_slice_blocks K
// blocks each: the order an all-gather delivers them), and before the injector reads a K block of slice s it waits
// until the L1 counter word s at slice_counters + 16 * s reaches the count the all-gather sends for that slice
// (wait_own for slice k_first, wait_near for its two ring neighbours, wait_far for the others; per core). in1 must use
// the same order. The counters are zeroed in init(): the all-gather signals long after the kernel starts.
//
// Added hook (m_batches > 1): the M rows are m_batches batches of m_batch_tiles rows; M block b of a core is its rows
// of batch b (M_start / M_end are the core's rows in batch 0). With signal_m_batches the output writer increments L1
// counter word 16 * b on each target once every output tile of batch b it writes has landed (e.g. a reduce-scatter
// consuming the output batch by batch). in1 and compute take the same m_batches / m_batch_tiles.
//
// Compile-time values (A::): M_tiles padded_M_tiles K_tiles padded_K_tiles N_tiles padded_N_tiles M_block_tiles
//   K_block_tiles N_block_tiles M_blocks_per_core N_blocks_per_core in0_tile_size out_tile_size
//   sender_sem receiver_sem valid_sem is_output_writer is_injector_core fuse_swiglu cb_in0 cb_out
//   cta_in0 cta_out; row_scale (+ cb_scale scale_sem scale_count scale_tile_size cta_scale);
//   k_order (+ k_slices k_slice_blocks k_first)
// Runtime args: in0_addr is_sink dest_noc_x dest_noc_y sender_noc_x sender_noc_y M_start M_end N_start N_end
//   defer_write_k_block max_defer_write_k_block out_addr [scale_addr] [slice_counters wait_own wait_near wait_far]
//   [signal_m_batches: batch_counters num_targets x0 y0 ...]

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

namespace lego::mm {

template <typename A>
struct In0 {
    static constexpr uint32_t K_num_blocks = A::padded_K_tiles / A::K_block_tiles;
    static constexpr uint32_t in0_block_num_tiles = A::M_block_tiles * A::K_block_tiles;
    static constexpr uint32_t out_block_num_tiles = A::M_block_tiles * A::N_block_tiles;
    static constexpr uint32_t out_N_block_tiles = A::fuse_swiglu ? A::N_block_tiles / 2 : A::N_block_tiles;
    static constexpr uint32_t out_written_block_tiles = A::M_block_tiles * out_N_block_tiles;

    uint32_t in0_addr, is_sink_core, in0_dest_noc_x, in0_dest_noc_y, in0_sender_noc_x, in0_sender_noc_y;
    uint32_t M_start_tile, M_end_tile, N_start_tile, N_end_tile, defer_write_k_block, max_defer_write_k_block;
    uint32_t out_addr, scale_addr = 0, slice_counters = 0, wait_own = 0, wait_near = 0, wait_far = 0;
    uint32_t mb_sig_arg = 0;

    // The output of M block mb (all its N blocks) has been written by this core: signal its batch.
    void after_write(uint32_t mb, bool last_n) {
        if constexpr (A::signal_m_batches) {
            if (!last_n) {
                return;
            }
            noc_async_write_barrier();
            const uint32_t counters = rt<A>(mb_sig_arg);
            const uint32_t n = rt<A>(mb_sig_arg + 1);
            for (uint32_t t = 0; t < n; t++) {
                noc_semaphore_inc(
                    get_noc_addr(rt<A>(mb_sig_arg + 2 + 2 * t), rt<A>(mb_sig_arg + 3 + 2 * t), counters + 16 * mb), 1);
            }
        }
    }

    // Slice-major K order (see k_order above): the slice visited at position i.
    static uint32_t slice_at(uint32_t i) {
        constexpr uint32_t n = A::k_order ? A::k_slices : 1;
        const uint32_t first = A::k_order ? A::k_first : 0;
        // first, first+1, first-1, first+2, first-2, ...
        const uint32_t step = (i + 1) / 2;
        const uint32_t s = (i % 2 == 1) ? first + step : first + n - step;
        return s % n;
    }
    static uint32_t actual_k_block(uint32_t k_block) {
        if constexpr (A::k_order) {
            return slice_at(k_block / A::k_slice_blocks) * A::k_slice_blocks + k_block % A::k_slice_blocks;
        } else {
            return k_block;
        }
    }
    void wait_slice(uint32_t k_block) {
        if constexpr (A::k_order) {
            const uint32_t s = k_block / A::k_slice_blocks;
            const uint32_t d = (s + A::k_slices - A::k_first) % A::k_slices;  // ring distance from the own slice
            const uint32_t want = d == 0 ? wait_own : (d == 1 || d == A::k_slices - 1) ? wait_near : wait_far;
            volatile tt_l1_ptr uint32_t* ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slice_counters + 16 * s);
            if (*ctr < want) {
                LEGO_ZONE("mm_kwait");
                noc_semaphore_wait_min(ctr, want);
            }
        }
    }

    void init() {
        uint32_t i = 0;
        in0_addr = rt<A>(i++);
        is_sink_core = rt<A>(i++);
        in0_dest_noc_x = rt<A>(i++);
        in0_dest_noc_y = rt<A>(i++);
        in0_sender_noc_x = rt<A>(i++);
        in0_sender_noc_y = rt<A>(i++);
        M_start_tile = rt<A>(i++);
        M_end_tile = rt<A>(i++);
        N_start_tile = rt<A>(i++);
        N_end_tile = rt<A>(i++);
        defer_write_k_block = rt<A>(i++);
        max_defer_write_k_block = rt<A>(i++);
        out_addr = rt<A>(i++);
        if constexpr (A::row_scale) {
            scale_addr = rt<A>(i++);
        }
        if constexpr (A::k_order) {
            slice_counters = rt<A>(i++);
            wait_own = rt<A>(i++);
            wait_near = rt<A>(i++);
            wait_far = rt<A>(i++);
            for (uint32_t s = 0; s < A::k_slices; s++) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slice_counters + 16 * s)[0] = 0;
            }
        }
        mb_sig_arg = i;
    }

    // Row-scale tiles of rows [m_tile, m_tile_end) into cb_scale (padded to M_block_tiles).
    void read_row_scale(Noc& noc, uint32_t m_tile, uint32_t m_tile_end) {
        if constexpr (A::row_scale) {
            volatile tt_l1_ptr uint32_t* sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A::scale_sem));
            noc_semaphore_wait_min(sem, A::scale_count);
            constexpr auto scale_args = TensorAccessorArgs<A::cta_scale>();
            const auto scale_reader = TensorAccessor(scale_args, scale_addr, A::scale_tile_size);
            CircularBuffer cb_scale(A::cb_scale);
            cb_scale.reserve_back(A::M_block_tiles);
            uint32_t off = 0;
            for (uint32_t m = m_tile; m < m_tile_end && m < A::M_tiles; m++) {
                noc.async_read(scale_reader, cb_scale, A::scale_tile_size, {.page_id = m}, {.offset_bytes = off});
                off += A::scale_tile_size;
            }
            noc.async_read_barrier();
            cb_scale.push_back(A::M_block_tiles);
        }
    }

    // Write one finished output block (deferred path: the whole block is already in cb_out).
    template <typename Out>
    void write_deferred(
        const Out& out,
        const TensorShape2D& out_shape,
        CircularBuffer& cb_out,
        uint32_t m0,
        uint32_t m1,
        uint32_t n0,
        uint32_t n1) {
        cb_out.wait_front(out_written_block_tiles);
        write_block_sync<A::M_block_tiles, out_N_block_tiles>(
            out, out_shape, cb_out.get_read_ptr(), A::out_tile_size, m0, m1, n0, n1);
        cb_out.pop_front(out_written_block_tiles);
    }

    void run() {
        Noc noc;
        Semaphore<> in0_sender_semaphore(A::sender_sem);
        Semaphore<> in0_receiver_semaphore(A::receiver_sem);
        Semaphore<> in0_valid_semaphore(A::valid_sem);

        constexpr auto in0_args = TensorAccessorArgs<A::cta_in0>();
        const auto in0_reader = TensorAccessor(in0_args, in0_addr);
        constexpr auto out_args = TensorAccessorArgs<A::cta_out>();
        const auto out_writer = TensorAccessor(out_args, out_addr, A::out_tile_size);

        const TensorShape2D in0_shape(A::M_tiles, A::K_tiles, A::padded_M_tiles, A::padded_K_tiles);
        // With SwiGLU the output is half the weight N; the n ranges are halved at each write.
        constexpr uint32_t out_N = A::fuse_swiglu ? A::N_tiles / 2 : A::N_tiles;
        constexpr uint32_t out_padded_N = A::fuse_swiglu ? A::padded_N_tiles / 2 : A::padded_N_tiles;
        const TensorShape2D out_shape(A::M_tiles, out_N, A::padded_M_tiles, out_padded_N);
        constexpr uint32_t n_div = A::fuse_swiglu ? 2 : 1;

        CircularBuffer cb_in0(A::cb_in0);
        CircularBuffer cb_out(A::cb_out);

        in0_valid_semaphore.set(VALID);

        bool k_forward = true;
        bool reuse_block = false;
        uint32_t defer_write_m_tile = 0, defer_write_m_tile_end = 0, defer_write_n_tile = 0, defer_write_n_tile_end = 0;
        bool defer_write = false;
        uint32_t defer_write_mb = 0;
        bool defer_write_last_n = false;

        for (uint32_t m_block_iter = 0; m_block_iter < A::M_blocks_per_core; m_block_iter++) {
            uint32_t m_tile, m_tile_end;
            if constexpr (A::m_batches > 1) {  // block b: this core's rows of batch b
                m_tile = M_start_tile + m_block_iter * A::m_batch_tiles;
                m_tile_end = M_end_tile + m_block_iter * A::m_batch_tiles;
            } else {
                m_tile = M_start_tile + m_block_iter * A::M_block_tiles;
                m_tile_end = std::min(m_tile + A::M_block_tiles, M_end_tile);
            }
            uint32_t current_M_block_tiles = m_tile_end - m_tile;
            uint32_t current_block_bytes = current_M_block_tiles * A::K_block_tiles * A::in0_tile_size;

            reuse_block = false;
            k_forward = true;
            for (uint32_t n_block_iter = 0; n_block_iter < A::N_blocks_per_core; n_block_iter++) {
                uint32_t n_tile = N_start_tile + n_block_iter * A::N_block_tiles;
                uint32_t n_tile_end = std::min(n_tile + A::N_block_tiles, N_end_tile);
                bool is_last_block =
                    (m_block_iter == A::M_blocks_per_core - 1) && (n_block_iter == (A::N_blocks_per_core - 1));

                for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                    if (defer_write && k_block_iter == defer_write_k_block) {
                        if constexpr (A::is_output_writer) {
                            write_deferred(
                                out_writer,
                                out_shape,
                                cb_out,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile / n_div,
                                defer_write_n_tile_end / n_div);
                            after_write(defer_write_mb, defer_write_last_n);
                        }
                    }
                    if (reuse_block && k_block_iter == 0) {
                        // Strided an N block: the last in0 block is reused, nothing to read.
                        reuse_block = false;
                        continue;
                    }
                    uint32_t k_block = actual_k_block(k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter);
                    cb_in0.reserve_back(in0_block_num_tiles);
                    uint32_t in0_start_address = cb_in0.get_write_ptr();
                    if constexpr (A::is_injector_core) {
                        wait_slice(k_block);
                        read_in0_block_sync<A::M_block_tiles, A::K_block_tiles>(
                            in0_reader,
                            in0_shape,
                            A::cb_in0,
                            A::in0_tile_size,
                            m_tile,
                            m_tile_end,
                            k_block * A::K_block_tiles,
                            (k_block + 1) * A::K_block_tiles);
                    } else {
                        // Get from the previous core
                        in0_receiver_semaphore.set(INVALID);
                        in0_sender_semaphore.up(noc, in0_sender_noc_x, in0_sender_noc_y, 1);
                        in0_receiver_semaphore.wait(VALID);
                    }
                    // Push to compute before forwarding: frees the reader for its next read sooner.
                    cb_in0.push_back(in0_block_num_tiles);

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
                }

                if (n_block_iter == 0) {
                    read_row_scale(noc, m_tile, m_tile_end);
                }
                k_forward = !k_forward;
                reuse_block = true;  // in0 is reused when striding N

                defer_write_m_tile = m_tile;
                defer_write_m_tile_end = m_tile_end;
                defer_write_n_tile = n_tile;
                defer_write_n_tile_end = n_tile_end;
                defer_write_mb = m_block_iter;
                defer_write_last_n = n_block_iter == A::N_blocks_per_core - 1;
                // Defer this block's write into the next block's K loop, except for the last block (and, when batches
                // are signalled, a batch's last block: its consumer is waiting for it).
                defer_write = !is_last_block && !A::is_injector_core &&
                              !(A::signal_m_batches && n_block_iter == A::N_blocks_per_core - 1);

                if (!defer_write) {
                    if constexpr (A::is_output_writer) {
                        write_block_sync_granular<A::M_block_tiles, out_N_block_tiles>(
                            out_writer,
                            out_shape,
                            A::cb_out,
                            A::out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile / n_div,
                            n_tile_end / n_div);
                        after_write(m_block_iter, n_block_iter == A::N_blocks_per_core - 1);
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
