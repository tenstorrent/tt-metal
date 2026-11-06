// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: A Metal 2.0 fork of this kernel lives beside it, as
// dm_in0_sender_metal2.cpp. Ops ported to Metal 2.0 bind the fork; this file serves
// the consumers still on the legacy API. Until the last of them migrates and
// this file is retired, changes here likely belong in the fork too.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "matmul_dataflow_common.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_M_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t padded_N_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(9);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(10);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(12);
    constexpr uint32_t in2_tile_size = get_compile_time_arg_val(13);
    Semaphore<> in0_sender_semaphore(get_compile_time_arg_val(14));
    Semaphore<> in0_receiver_semaphore(get_compile_time_arg_val(15));
    Semaphore<> in0_valid_semaphore(get_compile_time_arg_val(16));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(17);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(18);
    constexpr uint32_t N_chunks = get_compile_time_arg_val(19);
    constexpr uint32_t N_tiles_per_chunk = get_compile_time_arg_val(20);

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in2_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in3_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t max_defer_write_k_block = get_arg_val<uint32_t>(argidx++);

    const uint32_t out_addr_rt_arg_idx = argidx;  // Output addresses start here

    // Tensor accessor for input tensor
    constexpr auto in0_args = TensorAccessorArgs<22>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr);

    // Always create tuple of output accessors (size = N_chunks)
    constexpr uint32_t out_tensor_args_cta_offset = in0_args.next_compile_time_args_offset();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<N_chunks, out_tensor_args_cta_offset>();
    auto outputs_tuple = make_tensor_accessor_tuple_uniform_page_size(outputs_args, out_addr_rt_arg_idx, out_tile_size);

#ifdef FUSE_BIAS
    constexpr uint32_t in2_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
    constexpr auto in2_args = TensorAccessorArgs<in2_args_cta_offset>();
    const auto in2_reader = TensorAccessor(in2_args, in2_addr);
#endif

    const TensorShape2D in0_shape(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);
#ifdef MM_WINDOW_BLOCKS
    // The output tensor holds only the window, so its height is the host-computed
    // grid.y * MM_WINDOW_BLOCKS * M_block_tiles rather than the full M. Both the row bound and the
    // row stride come from it. Width is untouched — windowing is purely along M.
    const TensorShape2D out_shape(MM_WINDOW_TOTAL_M_TILES, N_tiles, MM_WINDOW_TOTAL_M_TILES, padded_N_tiles);
#else
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);
#endif
    const TensorShape2D out0_shape(M_tiles, N_tiles_per_chunk, padded_M_tiles, N_tiles_per_chunk);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_2;
#ifdef FUSE_BIAS
    constexpr uint32_t cb_in2_id = tt::CBIndex::c_4;
#endif

    CircularBuffer cb_in0(cb_in0_id);
    CircularBuffer cb_out(cb_out_id);
#ifdef FUSE_BIAS
    CircularBuffer cb_in2(cb_in2_id);
#endif

// in3 is the second in0 source buffer: the second concat half (e.g. mlp output) supplied via
// optional_input_tensor (virtual concat).
#ifdef IN0_VIRTUAL_CONCAT
#ifdef FUSE_BIAS
    // in3 (second concat half) sits right after in2 (bias); advance by in2's real arg count, not a fixed constant.
    constexpr auto in3_args = TensorAccessorArgs<in2_args.next_compile_time_args_offset()>();
#else
    constexpr uint32_t in3_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
    constexpr auto in3_args = TensorAccessorArgs<in3_args_cta_offset>();
#endif
    const auto in3_reader = TensorAccessor(in3_args, in3_addr);
#endif

#ifdef SRS_FUSE_OP_SIGNALER
    // OpSignaler runtime args start after output addresses
    uint32_t srs_fuse_signaler_rt_args_idx = out_addr_rt_arg_idx + N_chunks;
    OpSignaler srs_fuse_signaler;
    uint32_t mm_progress_counters_base = 0;
#ifdef MM_WINDOW_BLOCKS
    uint32_t M_window_start_tile = 0;
    uint32_t rs_credit_counters_base = 0;
    uint32_t num_rs_readers = 0;
#endif
    if constexpr (is_output_writer) {
        srs_fuse_signaler = OpSignaler(srs_fuse_signaler_rt_args_idx);
        // Per-core signaling: base L1 address of the RS cores' per-core progress counter array
        mm_progress_counters_base = get_arg_val<uint32_t>(srs_fuse_signaler_rt_args_idx++);
#ifdef MM_WINDOW_BLOCKS
        M_window_start_tile = get_arg_val<uint32_t>(srs_fuse_signaler_rt_args_idx++);
        rs_credit_counters_base = get_arg_val<uint32_t>(srs_fuse_signaler_rt_args_idx++);
        num_rs_readers = get_arg_val<uint32_t>(srs_fuse_signaler_rt_args_idx++);
        // Clear stale credits from whatever used this L1 before us. Safe against the readers: their
        // first credit only lands after they have consumed M block 0, which cannot happen until we
        // have written it, long after this point.
        volatile tt_l1_ptr uint32_t* credits = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs_credit_counters_base);
        for (uint32_t r = 0; r < num_rs_readers; r++) {
            credits[r] = 0;
        }
#endif
    }
#endif

    in0_valid_semaphore.set(VALID);

    /**
     * This is a Row-Major output block ordering.
     * It enables reuse of the last in0 block when striding the output block N dimension.
     */

    bool k_forward = true;
    bool reuse_block = false;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        // Rows this block writes to. Only the OUTPUT is windowed — this kernel also reads the
        // activations with m_tile/m_tile_end (see the in0 read below), and those must stay the true
        // rows, so the two cannot share a variable.
        uint32_t out_m_tile = m_tile;
        uint32_t out_m_tile_end = m_tile_end;
#ifdef MM_WINDOW_BLOCKS
        if constexpr (is_output_writer) {
            // Recycling this slot overwrites the block MM_WINDOW_BLOCKS earlier, so first wait until
            // EVERY RS reader has finished reading it. The minimum is what matters, not a total: the
            // readers stripe disjoint tiles and drift apart, so a fast one must not speak for a slow
            // one.
            if (m_block_iter >= MM_WINDOW_BLOCKS) {
                const uint32_t blocks_released_needed = m_block_iter - MM_WINDOW_BLOCKS + 1;
                volatile tt_l1_ptr uint32_t* credits =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs_credit_counters_base);
                for (uint32_t r = 0; r < num_rs_readers; r++) {
                    if (credits[r] < blocks_released_needed) {
                        noc_semaphore_wait_min(&credits[r], blocks_released_needed);
                    }
                }
            }
            out_m_tile = M_window_start_tile + (m_block_iter % MM_WINDOW_BLOCKS) * M_block_tiles;
            out_m_tile_end = out_m_tile + (m_tile_end - m_tile);
        }
#endif
        uint32_t current_M_block_tiles = m_tile_end - m_tile;
        uint32_t current_block_bytes = current_M_block_tiles * K_block_tiles * in0_tile_size;

        // When striding M block, in0 gets no reuse
        reuse_block = false;
        k_forward = true;
        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            bool is_last_block = (m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1));
            bool not_first_block = (n_block_iter > 0 || m_block_iter > 0);

            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
                        cb_out.wait_front(out_block_num_tiles);
                        uint32_t out_read_ptr = cb_out.get_read_ptr();

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
                        cb_out.pop_front(out_block_num_tiles);
                    }
                }

                if (reuse_block && k_block_iter == 0) {
                    // We strided an N block and this is the first k block, so we get reuse and do not need to read in0
                    reuse_block = false;
                    continue;
                }
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                cb_in0.reserve_back(in0_block_num_tiles);

                uint32_t in0_start_address = cb_in0.get_write_ptr();
                if constexpr (is_injector_core) {
                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                        in0_reader,
                        in0_shape,
                        cb_in0_id,
                        in0_tile_size,
#ifdef IN0_VIRTUAL_CONCAT
                        // Fused concatenation (concat-free): in0 (main) holds K-tiles [0, k_split);
                        // in3 holds K-tiles [k_split, K). main_Wt = k_split is the main buffer width.
                        in3_reader,
                        /*local_k_start=*/IN0_K_SPLIT_TILES,
                        /*local_k_end=*/K_tiles - 1,
                        /*input_tensor_Wt=*/K_tiles - IN0_K_SPLIT_TILES,
                        /*main_Wt=*/IN0_K_SPLIT_TILES,
#endif
                        m_tile,
                        m_tile_end,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles);
                } else {
                    // Get from previous device
                    in0_receiver_semaphore.set(INVALID);
                    in0_sender_semaphore.up(noc, in0_sender_noc_x, in0_sender_noc_y, 1);
                    in0_receiver_semaphore.wait(VALID);
                }

                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                cb_in0.push_back(in0_block_num_tiles);

                if (!is_sink_core) {
                    in0_sender_semaphore.wait(1);
                    in0_sender_semaphore.set(0);

                    /**
                     * in0 is M_block_tiles x K_block_tiles. When M block is partial, we don't need to write the
                     * padded tiles. Use `current_block_bytes`.
                     */
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
                    // Deferred-write path only (guarded by defer_write, which is false on the fused RS path).
                    if (defer_write && not_first_block && k_block_iter == max_defer_write_k_block) {
                        noc.async_write_barrier();
                        srs_fuse_signaler.signal_op_per_core(mm_progress_counters_base);
                    }
                }
#endif
            }
#ifdef FUSE_BIAS
            if constexpr (!is_output_writer) {
                cb_in2.reserve_back(N_block_tiles);

                uint32_t l1_write_addr_in2 = cb_in2.get_write_ptr();
                for (uint32_t n_tile_id = n_tile; n_tile_id < n_tile_end; n_tile_id++) {
                    noc.async_read(
                        in2_reader,
                        CoreLocalMem<uint32_t>(l1_write_addr_in2),
                        in2_tile_size,
                        {.page_id = n_tile_id},
                        {});
                    l1_write_addr_in2 += in2_tile_size;
                }
                noc.async_read_barrier();

                cb_in2.push_back(N_block_tiles);
            }
#endif

            k_forward = !k_forward;
            // We get reuse on in0 when striding N block
            reuse_block = true;

            defer_write_m_tile = out_m_tile;
            defer_write_m_tile_end = out_m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;
            /**
             * If this isn't the last output block, defer writing until the defer_k_write_block iteration
             * of the next output block.
             */
#ifdef SRS_FUSE_OP_SIGNALER
            // Fused RS path: write each block promptly
            defer_write = false;
#else
            defer_write = !is_last_block;
            defer_write = defer_write && !is_injector_core;
#endif

            if (!defer_write) {
                if constexpr (is_output_writer) {
                    // write_block_sync_granular_split is more generic (support multiple output tensors)
                    // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync_granular should be faster
                    if constexpr (N_chunks == 1) {
                        write_block_sync_granular<M_block_tiles, N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape,
                            cb_out_id,
                            out_tile_size,
                            out_m_tile,
                            out_m_tile_end,
                            n_tile,
                            n_tile_end);
                    } else {
                        write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape,
                            cb_out_id,
                            out_tile_size,
                            out_m_tile,
                            out_m_tile_end,
                            n_tile,
                            n_tile_end);
                    }
#ifdef SRS_FUSE_OP_SIGNALER
                    // Signal this core's per-core progress counter right after its prompt block write
                    noc.async_write_barrier();
                    srs_fuse_signaler.signal_op_per_core(mm_progress_counters_base);
#endif
                }
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
