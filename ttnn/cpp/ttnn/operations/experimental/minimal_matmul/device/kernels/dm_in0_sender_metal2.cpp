// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of dm_in0_sender.cpp. Bound by MinimalMatmulDeviceOperation::ProgramFactory;
// the legacy original beside it still serves the fused-CCL emitter
// (minimal_matmul_factory_helper_common).
//
// The FUSE_AG / SRS_FUSE_OP_SIGNALER / MM_WINDOW_BLOCKS / READ_FROM_LOCAL_INPUT regions below are
// retained verbatim from the original but are NOT converted: only the legacy emitter defines those
// macros, so the fork never compiles them. They still read positional runtime args (e.g. an
// out_addr_rt_arg_idx base) that this file no longer has. Converting one means giving the Metal 2.0
// spec the matching bindings first -- do that with the CCL op's own port, not here.

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
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp"

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
    constexpr auto in0_tile_size = get_arg(args::in0_tile_size);
    // Tile stride for the output L1 walks in write_block_sync*. (It also fed the legacy
    // TensorAccessor page-size 3rd argument, which the binding model now supplies itself.)
    constexpr auto out_tile_size = get_arg(args::out_tile_size);
    constexpr auto in2_tile_size = get_arg(args::in2_tile_size);
    Semaphore in0_sender_semaphore(sem::in0_sender);
    Semaphore in0_receiver_semaphore(sem::in0_receiver);
    Semaphore in0_valid_semaphore(sem::in0_valid);
    // is_output_writer was a compile-time arg; it is now a define, because it gates which
    // DataflowBuffers this kernel instance binds (dfb::out on the writer, dfb::in2 / dfb::ternary_*
    // on the non-writer) and `if constexpr` still name-looks-up the discarded branch.
    constexpr auto is_injector_core = get_arg(args::is_injector_core);
    constexpr auto N_chunks = get_arg(args::N_chunks);
    constexpr auto N_tiles_per_chunk = get_arg(args::N_tiles_per_chunk);

    // Range parameters. The buffer addresses that used to lead this list are TensorBindings now.
    const auto is_sink_core = get_arg(args::is_sink_core);
    const auto in0_dest_noc_x = get_arg(args::in0_dest_noc_x);
    const auto in0_dest_noc_y = get_arg(args::in0_dest_noc_y);
    const auto in0_sender_noc_x = get_arg(args::in0_sender_noc_x);
    const auto in0_sender_noc_y = get_arg(args::in0_sender_noc_y);
    const auto M_start_tile = get_arg(args::M_start_tile);
    const auto M_end_tile = get_arg(args::M_end_tile);
    const auto N_start_tile = get_arg(args::N_start_tile);
    const auto N_end_tile = get_arg(args::N_end_tile);
    const auto defer_write_k_block = get_arg(args::defer_write_k_block);
    const auto max_defer_write_k_block = get_arg(args::max_defer_write_k_block);

#ifdef FUSE_TERNARY
    const auto broadcast_ternary_b = get_arg(args::broadcast_ternary_b);
#endif  // FUSE_TERNARY

    const auto in0_reader = TensorAccessor(tensor::in0);

    // Tuple of output accessors, one per chunk. tensor::outputs is a TensorBindingSequence, which
    // is how a compile-time-variadic binding count is expressed: its size is N_chunks.
    auto outputs_tuple = make_tensor_accessors(tensor::outputs);

#ifdef FUSE_BIAS
    const auto in2_reader = TensorAccessor(tensor::in2);
#endif

#ifdef FUSE_TERNARY
    const auto ternary_a_reader = TensorAccessor(tensor::ternary_a);
    const auto ternary_b_reader = TensorAccessor(tensor::ternary_b);
#endif  // FUSE_TERNARY

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

#ifdef FUSE_SWIGLU
    // SwiGLU emits one output tile per interleaved gate/up pair, so the output along N
    // is half the matmul (weight) N. The weight-space n ranges are halved at each write.
    constexpr uint32_t out_N_block_tiles = N_block_tiles / 2;
    constexpr uint32_t out_block_num_tiles_swiglu = M_block_tiles * out_N_block_tiles;
    const TensorShape2D out_shape_swiglu(M_tiles, N_tiles / 2, padded_M_tiles, padded_N_tiles / 2);
    // Split (chunks>1): each output chunk is half the weight per-chunk width.
    constexpr uint32_t out_N_tiles_per_chunk = N_tiles_per_chunk / 2;
    const TensorShape2D out0_shape_swiglu(M_tiles, out_N_tiles_per_chunk, padded_M_tiles, out_N_tiles_per_chunk);
#endif

    // in0 is bound on every instance; out only on the writer, in2 / ternary only on the non-writer.
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
    // Legacy declared these constexpr, so they keep the free-function form with the binding token
    // (a member getter cannot produce a constant expression).
    constexpr uint32_t ternary_a_tile_size = get_tile_size(dfb::ternary_a);
    constexpr uint32_t ternary_b_tile_size = get_tile_size(dfb::ternary_b);
#endif

#ifdef FUSE_AG
    // Receiver for ccl fusing
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
#endif  // FUSE_AG

// in3 is the second in0 source buffer. AG path: this device's local pre-gather slice. Virtual
// concat: the second concat half (e.g. mlp output) supplied via optional_input_tensor. Set up
// whenever in0 has a second source, independent of FUSE_AG.
#ifdef IN0_HAS_SECOND_SOURCE
    const auto in3_reader = TensorAccessor(tensor::in3);
#endif

#ifdef SRS_FUSE_OP_SIGNALER
    // OpSignaler runtime args start after output addresses and optional FUSE_AG args
    uint32_t srs_fuse_signaler_rt_args_idx = out_addr_rt_arg_idx + N_chunks;
#ifdef FUSE_AG
    srs_fuse_signaler_rt_args_idx += 12;  // Skip MinimalMatmulFusedOpSignaler::push_matmul_fused_op_rt_args (12 args)
#endif
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
#ifdef FUSE_AG
        if constexpr (is_injector_core) {
            fused_op_receiver.reset();
        }
#endif

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

                if (reuse_block && k_block_iter == 0) {
                    // We strided an N block and this is the first k block, so we get reuse and do not need to read in0
                    reuse_block = false;
                    continue;
                }
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
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
                        // Fused concatenation (concat-free): in0 (main) holds K-tiles [0, k_split);
                        // in3 holds K-tiles [k_split, K). main_Wt = k_split is the main buffer width.
                        in3_reader,
                        /*local_k_start=*/IN0_K_SPLIT_TILES,
                        /*local_k_end=*/K_tiles - 1,
                        /*input_tensor_Wt=*/K_tiles - IN0_K_SPLIT_TILES,
                        /*main_Wt=*/IN0_K_SPLIT_TILES,
#else
                        // AG: in0 is the full gathered K; in3 is this device's local pre-gather slice.
                        in3_reader,
                        fused_op_receiver.local_k_start,
                        fused_op_receiver.local_k_end,
                        fused_op_receiver.input_tensor_Wt,
                        /*main_Wt=*/K_tiles,
#endif
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
                dfb_in0.push_back(in0_block_num_tiles);

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
                // write_block_sync_granular_split is more generic (support multiple output tensors)
                // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync_granular should be faster
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
#endif  // FUSE_SWIGLU
#ifdef SRS_FUSE_OP_SIGNALER
                // Signal this core's per-core progress counter right after its prompt block write
                noc.async_write_barrier();
                srs_fuse_signaler.signal_op_per_core(mm_progress_counters_base);
#endif
#endif  // IS_OUTPUT_WRITER
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
