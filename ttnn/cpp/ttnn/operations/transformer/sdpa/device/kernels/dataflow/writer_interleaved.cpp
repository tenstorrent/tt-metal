// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "dataflow_common.hpp"
#include "windowed_mask_gen.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);
    constexpr uint32_t valid_Sqt = get_compile_time_arg_val(4);
    constexpr uint32_t unpadded_Sk = get_compile_time_arg_val(5);
    constexpr uint32_t DHt = get_compile_time_arg_val(6);
    constexpr uint32_t vDHt = get_compile_time_arg_val(7);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(10);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(12);
    constexpr uint32_t scale_val = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores = get_compile_time_arg_val(14);
    constexpr uint32_t is_causal = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t use_provided_mask = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(19);
    constexpr bool use_lightweight_mask = get_compile_time_arg_val(20) == 1;
    constexpr bool use_streaming_compute = get_compile_time_arg_val(21) == 1;
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(22);
    constexpr uint32_t k_partial_col = get_compile_time_arg_val(23);
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(24) == 1;
    // Windowed (block-diagonal) mask generation flags. Fixed scalar slots BEFORE the tensor-accessor
    // block so the accessor offset chain stays intact for all configs.
    constexpr bool use_windowed_mask = get_compile_time_arg_val(25) == 1;
    // Fused-gather variant of 3D-neighborhood: mask is generated over the reader's dense-packed keys
    // (packed_to_flat) instead of the active-tile real positions. Build-out in progress.
    constexpr bool neighborhood_gather = get_compile_time_arg_val(26) == 1;

    // out accessor, then the cu_window accessor chained immediately after it (before the CB-id block).
    constexpr auto out_args = TensorAccessorArgs<27>();
    constexpr auto cu_window_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    // Per-device Q offset accessor, chained after cu_window so the offset chain stays intact.
    constexpr auto q_offset_args = TensorAccessorArgs<cu_window_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_id = get_arg_val<uint32_t>(1);
    const uint32_t num_phases = get_arg_val<uint32_t>(2);
    const uint32_t use_chunk_start_idx_tensor = get_arg_val<uint32_t>(3);
    uint32_t chunk_start_t_in_q_chunks_phase_1 = get_arg_val<uint32_t>(4);
    const uint32_t write_offset_phase_1 = get_arg_val<uint32_t>(5);
    uint32_t chunk_start_t_in_q_chunks_phase_2 = 0;
    uint32_t write_offset_phase_2 = 0;
    if (num_phases == 2) {
        chunk_start_t_in_q_chunks_phase_2 = get_arg_val<uint32_t>(6);
        write_offset_phase_2 = get_arg_val<uint32_t>(7);
    }

    // Global Q scheduling args follow phase_2 args.
    const uint32_t global_q_start = get_arg_val<uint32_t>(8);
    const uint32_t global_q_count = get_arg_val<uint32_t>(9);
    const uint32_t cu_window_seqlens_addr = get_arg_val<uint32_t>(10);
    const uint32_t cu_window_seqlens_eles = get_arg_val<uint32_t>(11);
    // Global row index of this tensor's first Q row. Non-zero when Q is a sequence-parallel shard:
    // Q/output are addressed locally, but cu_window_seqlens and K/V are global, so the mask generator
    // needs the shard's global origin. Zero for an unsharded Q, which is the only case today.
    uint32_t q_tok_offset = get_arg_val<uint32_t>(12);
    // Per-device form: when a tensor was supplied its value wins, read below once cb_cu_window_in is
    // available. Zero address means the caller used the scalar above.
    const uint32_t q_tok_offset_addr = get_arg_val<uint32_t>(13);
    // 3D-neighborhood descriptor {T,H,W,kt,kh,kw}; T==0 selects the block-diagonal (cu_window) path.
    const uint32_t nb_T = get_arg_val<uint32_t>(14);
    const uint32_t nb_H = get_arg_val<uint32_t>(15);
    const uint32_t nb_W = get_arg_val<uint32_t>(16);
    const uint32_t nb_kt = get_arg_val<uint32_t>(17);
    const uint32_t nb_kh = get_arg_val<uint32_t>(18);
    const uint32_t nb_kw = get_arg_val<uint32_t>(19);
    // Spatial-SP over W: full width + this shard's global W origin (signed). nb_W_full == 0 => not
    // W-sharded (mask uses local == global W).
    const uint32_t nb_W_full = get_arg_val<uint32_t>(20);
    const int32_t nb_w_origin = static_cast<int32_t>(get_arg_val<uint32_t>(21));
    // neighborhood_gather host-upload masks: when set the READER DMAs the mask from the uploaded pool,
    // so the writer skips its on-device neighborhood mask generation entirely.
    const uint32_t neighborhood_mask_provided = get_arg_val<uint32_t>(22);

    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;  // non-streaming drain only

    constexpr uint32_t cb_arg_offset = q_offset_args.next_compile_time_args_offset();
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_chunk_start_idx = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 4);
    // cu_window CB id lives in the CB-id block (appended by CBIds for windowed mode; inactive otherwise).
    constexpr uint32_t cb_cu_window_in = get_compile_time_arg_val(cb_arg_offset + 5);
    // Dedicated 1-tile CB for the per-device Q-offset tensor; allocated only when that tensor is passed
    // (q_in otherwise, same fallback rule as cb_cu_window_in), and only touched behind the runtime guard.
    constexpr uint32_t cb_windowed_q_offset = get_compile_time_arg_val(cb_arg_offset + 6);

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);

    const auto out_writer = TensorAccessor(out_args, out_addr);

    const auto out_tile_shape = TensorTileShape(B, NQH, valid_Sqt, vDHt);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);

    // Lightweight mask: generate template tiles once, leave permanently fronted.
    // Sliding layout: [neginf, trailing_primary, leading_prev, leading_current, trailing_next, k_partial?].
    // Non-sliding layout: [neginf, causal_diag?, k_partial?].
    if constexpr (use_lightweight_mask) {
        // is_causal handles K-partial via causal stamp; skip emitting partial tile in causal mode.
        constexpr uint32_t writer_partial_col = is_causal ? 0u : k_partial_col;
        generate_lightweight_mask_tiles<
            writer_partial_col,
            /*joint_l*/ 0u,
            cb_mask_in,
            is_causal,
            sliding_window_size>(noc);
    }

    // Windowed setup. The per-device Q origin (if passed as a tensor) lands in its OWN dedicated CB
    // and is read in both sub-modes -- 3D-neighborhood uses it for SP-over-T. The cu_window array load
    // is block-diagonal only (3D has no cu tensor; cu_window_seqlens_addr is 0 and cb_cu_window_in is
    // not dedicated there, so touching it would read null and desync a shared CB).
    if constexpr (use_windowed_mask) {
        if (q_tok_offset_addr != 0) {
            const auto q_offset_reader = TensorAccessor(q_offset_args, q_tok_offset_addr);
            CircularBuffer cb_off(cb_windowed_q_offset);
            cb_off.reserve_back(1);
            const uint32_t off_ptr = cb_off.get_write_ptr();
            noc.async_read(q_offset_reader, CoreLocalMem<uint32_t>(off_ptr), 4, {.page_id = 0}, {});
            noc.async_read_barrier();
            q_tok_offset = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(off_ptr);
        }
        if (nb_T == 0) {
            const auto cu_window_reader = TensorAccessor(cu_window_args, cu_window_seqlens_addr);
            constexpr uint32_t cu_tile_bytes = get_tile_size(cb_cu_window_in);
            CircularBuffer cb_cu(cb_cu_window_in);
            cb_cu.reserve_back(1);
            noc.async_read(
                cu_window_reader, CoreLocalMem<uint32_t>(cb_cu.get_write_ptr()), cu_tile_bytes, {.page_id = 0}, {});
            noc.async_read_barrier();
            cb_cu.push_back(1);
        }
    }

    if constexpr (is_chunked) {
        if (use_chunk_start_idx_tensor != 0) {
            CircularBuffer cb_chunk_start(cb_chunk_start_idx);
            cb_chunk_start.wait_front(1);
            auto chunk_start_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_chunk_start.get_read_ptr());
            uint32_t chunk_start_idx = chunk_start_ptr[0];
            cb_chunk_start.pop_front(1);
            const uint32_t q_chunk_size = Sq_chunk_t * tt::constants::TILE_HEIGHT;
            chunk_start_t_in_q_chunks_phase_1 = chunk_start_idx / q_chunk_size;
            if (num_phases == 2) {
                chunk_start_t_in_q_chunks_phase_2 = chunk_start_t_in_q_chunks_phase_1;
            }
        }
    }

    uint32_t chunk_start_t_in_q_chunks = 0;
    uint32_t write_offset = 0;
    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        if (phase == 0) {
            chunk_start_t_in_q_chunks = chunk_start_t_in_q_chunks_phase_1;
            write_offset = write_offset_phase_1;
        } else {
            chunk_start_t_in_q_chunks = chunk_start_t_in_q_chunks_phase_2;
            write_offset = write_offset_phase_2;
        }
        for (uint32_t global_q_iter = 0; global_q_iter < global_q_count; ++global_q_iter) {
            const auto decoded =
                decompose_global_q_index(global_q_start + global_q_iter, q_num_chunks, NQH, use_zigzag_balancing);
            const uint32_t nb = decoded.nb;
            const uint32_t nq = decoded.nq;
            const uint32_t q_chunk = decoded.q_chunk;

            // Generate masks only for legacy generated-mask variants. No-mask variants do not allocate cb_mask_in.
            if constexpr (
                !use_provided_mask && !use_lightweight_mask &&
                (is_causal || sliding_window_size > 0 || use_padded_mask)) {
                generate_mask<is_chunked, sliding_window_size, use_padded_mask, cb_mask_in>(
                    noc,
                    Sq_chunk_t,
                    Sk_chunk_t,
                    q_chunk,
                    chunk_start_t_in_q_chunks,
                    true,
                    false,
                    unpadded_Sk,
                    0,
                    is_causal);
            }

            // Windowed: synthesize this Q chunk's block-diagonal mask (all K chunks) before draining its
            // output. The call is a template wrapper that only instantiates the generator when
            // use_windowed_mask is true (kernel_main is not a template, so a bare `if constexpr` here
            // would still compile the discarded body). valid_Skt derived from the unpadded K length.
            constexpr uint32_t windowed_valid_Skt =
                (unpadded_Sk + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
            // Fused-gather: mask over the reader's DENSE packed keys (n_packed_chunks chunks). Only one
            // of these two generates -- the streamed path is disabled in gather mode and vice versa. When
            // the host uploaded the mask pool, the reader DMAs it instead and this is skipped at runtime.
            if (!neighborhood_mask_provided) {
                neighborhood_gather_generate_if_enabled<neighborhood_gather, cb_mask_in>(
                    noc,
                    q_chunk,
                    Sq_chunk_t,
                    Sk_chunk_t,
                    valid_Sqt,
                    q_tok_offset,
                    nb_T,
                    nb_H,
                    nb_W,
                    nb_kt,
                    nb_kh,
                    nb_kw,
                    nb_W_full,
                    nb_w_origin);
            }
            windowed_generate_if_enabled<use_windowed_mask && !neighborhood_gather, cb_mask_in, cb_cu_window_in>(
                noc,
                q_chunk,
                Sq_chunk_t,
                Sk_chunk_t,
                valid_Sqt,
                windowed_valid_Skt,
                k_num_chunks,
                cu_window_seqlens_eles,
                q_tok_offset,
                nb_T,
                nb_H,
                nb_W,
                nb_kt,
                nb_kh,
                nb_kw,
                nb_W_full,
                nb_w_origin);

            // Determine how many rows of OUT will be written. Both start and end rows are
            // capped by valid_Sqt, since Sq padding is independent of Sk padding.
            const uint32_t out_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
            const uint32_t out_row_end_tile = std::min(out_row_start_tile + Sq_chunk_t, valid_Sqt);
            const uint32_t out_row_tile_count = out_row_end_tile - out_row_start_tile;
            uint32_t out_tile_id = out_tile_shape.id_of(nb, nq, write_offset + out_row_start_tile, 0);
            if constexpr (use_streaming_compute) {
                // Streaming: drain per row-group (cb_out is a 2-slot ping-pong).
                // Compute always pushes Sq_chunk_t rows; rows past out_row_tile_count
                // are padding and get popped without being written.
                write_block_row_grouped(
                    noc,
                    out_writer,
                    cb_out,
                    Sq_chunk_t,
                    out_row_tile_count,
                    vDHt,
                    out_tile_id,
                    tile_bytes,
                    out_subblock_h,
                    barrier_threshold);
            } else {
                write_block(
                    noc,
                    out_writer,
                    cb_out,
                    out_chunk_tiles,
                    out_row_tile_count,
                    vDHt,
                    out_tile_id,
                    tile_bytes,
                    barrier_threshold);
            }
        }
    }  // close phase
}
