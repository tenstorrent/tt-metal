// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp.
// The main-tree original still serves RingDistributedSDPADeviceOperation (and any other
// consumer not yet on Metal 2.0); this fork is bound by the quasar SDPAProgramFactory
// (ttnn::prim::qsr).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/debug/assert.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "dataflow_common.hpp"
#include "windowed_mask_gen.hpp"

void kernel_main() {
    Noc noc;

    constexpr auto B = get_arg(args::B);
    constexpr auto NQH = get_arg(args::NQH);
    [[maybe_unused]] constexpr auto NKH = get_arg(args::NKH);
    [[maybe_unused]] constexpr auto Sqt = get_arg(args::Sqt);
    constexpr auto valid_Sqt = get_arg(args::valid_Sqt);
    constexpr auto unpadded_Sk = get_arg(args::unpadded_Sk);
    [[maybe_unused]] constexpr auto DHt = get_arg(args::DHt);
    constexpr auto vDHt = get_arg(args::vDHt);
    constexpr auto Sq_chunk_t = get_arg(args::Sq_chunk_t);
    constexpr auto q_num_chunks = get_arg(args::q_num_chunks);
    constexpr auto Sk_chunk_t = get_arg(args::Sk_chunk_t);
    constexpr auto k_num_chunks = get_arg(args::k_num_chunks);
    constexpr auto identity_scalar_packed = get_arg(args::identity_scalar_packed);
    [[maybe_unused]] constexpr auto scale_val = get_arg(args::scale_val);
    constexpr auto num_cores = get_arg(args::num_cores);
    constexpr uint32_t is_causal = get_arg(args::is_causal) == 1;
    constexpr uint32_t use_provided_mask = get_arg(args::use_provided_mask) == 1;
    constexpr uint32_t use_padded_mask = get_arg(args::use_padded_mask) == 1;
    constexpr uint32_t is_chunked = get_arg(args::is_chunked) == 1;
    constexpr uint32_t sliding_window_size = get_arg(args::sliding_window_size);
    constexpr bool use_lightweight_mask = get_arg(args::use_lightweight_mask) == 1;
    constexpr bool use_streaming_compute = get_arg(args::use_streaming_compute) == 1;
    constexpr uint32_t out_subblock_h = get_arg(args::out_subblock_h);
    constexpr uint32_t k_partial_col = get_arg(args::k_partial_col);
    constexpr bool use_zigzag_balancing = get_arg(args::use_zigzag_balancing) == 1;
    constexpr bool use_windowed_mask = get_arg(args::use_windowed_mask) == 1;

    const uint32_t core_id = get_arg(args::core_id);
    const uint32_t num_phases = get_arg(args::num_phases);
    const uint32_t use_chunk_start_idx_tensor = get_arg(args::use_chunk_start_idx_tensor);
    uint32_t chunk_start_t_in_q_chunks_phase_1 = get_arg(args::chunk_start_t_in_q_chunks_phase_1);
    const uint32_t write_offset_phase_1 = get_arg(args::write_offset_phase_1);
    uint32_t chunk_start_t_in_q_chunks_phase_2 = get_arg(args::chunk_start_t_in_q_chunks_phase_2);
    const uint32_t write_offset_phase_2 = get_arg(args::write_offset_phase_2);

    // Global Q scheduling args.
    const uint32_t global_q_start = get_arg(args::global_q_start);
    const uint32_t global_q_count = get_arg(args::global_q_count);
    const uint32_t cu_window_seqlens_eles = get_arg(args::cu_window_seqlens_eles);
    // #54492: global origin of this Q shard. Scalar via a named RTA (windowed builds only); a per-device
    // tensor, when supplied, overrides it at runtime (read in the windowed block below).
    uint32_t q_tok_offset = 0;
#ifdef USE_WINDOWED_MASK
    q_tok_offset = get_arg(args::windowed_q_tok_offset);
#endif

    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;  // non-streaming drain only

    // Named DFB handles. Conditionally-bound buffers alias to a bound placeholder (out) on the paths
    // where the host does not bind them; those paths never touch the buffer (guarded by if constexpr /
    // template no-op), so the placeholder is inert.
    constexpr auto dfb_identity_scale_in = dfb::identity_scale_in;
    constexpr auto dfb_col_identity = dfb::col_identity;
    constexpr auto dfb_out = dfb::out;
#ifdef WRITER_PRODUCES_MASK
    constexpr auto dfb_mask_in = dfb::mask_in;
#else
    constexpr auto dfb_mask_in = dfb::out;  // placeholder; writer generates no mask on this path
#endif
#ifdef USE_WINDOWED_MASK
    constexpr auto dfb_cu_window_in = dfb::cu_window_seqlens;
#else
    constexpr auto dfb_cu_window_in = dfb::out;  // placeholder; windowed generation disabled
#endif
#ifdef FLEXIBLE_CHUNKED
    constexpr auto dfb_chunk_start_idx = dfb::chunk_start_idx_writer;
#else
    constexpr auto dfb_chunk_start_idx = dfb::out;  // placeholder; chunk-start tensor path inactive
#endif
#ifdef WINDOWED_Q_OFFSET_TENSOR
    // Dedicated 1-tile CB for the per-device Q-offset tensor (self-loop; only referenced behind this
    // gate, so no placeholder alias is needed).
    constexpr auto dfb_windowed_q_offset = dfb::windowed_q_offset;
#endif

    constexpr uint32_t tile_bytes = get_tile_size(dfb_out);

    const auto out_writer = TensorAccessor(tensor::out);

    const auto out_tile_shape = TensorTileShape(B, NQH, valid_Sqt, vDHt);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        dfb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    DataflowBuffer dfb_col(dfb_col_identity);
    generate_bcast_col_scalar(dfb_col, identity_scalar_packed);

    // Lightweight mask: generate template tiles once, leave permanently fronted.
    // Sliding layout: [neginf, trailing_primary, leading_prev, leading_current, trailing_next, k_partial?].
    // Non-sliding layout: [neginf, causal_diag?, k_partial?].
    if constexpr (use_lightweight_mask) {
        // is_causal handles K-partial via causal stamp; skip emitting partial tile in causal mode.
        constexpr uint32_t writer_partial_col = is_causal ? 0u : k_partial_col;
        generate_lightweight_mask_tiles<
            writer_partial_col,
            /*joint_l*/ 0u,
            dfb_mask_in,
            is_causal,
            sliding_window_size>(noc);
    }

    // Windowed: load cu_window_seqlens into L1 once; the writer synthesizes the block-diagonal mask per
    // Q chunk from it (so the reader streams Q/K/V only). cu_window_seqlens is bound only in windowed
    // mode, so gate its TensorAccessor + DFB at the preprocessor level.
#ifdef USE_WINDOWED_MASK
    {
        const auto cu_window_reader = TensorAccessor(tensor::cu_window_seqlens);
        constexpr uint32_t cu_tile_bytes = get_tile_size(dfb_cu_window_in);
        DataflowBuffer dfb_cu(dfb_cu_window_in);
        dfb_cu.reserve_back(1);
        noc.async_read(
            cu_window_reader, CoreLocalMem<uint32_t>(dfb_cu.get_write_ptr()), cu_tile_bytes, {.page_id = 0}, {});
        noc.async_read_barrier();
        // Per-device Q origin, if supplied as a tensor: lands in its own dedicated CB (every other CB
        // here has a producer/consumer contract with another kernel that a writer-side reserve would
        // break). Its value overrides the scalar q_tok_offset.
#ifdef WINDOWED_Q_OFFSET_TENSOR
        {
            const auto q_offset_reader = TensorAccessor(tensor::windowed_q_offset);
            DataflowBuffer dfb_off(dfb_windowed_q_offset);
            dfb_off.reserve_back(1);
            const uint32_t off_ptr = dfb_off.get_write_ptr();
            noc.async_read(q_offset_reader, CoreLocalMem<uint32_t>(off_ptr), 4, {.page_id = 0}, {});
            noc.async_read_barrier();
            q_tok_offset = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(off_ptr);
        }
#endif
        // Watcher-build guard for the offset contract the host cannot check in the tensor form: the
        // origin must be tile-aligned and the Q shard must fit inside the K sequence.
        ASSERT(q_tok_offset % tt::constants::TILE_HEIGHT == 0);
        ASSERT(
            q_tok_offset / tt::constants::TILE_HEIGHT + valid_Sqt <=
            (unpadded_Sk + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT);
        dfb_cu.push_back(1);
    }
#endif

    if constexpr (is_chunked) {
        if (use_chunk_start_idx_tensor != 0) {
            DataflowBuffer dfb_chunk_start(dfb_chunk_start_idx);
            dfb_chunk_start.wait_front(1);
            auto chunk_start_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb_chunk_start.get_read_ptr());
            uint32_t chunk_start_idx = chunk_start_ptr[0];
            dfb_chunk_start.pop_front(1);
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

            // Generate masks only for legacy generated-mask variants. No-mask variants do not allocate dfb_mask_in.
            if constexpr (
                !use_provided_mask && !use_lightweight_mask &&
                (is_causal || sliding_window_size > 0 || use_padded_mask)) {
                generate_mask<is_chunked, sliding_window_size, use_padded_mask, dfb_mask_in>(
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
            windowed_generate_if_enabled<use_windowed_mask, dfb_mask_in, dfb_cu_window_in>(
                noc,
                q_chunk,
                Sq_chunk_t,
                Sk_chunk_t,
                valid_Sqt,
                windowed_valid_Skt,
                k_num_chunks,
                cu_window_seqlens_eles,
                q_tok_offset);

            // Determine how many rows of OUT will be written. Both start and end rows are
            // capped by valid_Sqt, since Sq padding is independent of Sk padding.
            const uint32_t out_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
            const uint32_t out_row_end_tile = std::min(out_row_start_tile + Sq_chunk_t, valid_Sqt);
            const uint32_t out_row_tile_count = out_row_end_tile - out_row_start_tile;
            uint32_t out_tile_id = out_tile_shape.id_of(nb, nq, write_offset + out_row_start_tile, 0);
            if constexpr (use_streaming_compute) {
                // Streaming: drain per row-group (dfb_out is a 2-slot ping-pong).
                // Compute always pushes Sq_chunk_t rows; rows past out_row_tile_count
                // are padding and get popped without being written.
                write_block_row_grouped(
                    noc,
                    out_writer,
                    dfb_out,
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
                    dfb_out,
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
