// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa writer: per token, drain the untilized [H, V_DIM] row-major block from compute (Sqt*vDHt
// tile-sized pages, row-major bytes inside) and write each head-row to the ROW-MAJOR [1,H,S,V_DIM] output.
// Also builds the three persistent compute-input tiles once at startup (the reader is busier): the reduce
// identity scaler, the col-identity (row-sum finalize), and the all-(-inf) mask tile. Optional per-head
// attention sinks are gathered once here too, while the reader loads Q and indices.
// Uses the object-based NoC + circular-buffer APIs.

#include <stdint.h>
#include "api/tile_constants.h"  // tt::constants::TILE_WIDTH
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_gather.hpp"  // dual-NoC trid-ring gather helper (shared with the reader)

constexpr uint32_t one_bf16_packed = 0x3F803F80u;  // bf16(1.0) double-packed; generate_bcast_col_scalar uses >>16
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::H);
    constexpr uint32_t S = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::S);
    constexpr uint32_t vDHt = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::V_DHT);
    // CB ids — passed by the factory (the SparseCB enum is the single source); order must match the
    // factory's writer CB-arg block (compile args 3..6).
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_OUT_RM);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_SCALE);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_COL_IDENTITY);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_NEGINF);
    constexpr uint32_t out_elem_bytes =
        get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::OUT_ELEM_BYTES);  // output tensor dtype
    constexpr bool block_cyclic = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BLOCK_CYCLIC) != 0;
    constexpr uint32_t bc_chunk_local = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BC_CHUNK_LOCAL);
    constexpr uint32_t bc_sp = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BC_SP);
    constexpr uint32_t bc_shard_stride_gap = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BC_SHARD_STRIDE_GAP);
    constexpr uint32_t bc_slab_stride_gap = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BC_SLAB_STRIDE_GAP);
    constexpr bool scaled_kv = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::SCALED_KV) != 0;
    constexpr uint32_t k_dim = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::K_DIM);
    constexpr uint32_t kv_elem_bytes = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::KV_ELEM_BYTES);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_IDX);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_KREQ);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_KACK);
    constexpr uint32_t packed_row_bytes = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::PACKED_ROW_BYTES);
    constexpr bool use_attention_sink = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::USE_ATTENTION_SINK) != 0;
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_ATTENTION_SINK);
    constexpr uint32_t cb_sink_scratch = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_SINK_SCRATCH);
    constexpr bool split_k = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::SPLIT_K) != 0;
    constexpr uint32_t cb_send_out = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_SEND_OUT);
    constexpr uint32_t cb_send_max = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_SEND_MAX);
    constexpr uint32_t cb_send_sum = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_SEND_SUM);
    constexpr uint32_t cb_recv_out = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_RECV_OUT);
    constexpr uint32_t cb_recv_max = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_RECV_MAX);
    constexpr uint32_t cb_recv_sum = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_RECV_SUM);
    constexpr uint32_t cb_merge_ctrl = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_MERGE_CTRL);
    constexpr auto out_args =
        TensorAccessorArgs<sparse_sdpa::writer_ct_arg::END, sparse_sdpa::writer_common_arg::END>();
    // kv carries a RUNTIME tensor shape (T dim in common runtime args), so its accessor spans both arg streams.
    constexpr auto kv_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::OUTPUT_ADDRESS);
    const uint32_t tok_start = get_arg_val<uint32_t>(sparse_sdpa::split_rt_arg::TOK_START);
    const uint32_t tok_count = get_arg_val<uint32_t>(sparse_sdpa::split_rt_arg::TOK_COUNT);
    const uint32_t head_c0 = get_arg_val<uint32_t>(sparse_sdpa::split_rt_arg::HEAD_C0);
    const uint32_t tail_c1 = get_arg_val<uint32_t>(sparse_sdpa::split_rt_arg::TAIL_C1);
    const uint32_t prev_noc_x = get_arg_val<uint32_t>(sparse_sdpa::split_rt_arg::PREV_NOC_X);
    const uint32_t prev_noc_y = get_arg_val<uint32_t>(sparse_sdpa::split_rt_arg::PREV_NOC_Y);
    const uint32_t kv_addr =
        get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::KV_ADDRESS);  // dual-NoC K-half gather
    // Indexed KV cache: page offset (cache_batch_idx * T) selecting the cache's batch slot; 0 if not indexed.
    const uint32_t kv_batch_page_offset =
        get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::KV_BATCH_PAGE_OFFSET);

    constexpr uint32_t row_bytes = vDHt * tt::constants::TILE_WIDTH * out_elem_bytes;  // V_DIM bytes per head-row
    constexpr uint32_t block_tiles = (H / tt::constants::TILE_HEIGHT) * vDHt;          // Sqt*vDHt: the [H,V_DIM] block

    Noc noc;
    experimental::CB out_cb(cb_out_rm);
    const auto out = TensorAccessor(out_args, out_addr);
    // Dual-NoC K gather: the writer co-gathers half of each K chunk on ITS NoC into the shared cb_k_rm L1
    // (the reader owns the reserve/push; we fill rows [0,half) at the same write ptr). This spreads the
    // K response-data load over both NoC links. Indices live in the reader's shared cb_idx
    // scratch; the kv accessor + row size mirror the reader.
    constexpr uint32_t k_row_bytes = k_dim * kv_elem_bytes;
    constexpr uint32_t latent_row_bytes = vDHt * tt::constants::TILE_WIDTH;
    constexpr uint32_t scale_blocks = sparse_sdpa::scale_block_count(latent_row_bytes);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    // Optional DRAM spill of kv (see the reader); chained after the attention-sink accessor.
    constexpr auto sink_args =
        TensorAccessorArgs<kv_args.next_compile_time_args_offset(), kv_args.next_common_runtime_args_offset()>();
    constexpr auto spill_args =
        TensorAccessorArgs<sink_args.next_compile_time_args_offset(), sink_args.next_common_runtime_args_offset()>();
    const auto kv_spill =
        TensorAccessor(spill_args, get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::KV_SPILL_ADDRESS));
    const sparse_sdpa::KvSpill kv_spill_rows{
        get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::KV_FRONT_PAGES),
        get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::KV_SPILL_ROWS),
        get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::KV_SPILL_ROW_STRIDE)};
    experimental::CB idx_cb(cb_idx), kreq_cb(cb_kreq), kack_cb(cb_kack);
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());

    // --- persistent compute-input tiles (built once; the reader is busier with the K gather) ---
    // Reduce identity scaler (value 1.0; the softmax scale is applied in compute's exp).
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        /*reduce_factor=*/1>();

    // Col-identity (column 0 = 1.0): compute matmul-reduces the partial row-sum against it to finalize the
    // within-tile reduction in normalize_row_streaming.
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), one_bf16_packed);

    // All-(-inf) tile (row 0; compute bcast-adds it to fully-masked key tiles). Only row 0 matters for the
    // row-broadcast add: the left and right half-rows live in face 0 and face 1 respectively.
    {
        experimental::CB neginf_cb(cb_neginf);
        neginf_cb.reserve_back(1);
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(neginf_cb.get_write_ptr());
        for (uint32_t c = 0; c < tt::constants::FACE_WIDTH; ++c) {
            p[c] = neg_inf_bf16;
            p[tt::constants::FACE_HW + c] = neg_inf_bf16;
        }
        neginf_cb.push_back(1);
    }

    if constexpr (use_attention_sink) {
        if (tok_count > 0) {
            const auto sink = TensorAccessor(
                sink_args, get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::ATTENTION_SINK_ADDRESS));
            experimental::CB sink_cb(cb_attention_sink), scratch(cb_sink_scratch);
            constexpr uint32_t sink_bytes = H * sizeof(uint16_t);
            constexpr uint32_t tile_bytes = tt::constants::TILE_HW * sizeof(uint16_t);
            constexpr uint32_t faces_per_row = tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH;
            static_assert(H % tt::constants::TILE_HEIGHT == 0);
            static_assert(sink_bytes % NOC_DRAM_READ_ALIGNMENT_BYTES == 0);
            constexpr uint32_t sink_tiles = H / tt::constants::TILE_HEIGHT;
            sink_cb.reserve_back(sink_tiles);
            scratch.reserve_back(1);
            noc.async_write_zeros(sink_cb, sink_tiles * tile_bytes);
            // [1,1,1,H] BF16 stores all heads in one row. Read once, then pack the scalars
            // into the first column of the compute tiles (one head per row).
            noc.async_read(sink, scratch, sink_bytes, {.page_id = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            noc.write_zeros_l1_barrier();
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sink_cb.get_write_ptr());
            auto* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch.get_write_ptr());
            for (uint32_t h = 0; h < H; ++h) {
                const uint32_t row = h % tt::constants::TILE_HEIGHT;
                const uint32_t offset = (h / tt::constants::TILE_HEIGHT) * tt::constants::TILE_HW +
                                        (row / tt::constants::FACE_HEIGHT) * faces_per_row * tt::constants::FACE_HW +
                                        (row % tt::constants::FACE_HEIGHT) * tt::constants::FACE_WIDTH;
                dst[offset] = src[h];
            }
            sink_cb.push_back(sink_tiles);  // reused by every token on this core
        }
    }

    // split-K: the neighbour (core i+1) NoC-writes its HEAD state straight into our cb_recv_* bases, which are the
    // same L1 addresses on every core (identical CB layout); capture them before any local FIFO movement.
    experimental::CB recv_out_cb(cb_recv_out), recv_max_cb(cb_recv_max), recv_sum_cb(cb_recv_sum);
    experimental::CB send_out_cb(cb_send_out), send_max_cb(cb_send_max), send_sum_cb(cb_send_sum);
    experimental::CB merge_ctrl_cb(cb_merge_ctrl);
    constexpr uint32_t sqt = H / tt::constants::TILE_HEIGHT;
    constexpr uint32_t state_tile_bytes = tt::constants::TILE_HW * sizeof(uint16_t);  // bf16 flash state
    uint32_t recv_out_l1 = 0, recv_max_l1 = 0, recv_sum_l1 = 0;
    if constexpr (split_k) {
        recv_out_l1 = recv_out_cb.get_write_ptr();
        recv_max_l1 = recv_max_cb.get_write_ptr();
        recv_sum_l1 = recv_sum_cb.get_write_ptr();
    }
    const uint32_t head_segs = head_c0 != 0 ? 1u : 0u;
    const uint32_t n_segs = head_segs + tok_count + (tail_c1 != 0 ? 1u : 0u);
    // Output rows of a finished segment are written lazily (polled between gather requests), so the next
    // segment's K gather is not held behind the previous segment's compute tail + untilize.
    constexpr uint32_t no_tok = 0xFFFFFFFFu;
    uint32_t pending_tok = no_tok;
    auto write_out = [&](uint32_t t) {
        out_cb.wait_front(block_tiles);
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_write(out_cb, out, row_bytes, {.offset_bytes = h * row_bytes}, {.page_id = h * S + t});
        }
        noc.async_write_barrier();
        out_cb.pop_front(block_tiles);
    };
    for (uint32_t seg = 0; seg < n_segs; ++seg) {
        const bool is_head = seg < head_segs;
        const uint32_t f = seg - head_segs;
        const uint32_t tok = is_head ? tok_start - 1 : tok_start + f;
        const bool is_tail = !is_head && f >= tok_count;
        // K-gather phase: process this segment's active chunks (reader signals per slab; is_last ends it).
        // Gather rows [0,split) on the writer's NoC into the reader-reserved destination, then ack.
        bool last = false;
        bool empty = false;
        while (!last) {
            if (!(sparse_sdpa::dbg::MASK & sparse_sdpa::dbg::EAGER_OUT) && pending_tok != no_tok &&
                cb_pages_available_at_front(cb_out_rm, block_tiles)) {
                write_out(pending_tok);
                pending_tok = no_tok;
            }
            kreq_cb.wait_front(1);
            uint32_t base, split, dst_l1, scale_dst_l1 = 0;
            {
                volatile tt_l1_ptr uint32_t* request =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                base = request[sparse_sdpa::gather_request::BASE];
                split = request[sparse_sdpa::gather_request::SPLIT];
                last = request[sparse_sdpa::gather_request::IS_LAST] != 0;
                dst_l1 = request[sparse_sdpa::gather_request::DST_L1];
                empty = request[sparse_sdpa::gather_request::EMPTY] != 0;
                if constexpr (scaled_kv) {
                    scale_dst_l1 = request[sparse_sdpa::gather_request::SCALE_DST_L1];
                }
            }
            kreq_cb.pop_front(1);
            if (empty) {
                break;  // empty HEAD segment: no gather, no ack
            }
            if constexpr (scaled_kv) {
                sparse_sdpa::trid_ring_gather<
                    block_cyclic,
                    bc_chunk_local,
                    bc_sp,
                    bc_shard_stride_gap,
                    bc_slab_stride_gap,
                    sparse_sdpa::SCALED_K_TRID_RING>(
                    noc,
                    kv,
                    kv_spill,
                    kv_spill_rows,
                    dst_l1,
                    idx_ptr,
                    base,
                    0,
                    split,
                    packed_row_bytes,
                    kv_batch_page_offset,
                    sparse_sdpa::SCALED_K_READ_BYTES);
                sparse_sdpa::scatter_packed_scales<scale_blocks>(
                    dst_l1, scale_dst_l1, 0, split, packed_row_bytes, latent_row_bytes);
            } else {
                sparse_sdpa::trid_ring_gather<
                    block_cyclic,
                    bc_chunk_local,
                    bc_sp,
                    bc_shard_stride_gap,
                    bc_slab_stride_gap,
                    sparse_sdpa::K_TRID_RING>(
                    noc,
                    kv,
                    kv_spill,
                    kv_spill_rows,
                    dst_l1,
                    idx_ptr,
                    base,
                    0,
                    split,
                    k_row_bytes,
                    kv_batch_page_offset);
            }
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
        }
        if constexpr (split_k) {
            if (is_head) {
                // Ship this HEAD's unnormalized state to core i-1 (which owns the token's first chunks).
                const uint32_t sem_l1 = get_semaphore(0);
                const uint64_t prev_sem = get_noc_addr(prev_noc_x, prev_noc_y, sem_l1);
                if (empty) {
                    noc_semaphore_inc(prev_sem, sparse_sdpa::SPLIT_PARTIAL_EMPTY);
                } else {
                    send_out_cb.wait_front(sqt * vDHt);
                    send_max_cb.wait_front(sqt);
                    send_sum_cb.wait_front(sqt);
                    noc_async_write(
                        send_out_cb.get_read_ptr(),
                        get_noc_addr(prev_noc_x, prev_noc_y, recv_out_l1),
                        sqt * vDHt * state_tile_bytes);
                    noc_async_write(
                        send_max_cb.get_read_ptr(), get_noc_addr(prev_noc_x, prev_noc_y, recv_max_l1), sqt * state_tile_bytes);
                    noc_async_write(
                        send_sum_cb.get_read_ptr(), get_noc_addr(prev_noc_x, prev_noc_y, recv_sum_l1), sqt * state_tile_bytes);
                    noc_async_write_barrier();
                    noc_semaphore_inc(prev_sem, sparse_sdpa::SPLIT_PARTIAL_VALID);
                    send_out_cb.pop_front(sqt * vDHt);
                    send_max_cb.pop_front(sqt);
                    send_sum_cb.pop_front(sqt);
                }
                noc_async_atomic_barrier();
                continue;  // core i-1 writes this token's output
            }
            if (is_tail) {
                // Wait for core i+1's HEAD state (it ran first on that core, so this rarely blocks).
                volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
                noc_semaphore_wait_min(sem, 1);
                const uint32_t v = *sem;
                noc_semaphore_set(sem, 0);
                merge_ctrl_cb.reserve_back(1);
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(merge_ctrl_cb.get_write_ptr())[0] = v;
                merge_ctrl_cb.push_back(1);
                if (v == sparse_sdpa::SPLIT_PARTIAL_VALID) {
                    recv_out_cb.reserve_back(sqt * vDHt);
                    recv_out_cb.push_back(sqt * vDHt);
                    recv_max_cb.reserve_back(sqt);
                    recv_max_cb.push_back(sqt);
                    recv_sum_cb.reserve_back(sqt);
                    recv_sum_cb.push_back(sqt);
                }
            }
        }
        // One untilized [H, V_DIM] block per FULL/TAIL segment; row h at byte offset h*row_bytes -> page h*S + tok.
        if (pending_tok != no_tok) {
            write_out(pending_tok);  // previous segment's block (compute is long past it by now)
        }
        pending_tok = tok;
        if constexpr (sparse_sdpa::dbg::MASK & sparse_sdpa::dbg::EAGER_OUT) {
            write_out(pending_tok);
            pending_tok = no_tok;
        }
    }
    if (pending_tok != no_tok) {
        write_out(pending_tok);
    }
}
