// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa writer: per token and head pass, drain the untilized row-major output rows from compute (vDHt tile-sized
// pages per tile row of heads, row-major bytes inside) and write each head-row to the ROW-MAJOR [1,H,S,V_DIM] output.
// Also builds the persistent recipe tiles once at startup (the reader is busier): the reduce scaler and the column
// identity, and the per-row attention sink tiles. Split-K: sends a HEAD segment's recipe state to core i-1 (compute's
// checkpoint request) and hands a TAIL segment the neighbour's state. Uses the object-based NoC + circular-buffer APIs.

#include <stdint.h>
#include "api/tile_constants.h"  // tt::constants::TILE_WIDTH
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar_zeroed
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/operations/transformer/sdpa/device/kernels/recipe_ring_cbs.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/recipe_state_layout.hpp"
#include "sparse_sdpa_gather.hpp"  // dual-NoC trid-ring gather helper (shared with the reader)

void kernel_main() {
    namespace cb = sparse_sdpa::recipe_cb;
    constexpr uint32_t H = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::H);
    constexpr uint32_t S = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::S);
    constexpr uint32_t vDHt = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::V_DHT);
    constexpr uint32_t out_elem_bytes =
        get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::OUT_ELEM_BYTES);  // output tensor dtype
    constexpr bool block_cyclic = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BLOCK_CYCLIC) != 0;
    constexpr uint32_t bc_chunk_local = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BC_CHUNK_LOCAL);
    constexpr uint32_t bc_sp = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BC_SP);
    constexpr uint32_t bc_shard_stride_gap = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BC_SHARD_STRIDE_GAP);
    constexpr uint32_t bc_slab_stride_gap = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BC_SLAB_STRIDE_GAP);
    constexpr bool scaled_kv = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::SCALED_KV) != 0;
    constexpr bool fused_dequant = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::FUSED_DEQUANT) != 0;
    constexpr uint32_t k_dim = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::K_DIM);
    constexpr uint32_t kv_elem_bytes = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::KV_ELEM_BYTES);
    constexpr uint32_t packed_row_bytes = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::PACKED_ROW_BYTES);
    constexpr bool use_attention_sink = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::USE_ATTENTION_SINK) != 0;
    constexpr bool split_k = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::SPLIT_K) != 0;
    constexpr uint32_t q_tiles = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::Q_TILES);
    constexpr uint32_t passes = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::PASSES);
    constexpr bool fp32_state = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::FP32_STATE) != 0;
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

    constexpr uint32_t head_tiles = H / tt::constants::TILE_HEIGHT;
    constexpr uint32_t row_bytes = vDHt * tt::constants::TILE_WIDTH * out_elem_bytes;  // V_DIM bytes per head-row

    Noc noc;
    experimental::CB out_cb(cb::out_rm);
    const auto out = TensorAccessor(out_args, out_addr);
    // Dual-NoC K gather: the writer co-gathers half of each K chunk on ITS NoC into the shared k_rm L1
    // (the reader owns the reserve/push; we fill rows [0,half) at the same write ptr). This spreads the
    // K response-data load over both NoC links. Indices live in the reader's shared idx
    // scratch; the kv accessor + row size mirror the reader.
    constexpr uint32_t k_row_bytes = k_dim * kv_elem_bytes;
    constexpr uint32_t latent_row_bytes = vDHt * tt::constants::TILE_WIDTH;
    constexpr uint32_t scale_blocks = sparse_sdpa::scale_block_count(latent_row_bytes);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    experimental::CB idx_cb(cb::idx), kreq_cb(cb::kreq), kack_cb(cb::kack);
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());

    // --- persistent recipe tiles (built once; the reader is busier with the K gather) ---
    // The recipe's reduce scaler and column identity (as the dense recipe reader, reader_recipe.cpp).
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        recipe_ring_cb::identity_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar_zeroed(CircularBuffer(recipe_ring_cb::col_identity), 0x3f803f80);

    if constexpr (use_attention_sink) {
        constexpr auto sink_args =
            TensorAccessorArgs<kv_args.next_compile_time_args_offset(), kv_args.next_common_runtime_args_offset()>();
        const auto sink = TensorAccessor(
            sink_args, get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::ATTENTION_SINK_ADDRESS));
        experimental::CB sink_cb(cb::sink), scratch(cb::sink_scratch);
        constexpr uint32_t sink_bytes = H * sizeof(uint16_t);
        constexpr uint32_t tile_bytes = tt::constants::TILE_HW * sizeof(uint16_t);
        constexpr uint32_t faces_per_row = tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH;
        static_assert(H % tt::constants::TILE_HEIGHT == 0);
        static_assert(sink_bytes % NOC_DRAM_READ_ALIGNMENT_BYTES == 0);
        // One tile per Q tile row of every pass (padding rows' tiles stay zero), fronted for compute's whole run.
        constexpr uint32_t sink_tiles = passes * q_tiles;
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

    // split-K: the neighbour (core i+1) NoC-writes its HEAD state straight into our recv CBs' bases, which are the same
    // L1 addresses on every core (identical CB layout); capture them before any local FIFO movement.
    using Transfer = sdpa::streaming::StateTransfer;
    constexpr uint32_t max_bytes = fp32_state ? 4096 : 2048;
    experimental::CB recv_out_cb(cb::recv_out), recv_max_cb(cb::recv_max), recv_sum_cb(cb::recv_sum);
    experimental::CB merge_ctrl_cb(cb::merge_ctrl);
    uint32_t recv_l1[3] = {0, 0, 0};
    if constexpr (split_k) {
        recv_l1[0] = recv_out_cb.get_write_ptr();
        recv_l1[1] = recv_max_cb.get_write_ptr();
        recv_l1[2] = recv_sum_cb.get_write_ptr();
    }
    const uint32_t head_segs = head_c0 != 0 ? 1u : 0u;
    const uint32_t n_segs = head_segs + tok_count + (tail_c1 != 0 ? 1u : 0u);
    for (uint32_t seg = 0; seg < n_segs; ++seg) {
        const bool is_head = seg < head_segs;
        const uint32_t f = seg - head_segs;
        const uint32_t tok = is_head ? tok_start - 1 : tok_start + f;
        const bool is_tail = !is_head && f >= tok_count;
        for (uint32_t pass = 0; pass < passes; ++pass) {
            // K-gather phase: this pass's active chunks (reader signals per slab; is_last ends the pass). Gather rows
            // [0,split) on the writer's NoC into the reader-reserved destination, then ack.
            bool last = false;
            bool empty = false;
            while (!last) {
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
                        noc, kv, dst_l1, idx_ptr, base, 0, split, packed_row_bytes, kv_batch_page_offset);
                    sparse_sdpa::scatter_packed_scales<scale_blocks, fused_dequant>(
                        dst_l1, scale_dst_l1, 0, split, packed_row_bytes, latent_row_bytes);
                } else {
                    sparse_sdpa::trid_ring_gather<
                        block_cyclic,
                        bc_chunk_local,
                        bc_sp,
                        bc_shard_stride_gap,
                        bc_slab_stride_gap,
                        sparse_sdpa::K_TRID_RING>(
                        noc, kv, dst_l1, idx_ptr, base, 0, split, k_row_bytes, kv_batch_page_offset);
                }
                kack_cb.reserve_back(1);
                kack_cb.push_back(1);
            }
            if constexpr (split_k) {
                if (is_head) {
                    // Ship this HEAD's unnormalized state to core i-1 (which owns the token's first chunks): compute's
                    // checkpoint request names its numerator, maxima and denominator CBs, at their bases (recipe state
                    // banks hold exactly one block, so their FIFO pointers are at the origin between segments).
                    const uint32_t sem_l1 = get_semaphore(0);
                    const uint64_t prev_sem = get_noc_addr(prev_noc_x, prev_noc_y, sem_l1);
                    if (empty) {
                        noc_semaphore_inc(prev_sem, sparse_sdpa::SPLIT_PARTIAL_EMPTY);
                    } else {
                        CircularBuffer request(recipe_ring_cb::checkpoint_request), ack(recipe_ring_cb::checkpoint_ack);
                        request.wait_front(1);
                        ack.reserve_back(1);
                        const auto* words = reinterpret_cast<const volatile uint32_t*>(request.get_read_ptr());
                        for (uint32_t plane = 0; plane < 3; ++plane) {
                            const uint32_t src = CircularBuffer(words[Transfer::Numerator + plane]).get_read_ptr();
                            const uint64_t dst = get_noc_addr(prev_noc_x, prev_noc_y, recv_l1[plane]);
                            if (plane == 0 && !fp32_state) {
                                // The BF16 recipes' numerator CB holds, per Q row, O (plane 0) then per-chunk scratch.
                                constexpr uint32_t o_row_bytes = vDHt * Transfer::page_bytes;
                                for (uint32_t r = 0; r < q_tiles; ++r) {
                                    noc_async_write(src + r * 2 * o_row_bytes, dst + r * o_row_bytes, o_row_bytes);
                                }
                            } else {
                                noc_async_write(src, dst, Transfer::plane_bytes<q_tiles, vDHt, max_bytes>(plane));
                            }
                        }
                        noc_async_write_barrier();
                        noc_semaphore_inc(prev_sem, sparse_sdpa::SPLIT_PARTIAL_VALID);
                        request.pop_front(1);
                        ack.push_back(1);
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
                        recv_out_cb.reserve_back(q_tiles * vDHt);
                        recv_out_cb.push_back(q_tiles * vDHt);
                        recv_max_cb.reserve_back(q_tiles);
                        recv_max_cb.push_back(q_tiles);
                        recv_sum_cb.reserve_back(q_tiles);
                        recv_sum_cb.push_back(q_tiles);
                    }
                }
            }
            // The pass's untilized rows (its heads past the token's H are padding compute dropped).
            const uint32_t h0 = pass * q_tiles * tt::constants::TILE_HEIGHT;
            const uint32_t pass_tiles = head_tiles - pass * q_tiles < q_tiles ? head_tiles - pass * q_tiles : q_tiles;
            out_cb.wait_front(pass_tiles * vDHt);
            for (uint32_t r = 0; r < pass_tiles * tt::constants::TILE_HEIGHT; ++r) {
                // head-row h0 + r at CB read-ptr byte offset r*row_bytes; DRAM page = h*S + tok.
                noc.async_write(
                    out_cb, out, row_bytes, {.offset_bytes = r * row_bytes}, {.page_id = (h0 + r) * S + tok});
            }
            noc.async_write_barrier();
            out_cb.pop_front(pass_tiles * vDHt);
        }
    }
}
