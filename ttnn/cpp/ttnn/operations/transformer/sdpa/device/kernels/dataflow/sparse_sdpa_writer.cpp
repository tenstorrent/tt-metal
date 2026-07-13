// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa writer: per token, drain the untilized [H, V_DIM] row-major block from compute (Sqt*vDHt
// tile-sized pages, row-major bytes inside) and write each head-row to the ROW-MAJOR [1,H,S,V_DIM] output.
// Also builds the three persistent compute-input tiles once at startup (the reader is busier): the reduce
// identity scaler, the col-identity (row-sum finalize), and the all-(-inf) mask tile.
// Uses the object-based NoC + circular-buffer APIs.

#include <stdint.h>
#include <tt-metalium/constants.hpp>  // tt::constants::TILE_WIDTH
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
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_K_RM);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_IDX);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_KREQ);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_KACK);
    constexpr uint32_t packed_row_bytes = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::PACKED_ROW_BYTES);
    constexpr auto out_args = TensorAccessorArgs<sparse_sdpa::writer_ct_arg::END, 0>();
    // kv carries a RUNTIME tensor shape (T dim in common runtime args), so its accessor spans both arg streams.
    constexpr auto kv_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t tok_start = get_arg_val<uint32_t>(1);
    const uint32_t tok_count = get_arg_val<uint32_t>(2);
    const uint32_t kv_addr = get_arg_val<uint32_t>(3);  // dual-NoC K-half gather
    // Indexed KV cache: page offset (cache_batch_idx * T) selecting the cache's batch slot; 0 if not indexed.
    const uint32_t kv_batch_page_offset = get_arg_val<uint32_t>(4);

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
    experimental::CB k_cb(cb_k_rm), idx_cb(cb_idx), kreq_cb(cb_kreq), kack_cb(cb_kack);
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

    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        // K-gather phase: process this token's active chunks (reader signals per chunk; is_last ends the token).
        // Gather rows [0,split) on the writer's NoC into the reader-reserved destination, then ack.
        bool last = false;
        while (!last) {
            kreq_cb.wait_front(1);
            uint32_t base, split, dst_l1, scale_dst_l1 = 0;
            {
                volatile tt_l1_ptr uint32_t* request =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                base = request[sparse_sdpa::gather_request::BASE];
                split = request[sparse_sdpa::gather_request::SPLIT];
                if constexpr (scaled_kv) {
                    last = request[sparse_sdpa::gather_request::IS_LAST] != 0;
                    dst_l1 = request[sparse_sdpa::gather_request::DST_L1];
                    scale_dst_l1 = request[sparse_sdpa::gather_request::SCALE_DST_L1];
                } else {
                    last = request[sparse_sdpa::gather_request::IS_LAST] != 0;
                    dst_l1 = request[sparse_sdpa::gather_request::DST_L1];
                }
            }
            kreq_cb.pop_front(1);
            if constexpr (scaled_kv) {
                sparse_sdpa::trid_ring_gather<
                    block_cyclic,
                    bc_chunk_local,
                    bc_sp,
                    bc_shard_stride_gap,
                    bc_slab_stride_gap,
                    sparse_sdpa::SCALED_K_TRID_RING>(
                    noc, kv, dst_l1, idx_ptr, base, 0, split, packed_row_bytes, kv_batch_page_offset);
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
                    noc, kv, dst_l1, idx_ptr, base, 0, split, k_row_bytes, kv_batch_page_offset);
            }
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
        }
        out_cb.wait_front(block_tiles);  // one untilized [H, V_DIM] block
        for (uint32_t h = 0; h < H; ++h) {
            // row h (head h) at CB read-ptr byte offset h*row_bytes; DRAM page = h*S + tok.
            noc.async_write(out_cb, out, row_bytes, {.offset_bytes = h * row_bytes}, {.page_id = h * S + tok});
        }
        noc.async_write_barrier();
        out_cb.pop_front(block_tiles);
    }
}
