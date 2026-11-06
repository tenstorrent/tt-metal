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
#ifdef SPARSE_KSPLIT
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_tree.hpp"
#endif

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
    constexpr auto out_args =
        TensorAccessorArgs<sparse_sdpa::writer_ct_arg::END, sparse_sdpa::writer_common_arg::END>();
    // kv carries a RUNTIME tensor shape (T dim in common runtime args), so its accessor spans both arg streams.
    constexpr auto kv_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::OUTPUT_ADDRESS);
    const uint32_t tok_start = get_arg_val<uint32_t>(0);
    const uint32_t tok_count = get_arg_val<uint32_t>(1);
#ifdef SPARSE_KSPLIT
    // Key-split: arg 2 = this core's rank, args 3.. = packed (x << 16 | y) NOC coords of the group's ranks.
    const uint32_t group_coords_addr_idx = 3;
#endif
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
#ifdef SPARSE_KSPLIT
    // generate_bcast_col_scalar writes only column 0 and assumes a zeroed tile; L1 is not cleared between programs.
    // Key-split runs the identity matmul twice (partial collapse, then normalize), so every other column must be 0.
    {
        volatile tt_l1_ptr uint32_t* identity =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(experimental::CB(cb_col_identity).get_write_ptr());
        for (uint32_t i = 0; i < tt::constants::TILE_HW / 2; ++i) {
            identity[i] = 0;
        }
    }
#endif
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
            constexpr auto sink_args = TensorAccessorArgs<
                kv_args.next_compile_time_args_offset(),
                kv_args.next_common_runtime_args_offset()>();
            const auto sink = TensorAccessor(sink_args, get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::ATTENTION_SINK_ADDRESS));
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

#ifdef SPARSE_KSPLIT_EPOCH
    // Key-split in turns: tok_start = group slot, tok_count = turns; turn k serves token k*G + slot (>= S: idle).
    const uint32_t ks_groups = get_arg_val<uint32_t>(3 + SPARSE_KSPLIT);
    for (uint32_t turn = 0; turn < tok_count; ++turn) {
        const uint32_t tok = turn * ks_groups + tok_start;
#else
    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
#endif
        // K-gather phase: process this token's active chunks (reader signals per chunk; is_last ends the token).
        // Gather rows [0,split) on the writer's NoC into the reader-reserved destination, then ack.
        bool last = false;
#ifdef SPARSE_KSPLIT
        uint32_t tree = 0;
#endif
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
#ifdef SPARSE_KSPLIT
                tree = request[sparse_sdpa::ksplit_req::TREE];
#endif
                if constexpr (scaled_kv) {
                    scale_dst_l1 = request[sparse_sdpa::gather_request::SCALE_DST_L1];
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
#ifdef SPARSE_KSPLIT
        {
            using namespace sparse_sdpa;
            const uint32_t role = tree_word::role(tree);
#ifdef SPARSE_KSPLIT_EPOCH
            // Epoch handshake (turns): a sender may write its parent's reducer slot / nibble only after the parent
            // finished the RECEIVE phase of the previous turn. This core publishes "turn+1 receive phases done" in
            // sem EPOCH after it has copied its children out and zeroed its nibbles (idle ranks publish too: they may
            // be a parent in the next turn).
            volatile tt_l1_ptr uint32_t* epoch_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SPARSE_KSPLIT_EPOCH_SEM_ID));
#endif
            if (role == tree_word::ROLE_IDLE) {
#ifdef SPARSE_KSPLIT_EPOCH
                *epoch_ptr = turn + 1;
#endif
                continue;  // no work for this token on this rank: nothing to receive, send or write
            }
            constexpr uint32_t tile_bytes_im = 2048;  // bf16 tile (all combine CBs are Float16_b)
            constexpr uint32_t ml_bytes = (H / tt::constants::TILE_HEIGHT) * tile_bytes_im;  // l and m: Sqt tiles
            constexpr uint32_t o_bytes = block_tiles * tile_bytes_im;                        // o: Sqt*vDHt tiles
            constexpr uint32_t slot_bytes = 2 * ml_bytes + o_bytes;                          // per-round slot: l, m, o
            constexpr uint32_t cb_red = SPARSE_KSPLIT_CB_RED;
            constexpr uint32_t cb_l_in = SPARSE_KSPLIT_CB_L_IN;
            constexpr uint32_t cb_m_in = SPARSE_KSPLIT_CB_M_IN;
            constexpr uint32_t cb_o_in = SPARSE_KSPLIT_CB_O_IN;
            volatile tt_l1_ptr uint32_t* sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SPARSE_KSPLIT_SEM_ID));
            experimental::CB red_cb(cb_red);
            const uint32_t child_mask = tree_word::child_mask(tree);
            // Receive: per round with a child, wait for its nibble, then copy its (l, m, o) slot from the local
            // reducer CB into the compute-visible staging CBs (sdpa_decode writer_decode_all.cpp:271-355).
            for (uint32_t round = 0; round < MAX_TREE_ROUNDS; ++round) {
                if (((child_mask >> round) & 1u) == 0) {
                    continue;
                }
                while (true) {
                    invalidate_l1_cache();
                    if (((*sem_ptr >> (4 * round)) & 0xFu) >= 1) {
                        break;
                    }
                }
                const uint32_t noc_id = noc.get_noc_id();
                UnicastEndpoint slot_src;
                uint32_t src_addr = red_cb.get_write_ptr() + round * slot_bytes;
                experimental::CB l_cb(cb_l_in), m_cb(cb_m_in), o_cb(cb_o_in);
                l_cb.reserve_back(H / tt::constants::TILE_HEIGHT);
                noc.async_read(
                    slot_src,
                    CoreLocalMem<uint32_t>(l_cb.get_write_ptr()),
                    ml_bytes,
                    {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = src_addr},
                    {});
                src_addr += ml_bytes;
                m_cb.reserve_back(H / tt::constants::TILE_HEIGHT);
                noc.async_read(
                    slot_src,
                    CoreLocalMem<uint32_t>(m_cb.get_write_ptr()),
                    ml_bytes,
                    {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = src_addr},
                    {});
                src_addr += ml_bytes;
                o_cb.reserve_back(block_tiles);
                noc.async_read(
                    slot_src,
                    CoreLocalMem<uint32_t>(o_cb.get_write_ptr()),
                    o_bytes,
                    {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = src_addr},
                    {});
                noc.async_read_barrier();
                l_cb.push_back(H / tt::constants::TILE_HEIGHT);
                m_cb.push_back(H / tt::constants::TILE_HEIGHT);
                o_cb.push_back(block_tiles);
            }
#ifdef SPARSE_KSPLIT_EPOCH
            if (child_mask != 0) {
                *sem_ptr = 0;  // every child of this turn has been consumed: no nibble can be lost or left stale
            }
            *epoch_ptr = turn + 1;  // order: nibbles zeroed first, then published
#endif
            if (role == tree_word::ROLE_WORKER) {
                // Send the unnormalized partial (l, m, o) to the parent's reducer slot for this round, then
                // raise the parent's nibble (writer_decode_all.cpp:364-420). Done after this.
                // Compute pushed the partial through cb_out_rm as [o | l | m] (block_tiles + 2*Sqt tiles).
                constexpr uint32_t sqt = H / tt::constants::TILE_HEIGHT;
                out_cb.wait_front(block_tiles + 2 * sqt);
                const uint32_t part_base = out_cb.get_read_ptr();
                const uint32_t send_round = tree_word::send_round(tree);
                const uint32_t parent = tree_word::parent(tree);
                const uint32_t pc = get_arg_val<uint32_t>(group_coords_addr_idx + parent);
                const uint32_t px = pc >> 16;
                const uint32_t py = pc & 0xFFFFu;
#ifdef SPARSE_KSPLIT_EPOCH
                if (turn > 0) {
                    // Poll the parent's epoch (NOC read of its sem word into the local POLL sem word) until it
                    // has finished receiving turn-1, i.e. its slots and nibbles are free for this turn.
                    const uint32_t poll_addr = get_semaphore(SPARSE_KSPLIT_POLL_SEM_ID);
                    volatile tt_l1_ptr uint32_t* poll_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll_addr);
                    UnicastEndpoint epoch_src;
                    while (true) {
                        noc.async_read(
                            epoch_src,
                            CoreLocalMem<uint32_t>(poll_addr),
                            4,
                            {.noc_x = px, .noc_y = py, .addr = get_semaphore(SPARSE_KSPLIT_EPOCH_SEM_ID)},
                            {});
                        noc.async_read_barrier();
                        invalidate_l1_cache();
                        if (*poll_ptr >= turn) {
                            break;
                        }
                    }
                }
#endif
                UnicastEndpoint dst;
                uint32_t dst_addr = red_cb.get_write_ptr() + send_round * slot_bytes;
                // Slot layout in the parent's reducer CB is [l | m | o] (the receive side reads that order).
                noc.async_write(
                    CoreLocalMem<uint32_t>(part_base + o_bytes),
                    dst,
                    ml_bytes,
                    {},
                    {.noc_x = px, .noc_y = py, .addr = dst_addr});
                dst_addr += ml_bytes;
                noc.async_write(
                    CoreLocalMem<uint32_t>(part_base + o_bytes + ml_bytes),
                    dst,
                    ml_bytes,
                    {},
                    {.noc_x = px, .noc_y = py, .addr = dst_addr});
                dst_addr += ml_bytes;
                noc.async_write(
                    CoreLocalMem<uint32_t>(part_base), dst, o_bytes, {}, {.noc_x = px, .noc_y = py, .addr = dst_addr});
                noc.async_write_barrier();
                Semaphore<>(SPARSE_KSPLIT_SEM_ID).up(noc, px, py, 1u << (4 * send_round));
                out_cb.pop_front(block_tiles + 2 * sqt);
                noc.async_atomic_barrier();
                continue;
            }
            // Root: fall through to the normal output write.
        }
#endif
#ifdef SPARSE_KSPLIT
        constexpr uint32_t root_pad_tiles = 2 * (H / tt::constants::TILE_HEIGHT);  // compute pads the root push
#else
        constexpr uint32_t root_pad_tiles = 0;
#endif
        out_cb.wait_front(block_tiles + root_pad_tiles);  // one untilized [H, V_DIM] block (+ ksplit pad)
        for (uint32_t h = 0; h < H; ++h) {
            // row h (head h) at CB read-ptr byte offset h*row_bytes; DRAM page = h*S + tok.
            noc.async_write(out_cb, out, row_bytes, {.offset_bytes = h * row_bytes}, {.page_id = h * S + tok});
        }
        noc.async_write_barrier();
        out_cb.pop_front(block_tiles + root_pad_tiles);
    }
}
