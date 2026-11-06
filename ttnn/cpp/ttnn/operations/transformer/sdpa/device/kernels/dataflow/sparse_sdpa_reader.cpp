// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa reader: per token, gather Q rows + per-chunk indexed K rows (sentinel suffix zero-filled),
// plus the per-token partial-boundary mask tile. (The persistent scaler/col-identity/-inf tiles are built
// by the lighter-loaded writer.) Uses the object-based NoC + circular-buffer APIs.
//
// Sentinels are a contiguous tail (producer contract): nv = first-sentinel index = valid-key count, so only
// ceil(nv/k_chunk) chunks are active; all-sentinel chunks are skipped (no read/fill/compute), compute learns
// nv + the active count via cb_ctrl. The per-chunk K gather is split across BOTH NoCs (the op is NoC-link
// bound): the reader gathers half the rows on its NoC and hands the other half to the writer (cb_kreq/kack),
// which gathers them on its NoC into the same shared cb_k_rm L1.

#include <stdint.h>
#include "api/tile_constants.h"  // tt::constants::TILE_HEIGHT
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_gather.hpp"  // dual-NoC trid-ring gather helper (shared with the writer)
#ifdef SPARSE_KSPLIT
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_tree.hpp"
#endif

constexpr uint32_t sentinel = 0xFFFFFFFFu;
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

FORCE_INLINE uint32_t valid_rows_in_slab(uint32_t valid, uint32_t row_base) {
    if (valid <= row_base) {
        return 0;
    }
    const uint32_t remaining = valid - row_base;
    return remaining < tt::constants::TILE_HEIGHT ? remaining : tt::constants::TILE_HEIGHT;
}

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::H);
    constexpr uint32_t S = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::S);
    constexpr uint32_t topk = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::TOPK);
    constexpr uint32_t k_chunk = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::K_CHUNK);
    constexpr uint32_t k_dim =
        get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::K_DIM);  // q/kv head dim (from the tensor)

    // CB ids — passed by the factory (the SparseCB enum is the single source); order must match the
    // factory's reader CB-arg block (compile args 5..9).
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::CB_Q_RM);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::CB_K_RM);
    constexpr uint32_t cb_mask_part = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::CB_MASK_PART);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::CB_IDX);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::CB_CTRL);

    // Element sizes (from the tensors' dtypes; passed after the CB ids so their indices stay fixed).
    constexpr uint32_t q_elem_bytes = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::Q_ELEM_BYTES);  // bf16
    constexpr uint32_t kv_elem_bytes =
        get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::KV_ELEM_BYTES);  // bf16 or fp8_e4m3
    constexpr uint32_t idx_elem_bytes = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::IDX_ELEM_BYTES);  // uint32
    constexpr bool block_cyclic = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::BLOCK_CYCLIC) != 0;
    constexpr uint32_t bc_chunk_local = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::BC_CHUNK_LOCAL);
    constexpr uint32_t bc_sp = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::BC_SP);
    constexpr uint32_t bc_shard_stride_gap = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::BC_SHARD_STRIDE_GAP);
    constexpr uint32_t bc_slab_stride_gap = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::BC_SLAB_STRIDE_GAP);
    constexpr bool scaled_kv = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::SCALED_KV) != 0;
    constexpr uint32_t latent_width = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::LATENT_WIDTH);
    constexpr uint32_t cb_k_scale_bcast = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::CB_K_SCALE_BCAST);
    constexpr uint32_t packed_row_bytes = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::PACKED_ROW_BYTES);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::CB_KREQ);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(sparse_sdpa::reader_ct_arg::CB_KACK);

    // kv carries a RUNTIME tensor shape (its T dim is common runtime args, not compile-time), so its accessor
    // spans both the compile-time AND common-runtime arg streams. Thread both offsets through all three.
    constexpr auto q_args = TensorAccessorArgs<sparse_sdpa::reader_ct_arg::END, sparse_sdpa::reader_common_arg::END>();
    constexpr auto kv_args =
        TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
    constexpr auto idx_args =
        TensorAccessorArgs<kv_args.next_compile_time_args_offset(), kv_args.next_common_runtime_args_offset()>();

    const uint32_t q_addr = get_common_arg_val<uint32_t>(sparse_sdpa::reader_common_arg::Q_ADDRESS);
    const uint32_t kv_addr = get_common_arg_val<uint32_t>(sparse_sdpa::reader_common_arg::KV_ADDRESS);
    const uint32_t idx_addr = get_common_arg_val<uint32_t>(sparse_sdpa::reader_common_arg::INDICES_ADDRESS);
    const uint32_t tok_start = get_arg_val<uint32_t>(0);
    const uint32_t tok_count = get_arg_val<uint32_t>(1);
#ifdef SPARSE_KSPLIT
    // Key-split: this core is rank `rank` of the SPARSE_KSPLIT cores that share token tok_start.
    const uint32_t rank = get_arg_val<uint32_t>(2);
#endif
    // Indexed KV cache: page offset (cache_batch_idx * T) selecting the cache's batch slot; 0 if not indexed.
    const uint32_t kv_batch_page_offset =
        get_common_arg_val<uint32_t>(sparse_sdpa::reader_common_arg::KV_BATCH_PAGE_OFFSET);

    constexpr uint32_t q_row_bytes = k_dim * q_elem_bytes;   // Q row (bf16)
    constexpr uint32_t k_row_bytes = k_dim * kv_elem_bytes;  // K row (native dtype: fp8 or bf16)
    constexpr uint32_t latent_row_bytes = latent_width;
    constexpr uint32_t scale_blocks = sparse_sdpa::scale_block_count(latent_width);
    constexpr uint32_t idx_row_bytes = topk * idx_elem_bytes;  // one index row

    Noc noc;
    experimental::CB q_cb(cb_q_rm), k_cb(cb_k_rm), mask_part_cb(cb_mask_part), idx_cb(cb_idx), ctrl_cb(cb_ctrl);
    experimental::CB kreq_cb(cb_kreq), kack_cb(cb_kack);  // dual-NoC K-gather handoff to the writer
    experimental::CB scale_bcast_cb(cb_k_scale_bcast);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    // Reader-internal scratch for one token's index row (reserved once, reused).
    idx_cb.reserve_back(1);
    const uint32_t idx_l1 = idx_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

#ifdef SPARSE_KSPLIT
#ifndef SPARSE_KSPLIT_MIN_CHUNKS
#define SPARSE_KSPLIT_MIN_CHUNKS 8
#endif
#endif
#ifdef SPARSE_KSPLIT_EPOCH
    // Key-split in turns: tok_start = group slot, tok_count = turns; turn k serves token k*G + slot (>= S: idle turn).
    constexpr uint32_t ks_groups_arg = 3 + SPARSE_KSPLIT;
    const uint32_t ks_groups = get_arg_val<uint32_t>(ks_groups_arg);
    for (uint32_t turn = 0; turn < tok_count; ++turn) {
        const uint32_t tok_raw = turn * ks_groups + tok_start;
        const bool tok_valid = tok_raw < S;
        const uint32_t tok = tok_valid ? tok_raw : 0;
#else
    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
#endif
        // Q: H head-rows -> cb_q_rm.  Q is [1,H,S,576] row-major: row (h,tok) = page h*S+tok.
        q_cb.reserve_back(H);
#ifndef SPARSE_KSPLIT
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_read(q, q_cb, q_row_bytes, {.page_id = h * S + tok}, {.offset_bytes = h * q_row_bytes});
        }
#endif
#ifdef SPARSE_KSPLIT
        // Key-split: the index row loads first; Q is read below only if this rank has work.
#ifdef SPARSE_KSPLIT_EPOCH
        if (tok_valid)
#endif
            noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = tok}, {.offset_bytes = 0});
        noc.async_read_barrier();
#else
        noc.async_read_barrier();
        q_cb.push_back(H);

        // indices row for this token (page = tok) -> idx_cb scratch
        noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = tok}, {.offset_bytes = 0});
        noc.async_read_barrier();
#endif

        // binary-search the first sentinel -> nv (valid-key count); only ceil(nv/k_chunk) chunks are active.
        uint32_t nv = topk;
#ifdef SPARSE_KSPLIT_EPOCH
        if (!tok_valid) {
            nv = 1;  // idle turn: no index row was loaded (the worker count below is forced to 0)
        } else
#endif
        {
            uint32_t lo = 0, hi = topk;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (idx_ptr[mid] == sentinel) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            nv = lo == 0 ? 1 : lo;  // contract guarantees >=1 valid; guard anyway
        }
        const uint32_t n_active = (nv + k_chunk - 1) / k_chunk;

        // Chunk range owned by this core. Without key-split: all active chunks.
        uint32_t chunk_begin = 0;
        uint32_t chunk_end = n_active;
#ifdef SPARSE_KSPLIT
        // Key-split: the group of SPARSE_KSPLIT cores splits the n_active chunks over n_eff = min(N, n_active)
        // ranks, balanced: rank j < n_eff owns [floor(j*n_active/n_eff), floor((j+1)*n_active/n_eff)) (never
        // empty). Ranks j >= n_eff are idle. Every core of the group derives the same n_eff from the same row.
        // Each working rank owns at least SPARSE_KSPLIT_MIN_CHUNKS chunks (fixed, batch invariant):
        // n_eff = min(N, ceil(n_active / M)); n_active <= M gives n_eff = 1 (same bits as the N=1 path).
        const uint32_t n_cap = (n_active + SPARSE_KSPLIT_MIN_CHUNKS - 1) / SPARSE_KSPLIT_MIN_CHUNKS;
        const uint32_t n_eff = n_cap < SPARSE_KSPLIT ? n_cap : SPARSE_KSPLIT;
#ifdef SPARSE_KSPLIT_EPOCH
        const bool working = tok_valid && rank < n_eff;
#else
        const bool working = rank < n_eff;
#endif
        uint32_t tree = sparse_sdpa::tree_word::pack(sparse_sdpa::tree_word::ROLE_IDLE, 0, 0, 0);
        uint32_t valid_last = k_chunk;  // valid keys in the last local chunk (k_chunk: no sentinel boundary here)
        if (working) {
            chunk_begin = (rank * n_active) / n_eff;
            chunk_end = ((rank + 1) * n_active) / n_eff;
            tree = sparse_sdpa::tree_params(rank, n_eff);
            if (chunk_end == n_active) {
                valid_last = nv - (n_active - 1) * k_chunk;
            }
        } else {
            chunk_begin = 0;
            chunk_end = 0;
        }
#endif

        // hand the active chunk count + valid-key count to compute (it can't issue DRAM reads to derive
        // them). nv lets compute place the boundary mask (which key tiles of the last chunk to -inf).
        ctrl_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
            cp[sparse_sdpa::control_message::ACTIVE_CHUNKS] = chunk_end - chunk_begin;
            cp[sparse_sdpa::control_message::VALID_KEYS] = nv;
#ifdef SPARSE_KSPLIT
            cp[sparse_sdpa::ksplit_ctrl::VALID_LAST] = valid_last;
            cp[sparse_sdpa::ksplit_ctrl::TREE] = tree;
#endif
        }
        ctrl_cb.push_back(1);

#ifdef SPARSE_KSPLIT
        if (working) {
            for (uint32_t h = 0; h < H; ++h) {
                noc.async_read(q, q_cb, q_row_bytes, {.page_id = h * S + tok}, {.offset_bytes = h * q_row_bytes});
            }
            noc.async_read_barrier();
            q_cb.push_back(H);
        } else {
            // Idle rank: no Q, no K. The writer still needs one terminating request for this token.
            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* request =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                request[sparse_sdpa::gather_request::BASE] = 0;
                request[sparse_sdpa::gather_request::SPLIT] = 0;
                request[sparse_sdpa::gather_request::IS_LAST] = 1;
                request[sparse_sdpa::gather_request::DST_L1] = 0;
                request[sparse_sdpa::ksplit_req::TREE] = tree;
            }
            kreq_cb.push_back(1);
            kack_cb.wait_front(1);
            kack_cb.pop_front(1);
        }
#endif

        for (uint32_t chunk = chunk_begin; chunk < chunk_end; ++chunk) {
            const uint32_t base = chunk * k_chunk;
            const uint32_t valid = (nv - base) < k_chunk ? (nv - base) : k_chunk;  // (0, k_chunk]

            if constexpr (scaled_kv) {
                // Stream one 32-row compressed slab at a time. cb_k_rm owns the double-buffered packed FIFO;
                // cb_k_rope_rm is a format-only BF16 view whose read pointer is derived by compute.
                constexpr uint32_t row_tiles = k_chunk / tt::constants::TILE_HEIGHT;
                for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
                    const uint32_t row_base = row_tile * tt::constants::TILE_HEIGHT;
                    const uint32_t rows = valid_rows_in_slab(valid, row_base);
                    const uint32_t split = rows / 2;

                    k_cb.reserve_back(tt::constants::TILE_HEIGHT);
                    scale_bcast_cb.reserve_back(scale_blocks);
                    const uint32_t k_write_ptr = k_cb.get_write_ptr();
                    const uint32_t scale_write_ptr = scale_bcast_cb.get_write_ptr();
                    const uint32_t slab_base = base + row_base;
                    const bool is_last_slab = (chunk == chunk_end - 1) && (row_tile == row_tiles - 1);

                    kreq_cb.reserve_back(1);
                    {
                        volatile tt_l1_ptr uint32_t* request =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                        request[sparse_sdpa::gather_request::BASE] = slab_base;
                        request[sparse_sdpa::gather_request::SPLIT] = split;
                        request[sparse_sdpa::gather_request::IS_LAST] = is_last_slab;
                        request[sparse_sdpa::gather_request::DST_L1] = k_write_ptr;
                        request[sparse_sdpa::gather_request::SCALE_DST_L1] = scale_write_ptr;
#ifdef SPARSE_KSPLIT
                        request[sparse_sdpa::ksplit_req::TREE] = tree;
#endif
                    }
                    kreq_cb.push_back(1);

                    sparse_sdpa::trid_ring_gather<
                        block_cyclic,
                        bc_chunk_local,
                        bc_sp,
                        bc_shard_stride_gap,
                        bc_slab_stride_gap,
                        sparse_sdpa::SCALED_K_TRID_RING>(
                        noc, kv, k_write_ptr, idx_ptr, slab_base, split, rows, packed_row_bytes, kv_batch_page_offset);
                    // Expand the scale rows from the reader's gathered half while the writer handles its
                    // own half. The two RISCs write disjoint rows of the same FP32 broadcast tiles.
                    sparse_sdpa::scatter_packed_scales<scale_blocks>(
                        k_write_ptr, scale_write_ptr, split, rows, packed_row_bytes, latent_row_bytes);
                    kack_cb.wait_front(1);
                    kack_cb.pop_front(1);

                    if (rows < tt::constants::TILE_HEIGHT) {
                        noc.async_write_zeros(
                            k_cb,
                            (tt::constants::TILE_HEIGHT - rows) * packed_row_bytes,
                            {.offset_bytes = rows * packed_row_bytes});
                        noc.write_zeros_l1_barrier();
                    }
                    // Zero the scale rows for the padded suffix after both real-row halves have joined.
                    sparse_sdpa::clear_scale_rows<scale_blocks>(scale_write_ptr, rows, tt::constants::TILE_HEIGHT);
                    k_cb.push_back(tt::constants::TILE_HEIGHT);
                    scale_bcast_cb.push_back(scale_blocks);
                }
            } else {
                // K chunk rows -> cb_k_rm: read the `valid` real keys, zero-fill the sentinel suffix.
                k_cb.reserve_back(k_chunk);
                const uint32_t k_write_ptr = k_cb.get_write_ptr();
                // Dual-NoC split gather: the K-row response data is the op's bottleneck (NoC-link bound, not
                // DRAM-controller bound), so we spread it across BOTH NoCs. The reader (this NoC) gathers
                // rows [half,valid); it hands the writer rows [0,half) via cb_kreq, and the writer gathers
                // them on ITS NoC into the same reserved cb_k_rm L1 (shared on-core), then acks via cb_kack -- each
                // link then carries half the gather traffic.
                const uint32_t half = valid >> 1;
                kreq_cb.reserve_back(1);
                {
                    volatile tt_l1_ptr uint32_t* request =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                    request[sparse_sdpa::gather_request::BASE] = base;
                    request[sparse_sdpa::gather_request::SPLIT] = half;
                    request[sparse_sdpa::gather_request::IS_LAST] = chunk == chunk_end - 1;
                    request[sparse_sdpa::gather_request::DST_L1] = k_write_ptr;
#ifdef SPARSE_KSPLIT
                    request[sparse_sdpa::ksplit_req::TREE] = tree;
#endif
                }
                kreq_cb.push_back(1);
                // Reader gathers its half [half, valid) on its NoC (depth-4 trid ring, shared helper).
                sparse_sdpa::trid_ring_gather<
                    block_cyclic,
                    bc_chunk_local,
                    bc_sp,
                    bc_shard_stride_gap,
                    bc_slab_stride_gap,
                    sparse_sdpa::K_TRID_RING>(
                    noc, kv, k_write_ptr, idx_ptr, base, half, valid, k_row_bytes, kv_batch_page_offset);
                kack_cb.wait_front(1);  // writer's half landed in cb_k_rm L1
                kack_cb.pop_front(1);
                // Zero-fill the sentinel suffix: masked-key scores become a defined 0 (compute adds -inf).
                if (valid < k_chunk) {
                    noc.async_write_zeros(k_cb, (k_chunk - valid) * k_row_bytes, {.offset_bytes = valid * k_row_bytes});
                    noc.write_zeros_l1_barrier();
                }
                k_cb.push_back(k_chunk);
            }

            // Boundary mask: only the LAST active chunk can have sentinels. Its key tile that straddles
            // `valid` (tile valid/32) needs cols >= valid%32 set to -inf; the tiles after it are fully
            // masked (compute bcast-adds the persistent cb_neginf there). Build the single partial-boundary
            // tile here only when the boundary lands mid-tile; row 0 only (compute adds it row-broadcast).
            if (chunk == n_active - 1 && (valid % tt::constants::TILE_WIDTH) != 0) {
                mask_part_cb.reserve_back(1);
                {
                    const uint32_t first_masked_col = valid % tt::constants::TILE_WIDTH;
                    volatile tt_l1_ptr uint16_t* mask_tile =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_part_cb.get_write_ptr());
                    // Write row 0 of the tile: cols [0,FACE_WIDTH) are face0 row 0 (offset = col); cols
                    // [FACE_WIDTH,TILE_WIDTH) are face1 row 0 (offset = FACE_HW + col - FACE_WIDTH). Cols at or
                    // past the boundary get -inf, the rest 0.
                    for (uint32_t col = 0; col < tt::constants::TILE_WIDTH; ++col) {
                        const uint32_t elem_off = (col < tt::constants::FACE_WIDTH)
                                                      ? col
                                                      : (tt::constants::FACE_HW + col - tt::constants::FACE_WIDTH);
                        mask_tile[elem_off] = (col >= first_masked_col) ? neg_inf_bf16 : (uint16_t)0;
                    }
                }
                mask_part_cb.push_back(1);
            }
        }
    }
}
