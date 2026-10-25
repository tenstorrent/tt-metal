// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#if !defined(KERNEL_BUILD)
#include <cstdlib>
#endif

// Host/device constants shared by the sparse-SDPA factory and JIT kernels. Keep this header free of host-only and
// kernel-only dependencies so both compilation paths use the same protocol and packed-cache geometry.
namespace sparse_sdpa {

// Uniform dispatch values precede tensor-accessor metadata in each kernel's common arguments.
namespace reader_common_arg {
enum : uint32_t {
    Q_ADDRESS,
    KV_ADDRESS,
    INDICES_ADDRESS,
    KV_BATCH_PAGE_OFFSET,
    KV_SPILL_ADDRESS,
    KV_FRONT_PAGES,
    KV_SPILL_ROWS,
    KV_SPILL_ROW_STRIDE,
    END
};
}
namespace writer_common_arg {
enum : uint32_t {
    OUTPUT_ADDRESS,
    KV_ADDRESS,
    KV_BATCH_PAGE_OFFSET,
    ATTENTION_SINK_ADDRESS,
    KV_SPILL_ADDRESS,
    KV_FRONT_PAGES,
    KV_SPILL_ROWS,
    KV_SPILL_ROW_STRIDE,
    END
};
}

constexpr uint32_t SCALE_BLOCK_WIDTH = 128;
constexpr uint32_t PACKED_FIELD_ADDRESS_UNIT_BYTES = 16;
constexpr uint32_t CB_PAGE_ALIGNMENT = 16;
constexpr uint32_t CB_DOUBLE_BUFFER_DEPTH = 2;

constexpr uint32_t align_cb_page(uint32_t bytes) {
    return ((bytes + CB_PAGE_ALIGNMENT - 1) / CB_PAGE_ALIGNMENT) * CB_PAGE_ALIGNMENT;
}

constexpr uint32_t scale_block_count(uint32_t latent_width) { return latent_width / SCALE_BLOCK_WIDTH; }

constexpr uint32_t scaled_kv_scale_bytes(uint32_t latent_width) {
    return scale_block_count(latent_width) * sizeof(float);
}

constexpr bool scaled_kv_rope_offset_is_aligned(uint32_t latent_width) {
    return (latent_width + scaled_kv_scale_bytes(latent_width)) % PACKED_FIELD_ADDRESS_UNIT_BYTES == 0;
}

constexpr uint32_t packed_kv_payload_bytes(uint32_t k_dim, uint32_t latent_width) {
    return latent_width + scaled_kv_scale_bytes(latent_width) + (k_dim - latent_width) * sizeof(uint16_t);
}

// Reader -> writer dual-NoC gather request. SCALE_DST_L1 is used only by the scaled-cache path.
namespace gather_request {
constexpr uint32_t BASE = 0;
constexpr uint32_t SPLIT = 1;
constexpr uint32_t IS_LAST = 2;
constexpr uint32_t DST_L1 = 3;
constexpr uint32_t SCALE_DST_L1 = 4;
constexpr uint32_t EMPTY = 5;  // split-K: the segment has no active chunk (no gather, no ack)
constexpr uint32_t WORD_COUNT = 6;
constexpr uint32_t PAGE_BYTES = align_cb_page(WORD_COUNT * sizeof(uint32_t));
}  // namespace gather_request

// Reader -> compute token control message.
namespace control_message {
constexpr uint32_t ACTIVE_CHUNKS = 0;  // active chunks in THIS segment (the whole token unless split-K)
constexpr uint32_t VALID_KEYS = 1;
constexpr uint32_t CHUNK_BASE = 2;  // first (token-global) chunk index of the segment
constexpr uint32_t SEG_KIND = 3;    // seg_kind::*
constexpr uint32_t WORD_COUNT = 4;
constexpr uint32_t PAGE_BYTES = align_cb_page(WORD_COUNT * sizeof(uint32_t));
}  // namespace control_message

constexpr uint32_t ACK_PAGE_BYTES = CB_PAGE_ALIGNMENT;

// Split-K work distribution (S > cores and S % cores != 0, e.g. GLM: 160 query rows per chip on 96/120 cores): the
// (token, k-chunk) units are dealt contiguously over the cores, so a token may straddle two neighbouring cores. Core i
// runs, in order: an optional HEAD segment (the last chunks of token tok_start-1; its unnormalized flash state
// (out, max, partial row-sum) is NoC-written into core i-1's receive CBs + a semaphore), its FULL tokens, and an
// optional TAIL segment (the first chunks of token tok_start+tok_count; core i+1's HEAD state is merged in before the
// normalize). The HEAD runs first on its core, so the TAIL's wait is normally already satisfied.
namespace seg_kind {
constexpr uint32_t FULL = 0;
constexpr uint32_t HEAD = 1;
constexpr uint32_t TAIL = 2;
}  // namespace seg_kind
// Per-core runtime args (all three kernels). HEAD_C0 = first chunk of the HEAD segment (0 = none); TAIL_C1 = chunk
// count of the TAIL segment (0 = none); PREV_NOC_X/Y = core i-1 (HEAD destination).
namespace split_rt_arg {
enum : uint32_t { TOK_START = 0, TOK_COUNT, HEAD_C0, TAIL_C1, PREV_NOC_X, PREV_NOC_Y, END };
}
// Values the HEAD sender adds to core i-1's semaphore.
constexpr uint32_t SPLIT_PARTIAL_EMPTY = 1;  // the HEAD had no active chunk (sentinel tail): nothing to merge
constexpr uint32_t SPLIT_PARTIAL_VALID = 2;

// Compile-time argument positions. The factory checks the vector size against each END value before appending
// TensorAccessorArgs, while the kernels use the same names to decode the stream.
namespace reader_ct_arg {
enum : uint32_t {
    H = 0,
    S,
    TOPK,
    K_CHUNK,
    K_DIM,
    CB_Q_RM,
    CB_K_RM,
    CB_MASK_PART,
    CB_IDX,
    CB_CTRL,
    Q_ELEM_BYTES,
    KV_ELEM_BYTES,
    IDX_ELEM_BYTES,
    BLOCK_CYCLIC,
    BC_CHUNK_LOCAL,
    BC_SP,
    BC_SHARD_STRIDE_GAP,
    BC_SLAB_STRIDE_GAP,
    SCALED_KV,
    LATENT_WIDTH,
    CB_K_SCALE_BCAST,
    PACKED_ROW_BYTES,
    CB_KREQ,
    CB_KACK,
    SPLIT_K,
    END,
};
}  // namespace reader_ct_arg

namespace writer_ct_arg {
enum : uint32_t {
    H = 0,
    S,
    V_DHT,
    CB_OUT_RM,
    CB_SCALE,
    CB_COL_IDENTITY,
    CB_NEGINF,
    OUT_ELEM_BYTES,
    BLOCK_CYCLIC,
    BC_CHUNK_LOCAL,
    BC_SP,
    BC_SHARD_STRIDE_GAP,
    BC_SLAB_STRIDE_GAP,
    SCALED_KV,
    K_DIM,
    KV_ELEM_BYTES,
    CB_IDX,
    CB_KREQ,
    CB_KACK,
    PACKED_ROW_BYTES,
    USE_ATTENTION_SINK,
    CB_ATTENTION_SINK,
    CB_SINK_SCRATCH,
    SPLIT_K,
    CB_SEND_OUT,
    CB_SEND_MAX,
    CB_SEND_SUM,
    CB_RECV_OUT,
    CB_RECV_MAX,
    CB_RECV_SUM,
    CB_MERGE_CTRL,
    END,
};
}  // namespace writer_ct_arg

namespace compute_ct_arg {
enum : uint32_t {
    H = 0,
    DHT,
    V_DHT,
    SKT,
    SCALE,
    CB_Q_RM,
    CB_Q_IN,
    CB_K_RM,
    CB_K_IN,
    CB_NEGINF,
    CB_MASK_PART,
    CB_SCALE,
    CB_QK_IM,
    CB_MAX_A,
    CB_MAX_B,
    CB_SUM_A,
    CB_SUM_B,
    CB_OUT_A,
    CB_OUT_B,
    CB_CORR,
    CB_OUT_IM,
    CB_OUT_RM,
    CB_CTRL,
    CB_COL_IDENTITY,
    CB_RECIP_SCRATCH,
    SCALED_KV,
    CB_K_ROPE_RM,
    CB_K_SCALE_BCAST,
    CB_K_LATENT_TILE,
    CB_K_ROPE_TILE,
    MATH_APPROX_MODE,
    QUERY_SUBBLOCK,
    PACKED_ROW_BYTES,
    USE_ATTENTION_SINK,
    CB_ATTENTION_SINK,
    SPLIT_K,
    CB_SEND_OUT,
    CB_SEND_MAX,
    CB_SEND_SUM,
    CB_RECV_OUT,
    CB_RECV_MAX,
    CB_RECV_SUM,
    CB_MERGE_CTRL,
    CB_MMAX,
    CB_CORR2,
    KDEQ_BATCH,
    END,
};
}  // namespace compute_ct_arg

#if !defined(KERNEL_BUILD)
// Host: batched (per-chunk) K reconstruct for L1-resident KV -- the compact front of the sparse KV push, where the
// op is compute-bound. cb_k_rm then holds KDEQ_SLABS 32-row slabs. SPARSE_SDPA_KDEQ_BATCH=0 disables.
constexpr uint32_t KDEQ_SLABS = 4;
// SPARSE_SDPA_KDEQ_BATCH: unset/1 = L1-resident KV only, 0 = never, 2 = always (tests: exercises the path on DRAM KV).
inline uint32_t kdeq_batch_mode() {
    const char* v = std::getenv("SPARSE_SDPA_KDEQ_BATCH");
    return v == nullptr ? 1u : static_cast<uint32_t>(std::strtoul(v, nullptr, 0));
}
// Host: split-K is on whenever it applies; SPARSE_SDPA_SPLITK=0 restores the whole-token split (A/B, bisection).
inline bool split_k_enabled() {
    const char* v = std::getenv("SPARSE_SDPA_SPLITK");
    return v == nullptr || v[0] != '0';
}
// Host: split-K for S < cores (small chunks: each token over <= 2 cores); SPARSE_SDPA_SPLITK_SMALL=0 disables.
inline bool split_k_small_enabled() {
    const char* v = std::getenv("SPARSE_SDPA_SPLITK_SMALL");
    return v == nullptr || v[0] != '0';
}
#endif

}  // namespace sparse_sdpa
