// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

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
constexpr uint32_t ACTIVE_CHUNKS = 0;  // active chunks in THIS segment (whole token unless split-K)
constexpr uint32_t VALID_KEYS = 1;
constexpr uint32_t CHUNK_BASE = 2;  // first (global) chunk index of the segment
constexpr uint32_t SEG_KIND = 3;    // seg_kind::*
constexpr uint32_t WORD_COUNT = 4;
constexpr uint32_t PAGE_BYTES = align_cb_page(WORD_COUNT * sizeof(uint32_t));
}  // namespace control_message

constexpr uint32_t ACK_PAGE_BYTES = CB_PAGE_ALIGNMENT;

// Split-K work distribution (S > cores, S % cores != 0): the (token, chunk) units are dealt contiguously over the
// cores, so a token may straddle two neighbouring cores. Core i runs, in order: an optional HEAD segment (the last
// chunks of token full_start-1; its unnormalized flash state is sent to core i-1), its FULL tokens, and an optional
// TAIL segment (the first chunks of token full_start+full_count; core i+1's HEAD state is merged in before the
// normalize). Per-core runtime args after (tok_start, tok_count): head_c0 (0 = none), tail_c1 (0 = none), and the
// NoC xy of core i-1 (HEAD destination).
namespace seg_kind {
constexpr uint32_t FULL = 0;
constexpr uint32_t HEAD = 1;
constexpr uint32_t TAIL = 2;
}  // namespace seg_kind
namespace split_rt_arg {
enum : uint32_t { TOK_START = 0, TOK_COUNT, HEAD_C0, TAIL_C1, PREV_NOC_X, PREV_NOC_Y, END };
}
// Semaphore values written by the HEAD sender into core i-1's semaphore.
constexpr uint32_t SPLIT_PARTIAL_EMPTY = 1;
constexpr uint32_t SPLIT_PARTIAL_VALID = 2;

// Perf-ablation / profiling bitmask (host: SPARSE_SDPA_DBG env -> kernel define SPARSE_SDPA_DBG). Debug only.
namespace dbg {
constexpr uint32_t NO_GATHER = 1u;        // reader/writer skip the KV-row NoC reads (compute-only timing)
constexpr uint32_t NO_COMPUTE = 2u;       // compute only drains the CBs (gather-only timing)
constexpr uint32_t PROF_COMPUTE = 4u;     // per-chunk compute phase zones (<=1 token/core or markers overflow)
constexpr uint32_t PROF_READER = 8u;      // per-slab reader zones
constexpr uint32_t NO_DEQUANT_MUL = 16u;  // skip the latent * scale multiply (timing only; wrong numerics)
constexpr uint32_t NO_KTILIZE = 32u;      // skip the latent/rope tilize (timing only; wrong numerics)
constexpr uint32_t PROF_READER_GATHER = 64u;  // per-slab reader zone around the gather + writer join
constexpr uint32_t PROF_COMPUTE_WAIT = 128u;  // per-slab compute zone around the K-row wait
constexpr uint32_t EAGER_OUT = 256u;          // writer drains each output block before the next gather (base order)
#if defined(SPARSE_SDPA_DBG)
constexpr uint32_t MASK = SPARSE_SDPA_DBG;
#else
constexpr uint32_t MASK = 0;
#endif
}  // namespace dbg
#if !defined(KERNEL_BUILD)
}  // namespace sparse_sdpa
#include <cstdlib>
namespace sparse_sdpa {
inline uint32_t debug_mask() {
    const char* v = std::getenv("SPARSE_SDPA_DBG");
    return v ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : 0u;
}
// Packed KV slabs buffered between gather and compute (SPARSE_SDPA_KRM_SLABS, default 2 = base double buffer).
inline uint32_t krm_slabs() {
    const char* v = std::getenv("SPARSE_SDPA_KRM_SLABS");
    const uint32_t n = v ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : 0u;
    return n >= 2 ? n : CB_DOUBLE_BUFFER_DEPTH;
}
inline float split_k_slope() {
    const char* v = std::getenv("SPARSE_SDPA_SPLITK_SLOPE");
    return v ? std::strtof(v, nullptr) : 0.0f;
}
// Batched (per-chunk) K reconstruct; needs SPARSE_SDPA_KRM_SLABS a multiple of Skt. SPARSE_SDPA_KDEQ_BATCH=0 disables.
inline bool kdeq_batch_enabled() {
    const char* v = std::getenv("SPARSE_SDPA_KDEQ_BATCH");
    return v == nullptr || v[0] != '0';
}
// PV output subblock width (1 = base; 2 needs qsb*2 <= DST tiles). SPARSE_SDPA_PV_W.
inline uint32_t pv_w() {
    const char* v = std::getenv("SPARSE_SDPA_PV_W");
    return v ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : 1u;
}
// Scaled-KV gather reads in flight per RISC (trid ring depth, 2..15). 0 = kernel default (8). SPARSE_SDPA_TRID.
inline uint32_t trid_ring() {
    const char* v = std::getenv("SPARSE_SDPA_TRID");
    const uint32_t n = v ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : 0u;
    return (n >= 2 && n <= 15) ? n : 0u;
}
// Split-K work distribution (prototype): SPARSE_SDPA_SPLITK=1 enables, unset/0 keeps the whole-token split.
inline bool split_k_enabled() {
    const char* v = std::getenv("SPARSE_SDPA_SPLITK");
    return v != nullptr && v[0] == '1';
}
#endif

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
    PV_W,
    END,
};
}  // namespace compute_ct_arg

}  // namespace sparse_sdpa
