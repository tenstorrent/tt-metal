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
enum : uint32_t { Q_ADDRESS, KV_ADDRESS, INDICES_ADDRESS, KV_BATCH_PAGE_OFFSET, END };
}
namespace writer_common_arg {
enum : uint32_t { OUTPUT_ADDRESS, KV_ADDRESS, KV_BATCH_PAGE_OFFSET, ATTENTION_SINK_ADDRESS, END };
}

constexpr uint32_t SCALE_BLOCK_WIDTH = 128;
constexpr uint32_t PACKED_FIELD_ADDRESS_UNIT_BYTES = 16;
constexpr uint32_t CB_PAGE_ALIGNMENT = 16;
constexpr uint32_t CB_DOUBLE_BUFFER_DEPTH = 2;

// Fused dequant (scaled FP8 KV, no attention sink): compute tilizes the FP8 latent and multiplies it by the
// per-row scale in one FPU pass. The FPU needs both sources in one exponent family, and FP8 unpacks as FP16, so
// the scale reaches srcB as FP16. The FPU flushes FP16 subnormals, and the smallest scale the FP8 cast makes is
// clamp(amax, 1e-4) / 448 ~= 2^-22.1, so every scale is sent as scale * 2^FUSED_DEQUANT_EXP_OFFSET to keep it a
// normal FP16 (>= 2^-14) and the factor is removed exactly: Q's latent columns are multiplied by 2^-OFFSET (so
// Q.K is unchanged) and the row-sum identity tile holds 2^OFFSET (so out/sum is unchanged). All are power-of-two
// rescales: bit-exact.
// The FPU's FP16 has no inf/NaN (exponent 31 is an ordinary binade, max 2^17 - 2^6), so the scale range is
// [2^(-14-OFFSET), 2^(17-OFFSET)) = [2^-23, 2^8): block amax from the 1e-4 clamp up to 448 * 2^8 = 114688.
constexpr int32_t FUSED_DEQUANT_EXP_OFFSET = 9;
constexpr uint16_t FP16_MAX_BITS = 0x7FFF;  // largest FPU FP16 (exponent 31 is finite)

// FP32 scale -> FP16 bits of scale * 2^FUSED_DEQUANT_EXP_OFFSET, rounded to nearest (even) at FP16's 10 mantissa
// bits; a power-of-two scale is exact. Zero (padding rows) stays zero. Out-of-range scales (outside the range
// above, which no valid cast produces) saturate to the FP16 max or flush to zero.
constexpr uint16_t fused_dequant_fp16_scale(uint32_t scale_fp32_bits) {
    const uint32_t biased = (scale_fp32_bits >> 23) & 0xff;
    const int32_t e = static_cast<int32_t>(biased) - 127 + FUSED_DEQUANT_EXP_OFFSET;
    if (biased == 0 || e < -14) {
        return 0;
    }
    if (e > 16) {
        return FP16_MAX_BITS;
    }
    const uint32_t mantissa = scale_fp32_bits & 0x7fffff;
    const uint32_t rounded = (mantissa + 0xfffu + ((mantissa >> 13) & 1u)) >> 13;  // may carry into the exponent
    const uint32_t bits = (static_cast<uint32_t>(e + 15) << 10) + rounded;
    return static_cast<uint16_t>(bits > FP16_MAX_BITS ? FP16_MAX_BITS : bits);
}

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
    FUSED_DEQUANT,  // scaled KV without sink: compute fuses tilize x scale (FP16 scale tiles)
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
    CB_HEAD_OUT,
    CB_HEAD_MAX,
    CB_HEAD_SUM,
    CB_HEAD_READY,
    CB_HEAD_ACK,
    CB_RECV_OUT,
    CB_RECV_MAX,
    CB_RECV_SUM,
    CB_MERGE_CTRL,
    FUSED_DEQUANT,
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
    CB_HEAD_READY,
    CB_HEAD_ACK,
    CB_RECV_OUT,
    CB_RECV_MAX,
    CB_RECV_SUM,
    CB_MERGE_CTRL,
    CB_CORR2,
    FUSED_DEQUANT,
    END,
};
}  // namespace compute_ct_arg

}  // namespace sparse_sdpa
