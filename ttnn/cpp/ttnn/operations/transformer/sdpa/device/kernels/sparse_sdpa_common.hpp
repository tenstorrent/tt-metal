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
// Q.K is unchanged) and the normalized output by 2^-OFFSET (V, the latent, carries 2^OFFSET). All are power-of-two
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

// Circular buffers beside the recipe compute layout (CBs 0-16 and 29-31, recipe_compute_program; compute and the writer
// use recipe_ring_cb's names for those and for the checkpoint pair). Compute packs only into CBs below 32.
namespace recipe_cb {
// Split-K HEAD: compute's state checkpoint request / ack (recipe_checkpoint.hpp; the writer sends the state to core
// i-1).
constexpr uint32_t checkpoint_request = 17, checkpoint_ack = 18;
// Per-row sinks: one column tile per Q tile row (recipe::sink_rows), built once by the writer.
constexpr uint32_t sink = 19;
constexpr uint32_t q_rm = 20;          // Q rows (row-major, reader -> compute tilize), one tile row of heads per push
constexpr uint32_t k_rm = 21;          // gathered K rows of a chunk (row-major), or a packed scaled-FP8 slab
constexpr uint32_t k_rope_rm = 22;     // scaled FP8: format-only BF16 view of the packed slab (RoPE field)
constexpr uint32_t k_scale = 23;       // scaled FP8: per-row broadcast scale tiles (FP32, or FP16 with fused dequant)
constexpr uint32_t k_latent = 24;      // scaled FP8 without fused dequant: one slab's BF16 latent tiles
constexpr uint32_t out_rm = 25;        // untilized output rows (compute -> writer)
constexpr uint32_t ctrl = 26;          // reader -> compute: per-segment control_message
constexpr uint32_t kreq = 27;          // reader -> writer K-gather handoff (dual-NoC split)
constexpr uint32_t kack = 28;          // writer -> reader: its half of the chunk landed
constexpr uint32_t merge_corr = 15;    // split-K TAIL: exp(m_b - m) per Q row (the recipe's mask CB, unused here)
constexpr uint32_t idx = 33;           // reader-internal: one token's index row (uint32); the writer reads it too
constexpr uint32_t sink_scratch = 34;  // writer scratch for gathering sink scalars
// Split-K TAIL: core i+1's HEAD state, NoC-written straight into these CBs' bases (the same L1 on every core): O rows
// (vd FP32 tiles each), maxima (the recipe's max format), l (FP32); then the writer's merge control page.
constexpr uint32_t recv_out = 35, recv_max = 36, recv_sum = 37, merge_ctrl = 38;
}  // namespace recipe_cb

// Compile-time argument positions. The factory checks the vector size against each END value before appending
// TensorAccessorArgs, while the kernels use the same names to decode the stream.
namespace reader_ct_arg {
enum : uint32_t {
    H = 0,
    S,
    TOPK,
    K_CHUNK,
    K_DIM,
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
    PACKED_ROW_BYTES,
    SPLIT_K,
    FUSED_DEQUANT,  // scaled KV without sink: compute fuses tilize x scale (FP16 scale tiles)
    Q_TILES,        // Q tile rows per head pass (with padding rows)
    PASSES,         // head passes per token
    END,
};
}  // namespace reader_ct_arg

namespace writer_ct_arg {
enum : uint32_t {
    H = 0,
    S,
    V_DHT,
    OUT_ELEM_BYTES,
    BLOCK_CYCLIC,
    BC_CHUNK_LOCAL,
    BC_SP,
    BC_SHARD_STRIDE_GAP,
    BC_SLAB_STRIDE_GAP,
    SCALED_KV,
    K_DIM,
    KV_ELEM_BYTES,
    PACKED_ROW_BYTES,
    USE_ATTENTION_SINK,
    SPLIT_K,
    FUSED_DEQUANT,
    Q_TILES,
    PASSES,
    FP32_STATE,  // the recipe's state maxima are FP32 tiles (BALANCED, ACCURATE), else BF16
    END,
};
}  // namespace writer_ct_arg

}  // namespace sparse_sdpa
