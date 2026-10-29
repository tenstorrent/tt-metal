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

// ROW_BFP4 (fork option): a row record = its k_dim / 16 shared exponents twice (copy 0 at byte 0, copy 1 at byte
// k_dim / 16), then k_dim / 2 bytes of nibbles (value 2i in the low nibble of byte i). A gathered row's exponents go to
// exps + p (k_dim / 16) of the staging block: the copy whose offset is congruent with that destination mod 64 (the
// DRAM read alignment) is read. The block [k_chunk rows' exponents | k_chunk rows' nibbles] is then a row-major
// bfloat4_b tensor the unpacker reads by datum index, one face row (16 values, 1 exponent) per UNPACR.
constexpr uint32_t ROW_BFP4_GROUP = 16;
constexpr uint32_t row_bfp4_exp_bytes(uint32_t k_dim) { return k_dim / ROW_BFP4_GROUP; }
constexpr uint32_t row_bfp4_bytes(uint32_t k_dim) { return 2 * row_bfp4_exp_bytes(k_dim) + k_dim / 2; }
constexpr uint32_t ROW_BFP4_ALIGN = 64;
// staging block: k_chunk rows' exponents, then their nibbles, + slack to put the block on a 64 B boundary
constexpr uint32_t row_bfp4_block_bytes(uint32_t k_dim, uint32_t k_chunk) {
    return k_chunk * (row_bfp4_exp_bytes(k_dim) + k_dim / 2) + ROW_BFP4_ALIGN;
}
// bfloat8_b row records (the window: kv_window uint8, or kv with ROW_BFP8): the same with one byte a value
constexpr uint32_t row_bfp8_bytes(uint32_t k_dim) { return 2 * row_bfp4_exp_bytes(k_dim) + k_dim; }
constexpr uint32_t row_bfp8_block_bytes(uint32_t k_dim, uint32_t k_chunk) {
    return k_chunk * (row_bfp4_exp_bytes(k_dim) + k_dim) + ROW_BFP4_ALIGN;
}

// Reader -> writer dual-NoC gather request. SCALE_DST_L1 is used only by the scaled-cache path.
namespace gather_request {
constexpr uint32_t BASE = 0;
constexpr uint32_t SPLIT = 1;
constexpr uint32_t IS_LAST = 2;
constexpr uint32_t DST_L1 = 3;
constexpr uint32_t SCALE_DST_L1 = 4;
constexpr uint32_t KIND = 5;         // ROW_BFP4: KIND_WINDOW (bf16 rows of kv_window) or KIND_BFP4 (row records)
constexpr uint32_t WORD_COUNT = 6;
constexpr uint32_t PAGE_BYTES = align_cb_page(WORD_COUNT * sizeof(uint32_t));
constexpr uint32_t KIND_WINDOW = 0;
constexpr uint32_t KIND_BFP4 = 1;
}  // namespace gather_request

// Reader -> compute token control message.
namespace control_message {
constexpr uint32_t ACTIVE_CHUNKS = 0;
constexpr uint32_t VALID_KEYS = 1;
constexpr uint32_t WINDOW_VALID = 2;  // ROW_BFP4: valid keys of chunk 0 (the window)
constexpr uint32_t WORD_COUNT = 3;
constexpr uint32_t PAGE_BYTES = align_cb_page(WORD_COUNT * sizeof(uint32_t));
}  // namespace control_message

constexpr uint32_t ACK_PAGE_BYTES = CB_PAGE_ALIGNMENT;

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
    END,
};
}  // namespace compute_ct_arg

// SPARSE_SDPA_ROW_BFP4 (fork option): extra compile-time args after the END of each block (reader / writer:
// staging row CB, tile CB, staging row stride; compute: the tile CB), then the tensor accessors.
// SPARSE_SDPA_WINDOW_BFP8 (with it): the window's bfloat8_b staging row CB and tile CB follow (COUNT_BFP8).
namespace row_bfp4_ct_arg {
enum : uint32_t { CB_K4_RM = 0, CB_K4, K_CHUNK, COUNT, CB_K8_RM = COUNT, CB_K8, COUNT_BFP8 };
}
#if defined(SPARSE_SDPA_ROW_BFP4) && defined(SPARSE_SDPA_WINDOW_BFP8)
constexpr uint32_t ROW_BFP4_CT_COUNT = row_bfp4_ct_arg::COUNT_BFP8;
constexpr uint32_t ROW_BFP4_COMMON_COUNT = 1;  // the window's address, after each kernel's fixed common args
#elif defined(SPARSE_SDPA_ROW_BFP4)
constexpr uint32_t ROW_BFP4_CT_COUNT = row_bfp4_ct_arg::COUNT;
constexpr uint32_t ROW_BFP4_COMMON_COUNT = 1;  // the kv_window address, after each kernel's fixed common args
#else
constexpr uint32_t ROW_BFP4_CT_COUNT = 0;
constexpr uint32_t ROW_BFP4_COMMON_COUNT = 0;
#endif

}  // namespace sparse_sdpa
