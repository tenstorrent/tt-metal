// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Host/device constants shared by the sparse-SDPA factory and JIT kernels. Keep this header free of host-only and
// kernel-only dependencies so both compilation paths use the same protocol and packed-cache geometry.
namespace sparse_sdpa {

constexpr uint32_t SCALE_BLOCK_WIDTH = 128;
constexpr uint32_t CB_PAGE_ALIGNMENT = 16;
constexpr uint32_t CB_DOUBLE_BUFFER_DEPTH = 2;

constexpr uint32_t align_cb_page(uint32_t bytes) {
    return ((bytes + CB_PAGE_ALIGNMENT - 1) / CB_PAGE_ALIGNMENT) * CB_PAGE_ALIGNMENT;
}

constexpr uint32_t scale_block_count(uint32_t latent_width) { return latent_width / SCALE_BLOCK_WIDTH; }

constexpr uint32_t packed_kv_width(uint32_t k_dim, uint32_t latent_width) {
    return latent_width + scale_block_count(latent_width) * sizeof(float) + (k_dim - latent_width) * sizeof(uint16_t);
}

// Reader -> writer dual-NoC gather request. SCALE_DST_L1 is used only by the scaled-cache path.
namespace gather_request {
constexpr uint32_t BASE = 0;
constexpr uint32_t SPLIT = 1;
constexpr uint32_t IS_LAST = 2;
constexpr uint32_t DST_L1 = 3;
constexpr uint32_t SCALE_DST_L1 = 4;
constexpr uint32_t WORD_COUNT = 5;
constexpr uint32_t PAGE_BYTES = align_cb_page(WORD_COUNT * sizeof(uint32_t));
}  // namespace gather_request

// Reader -> compute token control message.
namespace control_message {
constexpr uint32_t ACTIVE_CHUNKS = 0;
constexpr uint32_t VALID_KEYS = 1;
constexpr uint32_t WORD_COUNT = 2;
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
    CB_K_RM,
    CB_IDX,
    CB_KREQ,
    CB_KACK,
    PACKED_ROW_BYTES,
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
    END,
};
}  // namespace compute_ct_arg

}  // namespace sparse_sdpa
