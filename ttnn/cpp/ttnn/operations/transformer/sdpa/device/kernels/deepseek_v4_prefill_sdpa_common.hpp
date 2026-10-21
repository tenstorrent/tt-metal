// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Host/device constants shared by the deepseek_v4_prefill_sdpa factory and its dataflow kernels. The compute
// kernel is sparse_sdpa's, so its compile-time arguments, the reader -> compute control message and the
// reader -> writer gather request follow sparse_sdpa_common.hpp.
namespace deepseek_v4_prefill_sdpa {

// A key-list entry with this bit set is a row of window_kv; without it, a row of cache_kv. Both tensors are far
// below 2^31 rows.
constexpr uint32_t WINDOW_ROW = 0x80000000u;
constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
constexpr uint32_t START_PAGE_BYTES = 64;

namespace reader_common_arg {
enum : uint32_t { Q_ADDRESS, WINDOW_ADDRESS, CACHE_ADDRESS, INDICES_ADDRESS, START_ADDRESS, END };
}
namespace writer_common_arg {
enum : uint32_t { OUTPUT_ADDRESS, WINDOW_ADDRESS, CACHE_ADDRESS, ATTENTION_SINK_ADDRESS, END };
}

namespace reader_ct_arg {
enum : uint32_t {
    H = 0,
    S,
    TOPK,  // cache_idx width; 0 without a cache
    LIST_CAPACITY,
    K_CHUNK,
    K_DIM,
    WINDOW,
    CB_Q_RM,
    CB_K_RM,
    CB_MASK_PART,
    CB_IDX,
    CB_CTRL,
    CB_KREQ,
    CB_KACK,
    CB_START,
    COMPRESS_RATIO,  // cache row w is visible to query s iff w < (start + s + 1) / ratio; 0: every row is
    EMPTY_TAIL,      // window_kv's first WINDOW rows hold no token, whatever start is
    END,
};
}  // namespace reader_ct_arg

namespace writer_ct_arg {
enum : uint32_t {
    H = 0,
    S,
    V_DHT,
    K_DIM,
    CB_OUT_RM,
    CB_SCALE,
    CB_COL_IDENTITY,
    CB_NEGINF,
    CB_IDX,
    CB_KREQ,
    CB_KACK,
    USE_ATTENTION_SINK,
    CB_ATTENTION_SINK,
    CB_SINK_SCRATCH,
    END,
};
}  // namespace writer_ct_arg

}  // namespace deepseek_v4_prefill_sdpa
