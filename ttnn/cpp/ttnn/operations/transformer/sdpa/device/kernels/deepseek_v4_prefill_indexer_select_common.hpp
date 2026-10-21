// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Host/device constants shared by the deepseek_v4_prefill_indexer_select factory and its kernels.
//
// Work unit: one tile row of 32 queries against one contiguous range of 512-entry blocks of idx_keys. A query tile
// whose visible entries span many blocks is split across NUM_SPLITS cores; split 0 (the leader) merges the other
// splits' top-k survivors, which they leave in a DRAM scratch tensor, and writes the output rows.
namespace deepseek_v4_prefill_indexer_select {

constexpr uint32_t TOPK = 512;          // entries kept per query, and entries per top-k chunk (block)
constexpr uint32_t ROWS = 32;           // queries per work unit
constexpr uint32_t STEP_KEY_TILES = 4;  // key tiles per matmul / gate pass; TOPK / 32 / STEP_KEY_TILES steps
constexpr uint32_t HEADS_PER_PASS = 8;  // q heads per matmul DEST pass (fp32 full-sync DEST holds 8 tiles)
constexpr uint32_t MAX_SPLITS = 8;      // cores one query tile's blocks may be split across
constexpr uint32_t GROUP_BLOCKS = 4;    // blocks scored before each top-k merge into the survivors
constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
constexpr uint32_t START_PAGE_BYTES = 64;
constexpr uint32_t TOKEN_PAGE_BYTES = 16;

// Reader -> compute mailbox, one page per work unit (uint32 words).
namespace meta_word {
enum : uint32_t {
    NUM_BLOCKS,   // blocks this split scans
    FIRST_ENTRY,  // entry index of its first block (a multiple of TOPK)
    TILE_BLOCKS,  // blocks visible to the whole query tile (shared by all its splits)
    LIMITS,       // ROWS words: query s sees entries [0, limit_s)
    END = LIMITS + ROWS,
};
}
constexpr uint32_t META_PAGE_BYTES = 256;
static_assert(meta_word::END * sizeof(uint32_t) <= META_PAGE_BYTES);

// The splits of a query tile share its visible blocks evenly; split s scans [first_block(s), first_block(s + 1)).
// Rounding up gives the leader (split 0) a block whenever the tile sees any; splits left empty are the last ones.
inline uint32_t split_first_block(uint32_t split, uint32_t tile_blocks, uint32_t num_splits) {
    return (split * tile_blocks + num_splits - 1) / num_splits;
}

// Per-core runtime args, identical for the three kernels.
namespace core_arg {
enum : uint32_t {
    FIRST_Q_TILE,
    Q_TILE_STRIDE,
    NUM_Q_TILES,
    SPLIT,
    LEADER_X,  // leader's NoC coordinates (split > 0 only)
    LEADER_Y,
    END,
};
}

namespace reader_common_arg {
enum : uint32_t { Q_ADDRESS, W_ADDRESS, KEYS_ADDRESS, START_ADDRESS, NUM_ENTRIES, END };
}
namespace writer_common_arg {
enum : uint32_t { OUTPUT_ADDRESS, SCRATCH_ADDRESS, END };
}

namespace reader_ct_arg {
enum : uint32_t {
    NUM_HEADS = 0,
    HEAD_DIM,
    NUM_TOKENS,
    RATE,  // query s (absolute position p) sees entry t iff t < (p + 1) / RATE
    NUM_SPLITS,
    CB_Q,
    CB_W,
    CB_W_IN,
    CB_K_RM,
    CB_NEGINF,
    CB_META,
    CB_START,
    CB_PLAN,
    END,
};
}

namespace writer_ct_arg {
enum : uint32_t {
    NUM_TOKENS = 0,
    NUM_SPLITS,
    CB_SURV_V,
    CB_SURV_I,
    CB_PEER,
    CB_OUT,
    CB_DONE,
    CB_PLAN,
    SEMAPHORE,
    END,
};
}

namespace compute_ct_arg {
enum : uint32_t {
    NUM_HEADS = 0,
    HEAD_DIM_TILES,
    NUM_SPLITS,
    CB_Q,
    CB_W,
    CB_K_RM,
    CB_K,
    CB_QK,
    CB_SCORES,
    CB_NEGINF,
    CB_SURV_V,
    CB_SURV_I,
    CB_PEER,
    CB_OUT,
    CB_META,
    CB_DONE,
    END,
};
}

}  // namespace deepseek_v4_prefill_indexer_select
