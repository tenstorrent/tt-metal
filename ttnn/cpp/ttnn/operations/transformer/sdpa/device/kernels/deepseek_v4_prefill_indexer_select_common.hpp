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

// Candidate mode (the two-level select's level one): each query also keeps its best CAND_TOPK blocks of CAND_BLOCK
// entries, a block scored by its best entry, the block holding the query's newest entry always kept.
constexpr uint32_t CAND_TOPK = 2048;
constexpr uint32_t CAND_BLOCK = 8;
constexpr uint32_t CAND_PER_BLOCK = TOPK / CAND_BLOCK;              // candidate blocks per 512-entry block
constexpr uint32_t CAND_CHUNK_BLOCKS = CAND_TOPK / CAND_PER_BLOCK;  // 512-entry blocks per candidate top-k chunk
constexpr uint32_t CAND_ROW_BYTES = CAND_TOPK * sizeof(uint16_t);   // one query's staged block maxima (bf16)
constexpr uint32_t CAND_SURV_TILES = 4;                             // fp32 [values x 2, indices x 2] tiles
static_assert(CAND_CHUNK_BLOCKS % GROUP_BLOCKS == 0, "candidate chunks must close on group boundaries");
constexpr uint16_t BF16_NEG_INF = 0xFF80u;
constexpr uint16_t BF16_POS_INF = 0x7F80u;
constexpr uint32_t NEGINF_TILES = 2;  // the all -inf seed chunk covers CAND_TOPK bf16 entries

// Candidate blocks a query with `limit` visible entries keeps, at most CAND_TOPK of them, from candidate chunk
// `chunk` of a split that scans 512-entry blocks [first_block, end_block): 0 when the split shows it none of the
// chunk's blocks.
inline uint32_t cand_active(uint32_t limit, uint32_t chunk, uint32_t first_block, uint32_t end_block) {
    const uint32_t visible = (limit + CAND_BLOCK - 1) / CAND_BLOCK;
    const uint32_t lo = first_block * CAND_PER_BLOCK;
    const uint32_t split_hi = end_block * CAND_PER_BLOCK;
    const uint32_t hi = visible < split_hi ? visible : split_hi;
    const uint32_t base = chunk * CAND_TOPK;
    if (hi <= lo || hi <= base) {
        return 0;
    }
    return hi - base < CAND_TOPK ? hi - base : CAND_TOPK;
}

// Whether candidate chunk `chunk` is the split's last one with blocks visible to the query.
inline bool cand_last_chunk(uint32_t limit, uint32_t chunk, uint32_t end_block) {
    const uint32_t visible = (limit + CAND_BLOCK - 1) / CAND_BLOCK;
    const uint32_t split_hi = end_block * CAND_PER_BLOCK;
    const uint32_t hi = visible < split_hi ? visible : split_hi;
    return (chunk + 1) * CAND_TOPK >= hi;
}

// Pool mode (the two-level select's level two): the work unit is one query, scored against only the entries of
// its candidate blocks (its pool, at most CAND_TOPK * CAND_BLOCK entries) and selecting the top TOPK of them.
// Pool position p is entry list[p / CAND_BLOCK] * CAND_BLOCK + p % CAND_BLOCK, list being the query's visible
// candidate blocks with the one holding its newest entry (the only partial one) last.
constexpr uint32_t POOL_LIST_WORDS = 16;  // header words before the list
constexpr uint32_t POOL_LIST_PAGE_BYTES = (POOL_LIST_WORDS + CAND_TOPK) * sizeof(uint32_t);
constexpr uint32_t POOL_PIECE_BYTES = 64;  // aligned read holding one 16-datum face row of a bf16 tile

// Reader -> compute mailbox (and its copy to the writer), one page per work unit (uint32 words).
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
enum : uint32_t { OUTPUT_ADDRESS, SCRATCH_ADDRESS, CAND_OUTPUT_ADDRESS, CAND_SCRATCH_ADDRESS, END };
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
    CANDIDATES,  // 1: candidate mode
    CB_SCORES,
    CB_GROUP,
    CB_GROUP_ACK,
    CB_STAGE,
    CB_CSURV_IN,
    CB_CSURV_OUT,
    CB_CAND_OUT,
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
    CANDIDATES,
    CB_GROUP,
    CB_GROUP_ACK,
    CB_STAGE,
    CB_CSURV_IN,
    CB_CSURV_OUT,
    CB_CAND_OUT,
    POOL,  // 1: pool mode (one query per unit, q as [heads, dims] tiles, one gate tile)
    END,
};
}

// Pool mode: per-core runtime args of its reader and writer (compute takes core_arg, one query per "tile").
namespace pool_core_arg {
enum : uint32_t { FIRST_QUERY, NUM_QUERIES, END };
}
namespace pool_reader_common_arg {
enum : uint32_t { Q_ADDRESS, W_ADDRESS, KEYS_ADDRESS, START_ADDRESS, CANDS_ADDRESS, NUM_ENTRIES, END };
}
namespace pool_writer_common_arg {
enum : uint32_t { OUTPUT_ADDRESS, END };
}
namespace pool_reader_ct_arg {
enum : uint32_t {
    NUM_HEADS = 0,
    HEAD_DIM,
    NUM_TOKENS,
    RATE,
    CB_Q,
    CB_W,
    CB_K_RM,
    CB_NEGINF,
    CB_META,
    CB_START,
    CB_LIST,    // reader -> writer: the query's pool list
    CB_BOUNCE,  // reader scratch: aligned reads of q and gate rows
    CB_CANDS,   // reader scratch: the query's candidate row
    END,
};
}
namespace pool_writer_ct_arg {
enum : uint32_t { CB_OUT = 0, CB_LIST, END };
}

}  // namespace deepseek_v4_prefill_indexer_select
