// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, argument slots and constants shared by the groupnorm_sc_N_1_HW_C_b kernels and the program
// factory. Values are the generic op's, unchanged. Every runtime arg stays per-core, in the Python's order.
namespace gn_sc_args {

namespace cb {
constexpr uint32_t INPUT_RM = 0;
constexpr uint32_t OUTPUT_RM = 1;
constexpr uint32_t INPUT_TILES = 2;
constexpr uint32_t INPUT_KEEP = 3;
constexpr uint32_t OUTPUT_TILES = 5;
constexpr uint32_t NEG_ONE = 6;
constexpr uint32_t SUM_ACC = 7;
constexpr uint32_t SUMSQ_ACC = 8;
constexpr uint32_t CHAN_SCALE = 9;
constexpr uint32_t CHAN_WORK = 10;
constexpr uint32_t GROUP_MAP = 12;
constexpr uint32_t GAMMA_ROWS = 13;
constexpr uint32_t LOCAL_PARTIALS = 15;
constexpr uint32_t GATHERED_PARTIALS = 16;
constexpr uint32_t STATS_OUT = 19;
constexpr uint32_t GROUP_STATS = 20;
constexpr uint32_t AFFINE_STAGING = 21;
constexpr uint32_t PILOT_ROWS = 22;
constexpr uint32_t GROUP_SHIFT = 24;
constexpr uint32_t RM_STAGE_IN = 25;
constexpr uint32_t RM_STAGE_OUT = 26;
constexpr uint32_t OUTPUT_B = 27;
constexpr uint32_t OUT_SEQ_A = 28;
constexpr uint32_t OUT_SEQ_B = 29;
constexpr uint32_t MAX_ID = OUT_SEQ_B;
}  // namespace cb

// The writer reads its gather semaphore at this id; the factory checks find_available_semaphore_id returns it.
constexpr uint32_t GATHER_SEMAPHORE_ID = 0;

// Sharded op words (compute per-core tail).
constexpr uint32_t OP_ROWS = 0xFFFF;
constexpr uint32_t OP_STAGED = 1u << 16;
constexpr uint32_t OP_SEG_END = 1u << 17;
constexpr uint32_t OP_SKIP = 1u << 18;
constexpr uint32_t OUT_SEQ_END = 0xFFFFFFFFu;
constexpr uint32_t OUT_SEQ_PAGES = 8;
constexpr uint32_t OUT_SEQ_PAGE_BYTES = 16;
constexpr uint32_t STAGED_WORDS = 3;  // (shard stick, position, sticks) per staged row

// Reader compile-time args; TensorAccessorArgs of input, gamma, beta follow from COUNT, then OUT_PAGE_BYTES, then
// the output's TensorAccessorArgs.
namespace reader_ct {
enum : uint32_t {
    W,
    CHUNK_HT,
    IS_RM,
    HAS_GAMMA,
    HAS_BETA,
    R,
    CG,
    WT_TOTAL,
    HT,
    GX,
    IN_PAGE_BYTES,
    MAP_PERIOD,
    SHARDED,
    CHUNK_ZONES,
    HAS_STAGE,
    OPS,
    IN_TILE_BYTES,
    AFF_ELEM,
    AFF_TILE,
    AFF_PAGE_BYTES,
    HW,
    VC_LAST,
    IN_BFP8,
    AFF_BFP8,
    COUNT
};
}  // namespace reader_ct

namespace compute_ct {
enum : uint32_t {
    W,
    CHUNK_HT,
    IS_RM,
    HAS_GAMMA,
    HAS_BETA,
    R,
    SHARDED,
    IS_ROOT,
    CHUNK_ZONES,
    NU,
    MAP_PERIOD,
    COL_MASK,
    HAS_STAGE,
    OPS,
    GX,
    VC_LAST,
    COUNT
};
}  // namespace compute_ct

// Writer compile-time args; the output's TensorAccessorArgs follow from COUNT, then Mcast::attach appends its own.
namespace writer_ct {
enum : uint32_t {
    W,
    CHUNK_HT,
    R,
    WT_TOTAL,
    HT,
    GX,
    OUT_PAGE_BYTES,
    IS_ROOT,
    IS_RM,
    SHARDED,
    CHUNK_ZONES,
    NU,
    HAS_STAGE,
    OPS,
    MAP_PERIOD,
    VC_LAST,
    COUNT
};
}  // namespace writer_ct

// Reader per-core args: fixed slots, then N_COLS pilot-column virtual x's, then N_STAGED, then 3 words per staged
// row.
namespace reader_rt {
enum : uint32_t {
    IN_ADDR,
    GAMMA_ADDR,
    BETA_ADDR,
    UNIT_START,
    NUM_BLOCKS,
    ROW_START,
    ROWS_CORE,
    RESIDENT_CHUNKS,
    IS_ROOT,
    ROOT_X,
    ROOT_Y,
    OUT_ADDR,
    XS,
    VC,
    L0,
    OFF0,
    G_UNIT,
    RM_CREDIT_PAGES,
    N_COLS,
    COLS_TAIL  // first pilot column; N_STAGED sits at COLS_TAIL + n_cols
};
}  // namespace reader_rt

// Compute per-core args: fixed slots, then N_OPS op words.
namespace compute_rt {
enum : uint32_t {
    NUM_BLOCKS,
    ROWS_CORE,
    RESIDENT_CHUNKS,
    INV_N_BITS,
    EPS_BITS,
    N_IMG,
    SEG_IMG0,
    W_VALID,
    VC,
    UNIT_START,
    N_OPS,
    OPS_TAIL
};
}  // namespace compute_rt

// Writer per-core args: fixed slots, then 3 words per staged row, then the mcast args Mcast::attach appends.
namespace writer_rt {
enum : uint32_t {
    OUT_ADDR,
    UNIT_START,
    NUM_BLOCKS,
    ROW_START,
    ROWS_CORE,
    RESIDENT_CHUNKS,
    SLOT,
    ROOT_X,
    ROOT_Y,
    CPI,
    N_IMG,
    N_SEG,
    SEG_IMG0,
    RM_CREDIT_PAGES,
    N_STAGED,
    STAGED_TAIL
};
}  // namespace writer_rt

}  // namespace gn_sc_args
