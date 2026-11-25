// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm: the one schema the program factory and the kernels share. Constants only (no
// get_compile_time_arg_val here), so the host includes it as is. Values are the generic op's.
#pragma once

#include <cstdint>

namespace agrn_args {

// ---- circular buffers (agrn_common.hpp; id 11 was never used and stays free) ----
namespace cb {
constexpr uint32_t X = 0;
constexpr uint32_t GAMMA = 1;
constexpr uint32_t BETA = 2;
constexpr uint32_t SCALER = 3;
constexpr uint32_t MASK = 4;
constexpr uint32_t SUMSQ = 5;
constexpr uint32_t LN_M = 6;
constexpr uint32_t LN_SHIFT = 7;
constexpr uint32_t LN_SQ = 8;
constexpr uint32_t PARTIAL = 9;
constexpr uint32_t RECORD_STAGING = 10;
constexpr uint32_t GATHERED = 12;
constexpr uint32_t STAT_CONSTS = 13;
constexpr uint32_t INV = 14;
constexpr uint32_t MEAN_COL = 15;
constexpr uint32_t MEAN_B = 16;
constexpr uint32_t RSTD_B = 17;
constexpr uint32_t LN_TMP = 18;
constexpr uint32_t OUT_BLOCK = 19;
constexpr uint32_t ASSEMBLE = 20;
constexpr uint32_t PORT_CHUNKS = 21;
constexpr uint32_t X32 = 22;
constexpr uint32_t GAMMA_ROWS = 23;
constexpr uint32_t GATHER_ROWS = 24;
constexpr uint32_t STATS_TABLE = 25;
constexpr uint32_t X_STATS = 26;
constexpr uint32_t PRE_TABLE = 27;
constexpr uint32_t FINAL_TABLE = 28;
constexpr uint32_t FINAL_INV = 29;
constexpr uint32_t FINAL_MEAN = 30;
constexpr uint32_t STAT_CONSTS_X = 31;
constexpr uint32_t PORT_DESC = 32;
constexpr uint32_t XG = 33;
constexpr uint32_t STAGE = 34;
constexpr uint32_t MAX_ID = STAGE;
}  // namespace cb

// ---- program-local semaphores (declared first on the full grid, in this order) ----
namespace sem {
constexpr uint32_t RECORDS_LOCAL = 0;
constexpr uint32_t STATS_READY = 1;
constexpr uint32_t OWN_READY = 2;
constexpr uint32_t LANDED = 3;
constexpr uint32_t STATS_READY_BWD = 4;
constexpr uint32_t STATS_DONE = 5;
constexpr uint32_t PRE = 6;
constexpr uint32_t FINAL = 7;
constexpr uint32_t COUNT = 8;
}  // namespace sem

constexpr uint32_t RECORD_BYTES = 128;    // 32 fp32 values: column 0 of one stat tile
constexpr uint32_t PORT_DESC_BYTES = 64;  // agrn_port_common.hpp DESC_BYTES

// ---- kernel order in every program descriptor ----
namespace kernel {
enum : uint32_t { READER, WRITER, COMPUTE, PORT_READER, PORT_SENDER, COUNT };
}  // namespace kernel

// ---- worker compile-time args (agrn_worker_args.hpp), same prefix on reader / writer / compute ----
namespace worker_ct {
enum : uint32_t {
    R,
    WLT,
    K,
    N_GROUPS,
    ROWS_HI,
    BIG,
    CW_NOM,
    GRID_X,
    G,
    NSTATS,
    GATHER_TILES,
    IS_LN,
    HAS_GAMMA,
    HAS_BETA,
    GAMMA_IS_TILE,
    X_PAGE_BYTES,
    OUT_PAGE_BYTES,
    GAMMA_ELEM_BYTES,
    W_VALID_LAST,
    W_LOCAL,
    WT,
    IS_RU,
    X_IS_F32,
    OUT_ELEM_BYTES,
    N_PORTS,
    ASSEMBLE_SRC_MAX,
    GAMMA_PAGE_BYTES,
    GAMMA_SRC_PAGE_BYTES,
    X_SHARDED,
    LAST_CRT,
    OUT_IN_PLACE,
    PRE_COMBINE,
    ROW_CHUNK,
    ROWS_PER_OWNER,
    OWN_BANDS,
    XG_PRE,
    // own-entry staging block
    STAGE_ON,
    ST_L,
    ST_NB,
    ST_COLS,
    ST_ROW_STRIDE,
    ST_SLOT_STRIDE,
    ST_BY_ROW,
    ST_LANES,                    // ST_MAX_LANES staged-unit counts
    ST_UNIT_OFF = ST_LANES + 4,  // ST_MAX_NB per-bank lane offsets
    ST_RP = ST_UNIT_OFF + 16,
    ST_NCH,
    ST_NACT,
    ST_TBL,  // common-arg index of the staging core table
    ST_CTR,  // port readiness counter (global semaphore address)
    COUNT    // TensorAccessorArgs follow on reader / writer: x, dst, out, gamma, beta
};
constexpr uint32_t ST_MAX_LANES = 4;
constexpr uint32_t ST_MAX_NB = 16;
static_assert(ST_UNIT_OFF - ST_LANES == ST_MAX_LANES && ST_RP - ST_UNIT_OFF == ST_MAX_NB);
}  // namespace worker_ct

// ---- worker common runtime args (agrn_worker_args.hpp CRT_*), shared by reader / writer / compute ----
// Fixed slots; then n_ports linear port indices, n_ports port NoC coords, K inv_n bits, inv_wl bits, own_bands band
// addresses, [pre_combine: K member NoC coords], [x_sharded: packed worker map], [stage_on: n_active staging coords].
namespace worker_common {
enum : uint32_t {
    X_ADDR,
    DST_ADDR,
    OUT_ADDR,
    GAMMA_ADDR,
    BETA_ADDR,
    EPS_BITS,
    INV_W_BITS,
    P,
    HUB_MASK,
    FWD_HUB,
    BWD_HUB,
    PORTS,  // start of the variable tail
};
}  // namespace worker_common

// ---- port compile-time args (agrn_port_common.hpp), reader and sender alike ----
namespace port_ct {
enum : uint32_t {
    CB_CHUNKS,
    PAGE_BYTES,
    RUN_PAGES,
    GROUP,
    INC_EVERY,
    NB,
    ROWS_EFF,
    COLS_EFF,
    ROW_STRIDE,
    SLOT_STRIDE,
    FLAT_SLOT,
    WLT,
    K,
    N_GROUPS,
    ROWS_HI,
    BIG,
    CW_NOM,
    PAYLOAD_BYTES,
    SCATTER_RUNS,
    SHARD_W,
    LAST_CRT,
    READ_DEPTH,
    OWN_FROM_DST,
    CB_DESC,
    SPLIT_RUNS,
    OWN_BANDS,
    EAGER_PUSH,
    CB_STAGE,
    STAGE_ON,
    COUNT  // TensorAccessorArgs(dst) follow
};
}  // namespace port_ct

// ---- port reader: common args (same on every port core) ----
namespace port_reader_common {
enum : uint32_t { DST_ADDR, BANK_STRIDE, ARRIVAL_ADDR, N_ACTIVE, NOTIFY_LANDED, COUNT };
}  // namespace port_reader_common

// ---- port reader: per-core args. Fixed head, then 3 * N_ENTRIES entry words, n_active worker NoC coords,
// HUB_SEM, MC_START, MC_END, MC_DESTS, STATS_ARRIVED_ADDR, SLOT_BYTES, MC_OWN_SLOT, N_REC_WRITERS, N_RECV, N_RECV
// slots, own_bands band_need, own_bands band_row_end, own_bands band addresses, ST_ON [, 7 staging words].
namespace port_reader_rt {
enum : uint32_t { FIRST_BANK, EXPECT_IN, N_ENTRIES, ENTRIES };
constexpr uint32_t WORDS_PER_ENTRY = 3;
constexpr uint32_t HUB_TAIL_FIXED = 9;    // HUB_SEM .. N_RECV
constexpr uint32_t STAGE_TAIL_WORDS = 7;  // st_pages, st_ctr, st_nact, st_nch, st_lane, st_L, st_cp
}  // namespace port_reader_rt

// ---- port sender: common args (same on every port core) ----
namespace port_sender_common {
enum : uint32_t { DST_ADDR, BANK_STRIDE, ARRIVAL_ADDR, READY_ADDR, SLOT_BYTES, STATS_ARRIVED_ADDR, N_ACTIVE, COUNT };
}  // namespace port_sender_common

// ---- port sender: per-core args. Fixed head, then N_STATS slots, n_entries, 2 * n_entries entry words, then the
// fabric connection args (appended last).
namespace port_sender_rt {
enum : uint32_t {
    FIRST_BANK,
    SEND_READY,
    READY_XY,
    PEER_XY,
    DST_MESH_ID,
    DST_CHIP_ID,
    HAS_CONN,
    IS_HUB,
    STATS_EXPECT_IN,
    N_STATS,
    STATS  // start of the variable tail
};
constexpr uint32_t WORDS_PER_ENTRY = 2;
}  // namespace port_sender_rt

}  // namespace agrn_args
