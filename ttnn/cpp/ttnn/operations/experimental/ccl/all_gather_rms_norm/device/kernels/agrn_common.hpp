// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm — shared by every kernel of the op.
//   * CB indices (must match the CB_* constants in all_gather_rms_norm_program_descriptor.py).
//   * Program-local semaphore ids (declared on the full program core range, before the fabric connections).
//   * Worker identity: a worker kernel takes COMMON runtime args only (no per-core args — dispatch cost); it derives
//     its (row-group, W member) from its logical position and the CT/common list of port cores, exactly as the host
//     select_split() / worker ordering does (row-major logical order with the port cores removed).
#pragma once

#include <cstdint>

#include "all_gather_rms_norm_args.hpp"

// Per-stage profiler zones, opt-in (host define AGRN_PROFILE_ZONES, env AGRN_PROFILE_ZONES=1): they would otherwise
// fill the profiler buffers the perf harness reads its per-program times from.
#if defined(AGRN_PROFILE_ZONES)
#include "tools/profiler/kernel_profiler.hpp"
#define AGRN_ZONE(name) DeviceZoneScopedN(name)
#else
#define AGRN_ZONE(name)
#endif

namespace agrn {

// ---- circular buffers ----
constexpr uint32_t cb_x = agrn_args::cb::X;                // resident x block (crt x cw tiles, row pitch cw)
constexpr uint32_t cb_gamma = agrn_args::cb::GAMMA;        // gamma slice: compact pages, row 0 of faces 0/1
constexpr uint32_t cb_beta = agrn_args::cb::BETA;          // beta slice (LN), same page format as gamma
constexpr uint32_t cb_scaler = agrn_args::cb::SCALER;      // 1.0 reduce scaler (bf16)
constexpr uint32_t cb_mask = agrn_args::cb::MASK;          // LN: 0/1 row mask of the W-tail tile
constexpr uint32_t cb_sumsq = agrn_args::cb::SUMSQ;        // RMS: per tile-row DEST-accumulated sum of x^2
constexpr uint32_t cb_ln_m = agrn_args::cb::LN_M;          // LN: local row mean (col 0), source of the shift
constexpr uint32_t cb_ln_shift = agrn_args::cb::LN_SHIFT;  // LN: K = tf32(m) broadcast to every column
constexpr uint32_t cb_ln_sq = agrn_args::cb::LN_SQ;        // LN: (x - K)^2 tiles of the block
constexpr uint32_t cb_partial = agrn_args::cb::PARTIAL;    // per-row stats (col 0), Float32 wire contract
constexpr uint32_t cb_record_staging = agrn_args::cb::RECORD_STAGING;  // writer scratch: 128 B records
constexpr uint32_t cb_gathered = agrn_args::cb::GATHERED;              // Float32 gather tiles, col j = partition j
constexpr uint32_t cb_stat_consts = agrn_args::cb::STAT_CONSTS;        // LN: n_j per partition column (Float32)
constexpr uint32_t cb_inv = agrn_args::cb::INV;                        // RMS inv_rms / LN rstd (col 0)
constexpr uint32_t cb_mean_col = agrn_args::cb::MEAN_COL;              // LN: exact row mean (col 0)
constexpr uint32_t cb_mean_b = agrn_args::cb::MEAN_B;                  // LN: d = mean - K broadcast
constexpr uint32_t cb_rstd_b = agrn_args::cb::RSTD_B;                  // LN: rstd broadcast to every column
constexpr uint32_t cb_ln_tmp = agrn_args::cb::LN_TMP;                  // LN combine scratch tiles (Float32)
constexpr uint32_t cb_out_block = agrn_args::cb::OUT_BLOCK;            // normalized own block, resident (ports pull it)
constexpr uint32_t cb_assemble = agrn_args::cb::ASSEMBLE;              // R_U writer scratch
constexpr uint32_t cb_port_chunks = agrn_args::cb::PORT_CHUNKS;        // port: 2 x batch_chunks packets
constexpr uint32_t cb_x32 = agrn_args::cb::X32;                        // LN, non-Float32 x: exact Float32 copy
constexpr uint32_t cb_gamma_rows = agrn_args::cb::GAMMA_ROWS;          // reader scratch: raw gamma / beta source rows
constexpr uint32_t cb_gather_rows =
    agrn_args::cb::GATHER_ROWS;  // Float32 gather tiles in ROW form (row j = partition j), read straight
                                 // from the hub tables; compute transposes them into cb_gathered
constexpr uint32_t cb_stats_table =
    agrn_args::cb::STATS_TABLE;  // hubs: the per-call stats table (G slots of R x K x nstats records).
                                 // A program-scoped CB on the full core range: the same L1 address on
                                 // every core of every chip (identical CB lists), live only while the
                                 // program runs (peers write into it only after the ready fence)
// WIDTH_SHARDED row-owner pre-combine (pre_combine)
constexpr uint32_t cb_x_stats =
    agrn_args::cb::X_STATS;  // second zero-copy view of the input shard, popped per row chunk by the
                             // local-stats pass (cb_x stays whole for the normalize)
constexpr uint32_t cb_pre_table =
    agrn_args::cb::PRE_TABLE;  // row owner: [row][stat][half][member] 64 B records of the k members
constexpr uint32_t cb_final_table =
    agrn_args::cb::FINAL_TABLE;  // member: [row][final stat] 128 B records from the row owners
constexpr uint32_t cb_final_inv =
    agrn_args::cb::FINAL_INV;  // reader -> compute: RMS inv / LN rstd in column 0, one row chunk
constexpr uint32_t cb_final_mean = agrn_args::cb::FINAL_MEAN;  // reader -> compute: LN mean in column 0, one row chunk
constexpr uint32_t cb_stat_consts_x =
    agrn_args::cb::STAT_CONSTS_X;                    // LN: n_q of the cross-chip partitions (one per chip)
constexpr uint32_t cb_stage = agrn_args::cb::STAGE;  // own-entry staging: run_pages-page chunks of the ports' own walks
constexpr uint32_t cb_xg =
    agrn_args::cb::XG;  // RMS (xg_pre): x * gamma of the block, computed in the stats shadow (T_acc)

// ---- program-local semaphores ----
constexpr uint32_t sem_records_local =
    agrn_args::sem::RECORDS_LOCAL;  // hubs: one inc per active worker (records written into my table)
constexpr uint32_t sem_stats_ready =
    agrn_args::sem::STATS_READY;  // workers: one inc per hub (all slots of the stats table landed)
constexpr uint32_t sem_own_ready =
    agrn_args::sem::OWN_READY;  // ports: one inc per active worker (own block written + resident)
constexpr uint32_t sem_landed = agrn_args::sem::LANDED;  // workers (R_U): one inc per port (its arrivals landed)
constexpr uint32_t sem_stats_ready_bwd =
    agrn_args::sem::STATS_READY_BWD;  // workers: set to 1 by the bwd hub's multicast (fwd hub: sem_stats_ready)
constexpr uint32_t sem_stats_done =
    agrn_args::sem::STATS_DONE;  // hubs: the sender tells its own reader the table is complete
constexpr uint32_t sem_pre =
    agrn_args::sem::PRE;  // row owner: one inc per member (its records of my rows are in my pre table)
constexpr uint32_t sem_final =
    agrn_args::sem::FINAL;  // member: one inc per row owner (its final records are in my final table)

constexpr uint32_t RECORD_BYTES =
    agrn_args::RECORD_BYTES;  // 32 fp32 values: column 0 of one stat tile (one value per row)

// Worker identity, from the common runtime args (see the host's worker ordering).
struct WorkerId {
    bool active;
    uint32_t idx;        // index in the ordered worker list
    uint32_t g, m;       // row-group, W member
    uint32_t row_start;  // first global tile-row of the group
    uint32_t crt;        // tile-rows of the group
    uint32_t w_start;    // first local tile-column of the member
    uint32_t cw;         // tile-columns of the member
};

// Worker from its index in the ordered worker list. last_crt: tile-rows of the last group (a ragged last shard).
inline WorkerId worker_from_idx(
    uint32_t idx,
    uint32_t k,
    uint32_t n_groups,
    uint32_t rows_hi,
    uint32_t big,
    uint32_t cw_nom,
    uint32_t slice_w_tiles,
    uint32_t last_crt) {
    WorkerId w{};
    w.idx = idx;
    w.active = w.idx < n_groups * k;
    if (!w.active) {
        return w;
    }
    w.g = w.idx / k;
    w.m = w.idx - w.g * k;
    if (w.g < big) {
        w.row_start = w.g * rows_hi;
        w.crt = rows_hi;
    } else {
        w.row_start = big * rows_hi + (w.g - big) * (rows_hi - 1);
        w.crt = rows_hi - 1;
    }
    if (w.g + 1 == n_groups) {
        w.crt = last_crt;
    }
    w.w_start = w.m * cw_nom;
    w.cw = (slice_w_tiles - w.w_start) < cw_nom ? (slice_w_tiles - w.w_start) : cw_nom;
    return w;
}

}  // namespace agrn
