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

namespace agrn {

// ---- circular buffers ----
constexpr uint32_t cb_x = 0;                // resident x block (crt x cw tiles, row pitch cw)
constexpr uint32_t cb_gamma = 1;            // gamma slice: compact pages, row 0 of faces 0/1 (ROW-broadcast)
constexpr uint32_t cb_beta = 2;             // beta slice (LN), same page format as gamma
constexpr uint32_t cb_scaler = 3;           // 1.0 reduce scaler (bf16)
constexpr uint32_t cb_mask = 4;             // LN: 0/1 row mask of the W-tail tile (AccumulateViaAdd only_partial)
constexpr uint32_t cb_sumsq = 5;            // RMS: per tile-row DEST-accumulated sum of x^2
constexpr uint32_t cb_ln_m = 6;             // LN: local row mean (col 0), source of the shift broadcast
constexpr uint32_t cb_ln_shift = 7;         // LN: K = tf32(m) broadcast to every column
constexpr uint32_t cb_ln_sq = 8;            // LN: (x - K)^2 tiles of the block
constexpr uint32_t cb_partial = 9;          // per-row stats (col 0), Float32 wire contract: [stat][row]
constexpr uint32_t cb_record_staging = 10;  // writer scratch: 128 B records
constexpr uint32_t cb_record_landing = 11;  // reader scratch: one tile-row's G*k*nstats records
constexpr uint32_t cb_gathered = 12;        // Float32 gather tiles: per row [stat][gather tile], col j = partition j
constexpr uint32_t cb_stat_consts = 13;     // LN: n_j per partition column (Float32)
constexpr uint32_t cb_inv = 14;             // RMS inv_rms / LN rstd (col 0)
constexpr uint32_t cb_mean_col = 15;        // LN: exact row mean (col 0), broadcast by the reader
constexpr uint32_t cb_mean_b = 16;          // LN: exact row mean in every column (dataflow broadcast)
constexpr uint32_t cb_rstd_b = 17;          // LN: rstd broadcast to every column
constexpr uint32_t cb_ln_tmp = 18;          // LN combine scratch tiles (Float32)
constexpr uint32_t cb_out_block = 19;       // normalized own block, resident until program end (ports pull it)
constexpr uint32_t cb_assemble = 20;        // R_U writer scratch: source tiles + one output tile
constexpr uint32_t cb_port_chunks = 21;     // port: 2 x batch_chunks packets
constexpr uint32_t cb_x32 = 22;             // LN, non-Float32 x: exact Float32 copy of the x block
constexpr uint32_t cb_gamma_rows = 23;      // reader scratch: raw gamma / beta source rows (128 B stride)

// ---- program-local semaphores ----
constexpr uint32_t sem_records_local = 0;  // hubs: one inc per active worker (records written into my table)
constexpr uint32_t sem_stats_ready = 1;    // workers: one inc per hub (all slots of the stats table landed)
constexpr uint32_t sem_own_ready = 2;      // ports: one inc per active worker (own block written + resident)
constexpr uint32_t sem_landed = 3;         // workers (R_U): one inc per port (its arrivals landed)

constexpr uint32_t RECORD_BYTES = 128;  // 32 fp32 values: column 0 of one stat tile (one value per row)

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

// lin: my linear logical index (y * grid_x + x). ports: linear logical indices of the port cores (any order).
inline WorkerId derive_worker(
    uint32_t lin,
    const uint32_t* ports,
    uint32_t n_ports,
    uint32_t k,
    uint32_t n_groups,
    uint32_t rows_hi,
    uint32_t big,
    uint32_t cw_nom,
    uint32_t slice_w_tiles) {
    WorkerId w{};
    uint32_t before = 0;
    for (uint32_t i = 0; i < n_ports; ++i) {
        before += ports[i] < lin ? 1u : 0u;
    }
    w.idx = lin - before;
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
    w.w_start = w.m * cw_nom;
    w.cw = (slice_w_tiles - w.w_start) < cw_nom ? (slice_w_tiles - w.w_start) : cw_nom;
    return w;
}

}  // namespace agrn
