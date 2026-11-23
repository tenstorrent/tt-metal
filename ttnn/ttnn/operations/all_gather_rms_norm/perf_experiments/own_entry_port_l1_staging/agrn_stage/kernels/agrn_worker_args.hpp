// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm — the worker kernels' shared compile-time / common-runtime argument layout.
// Must match _worker_ct() / _worker_common_rt() in all_gather_rms_norm_program_descriptor.py.
#pragma once

#include <cstdint>

#include "agrn_common.hpp"

namespace agrn {

// ---- compile-time args (identical scalar prefix for reader / compute / writer) ----
constexpr uint32_t R = get_compile_time_arg_val(0);              // tile-rows of the slice (leading dims folded)
constexpr uint32_t Wlt = get_compile_time_arg_val(1);            // tile-columns of the local slice
constexpr uint32_t K = get_compile_time_arg_val(2);              // W members per row-group
constexpr uint32_t n_groups = get_compile_time_arg_val(3);       // row-groups
constexpr uint32_t rows_hi = get_compile_time_arg_val(4);        // tile-rows of the first `big` groups
constexpr uint32_t big = get_compile_time_arg_val(5);            // groups with rows_hi tile-rows (rest rows_hi - 1)
constexpr uint32_t cw_nom = get_compile_time_arg_val(6);         // tile-columns per member (last: remainder)
constexpr uint32_t grid_x = get_compile_time_arg_val(7);         // logical grid width (linear worker index)
constexpr uint32_t G = get_compile_time_arg_val(8);              // group size (chips)
constexpr uint32_t nstats = get_compile_time_arg_val(9);         // stats per row (RMS 1: sumsq; LN 3: m, T, K)
constexpr uint32_t gather_tiles = get_compile_time_arg_val(10);  // ceil(G * K / 32)
constexpr bool is_ln = get_compile_time_arg_val(11) != 0;
constexpr bool has_gamma = get_compile_time_arg_val(12) != 0;
constexpr bool has_beta = get_compile_time_arg_val(13) != 0;
constexpr bool gamma_is_tile = get_compile_time_arg_val(14) != 0;
constexpr uint32_t x_page_bytes = get_compile_time_arg_val(15);
constexpr uint32_t out_page_bytes = get_compile_time_arg_val(16);
constexpr uint32_t gamma_elem_bytes = get_compile_time_arg_val(17);
constexpr uint32_t w_valid_last = get_compile_time_arg_val(18);  // valid columns of the last local tile (1..32)
constexpr uint32_t W_local = get_compile_time_arg_val(19);
constexpr uint32_t Wt = get_compile_time_arg_val(20);      // output tile-columns
constexpr bool is_ru = get_compile_time_arg_val(21) != 0;  // W_local % 32 != 0: staged gather + assembly
constexpr bool x_is_f32 = get_compile_time_arg_val(22) != 0;
constexpr uint32_t out_elem_bytes = get_compile_time_arg_val(23);
constexpr uint32_t n_ports = get_compile_time_arg_val(24);
constexpr uint32_t assemble_src_max = get_compile_time_arg_val(25);      // R_U: source tiles per output tile (max)
constexpr uint32_t gamma_page_bytes = get_compile_time_arg_val(26);      // cb_gamma page (2 faces of a gamma tile)
constexpr uint32_t gamma_src_page_bytes = get_compile_time_arg_val(27);  // gamma DRAM page (stick or tile)
// HEIGHT / WIDTH_SHARDED input: cb_x is backed on the resident shard (zero-copy), worker identity comes from the shard
// map (common args), and the W tail of the caller's shard is masked in compute (never zeroed in place)
constexpr bool x_sharded = get_compile_time_arg_val(28) != 0;
constexpr uint32_t last_crt = get_compile_time_arg_val(29);  // tile-rows of the last row-group
// cb_out_block is backed on the own slice of this core's output shard (HEIGHT, one tile-row per shard): normalize
// packs the pages into their final place, there is no own write
constexpr bool out_in_place = get_compile_time_arg_val(30) != 0;
// WIDTH_SHARDED row-owner pre-combine: member m owns tile-rows [m * rows_per_owner, ...); the k members' partials of
// a row are combined on its owner into ONE record per (row, stat) before the fabric round, the owner combines the G
// chips' records after it and returns the final per-row stats to every member
constexpr bool pre_combine = get_compile_time_arg_val(31) != 0;
constexpr uint32_t row_chunk = get_compile_time_arg_val(32);       // tile-rows per pass of the per-row CBs
constexpr uint32_t rows_per_owner = get_compile_time_arg_val(33);  // pre_combine only
// own-ready bands (OVERLAP): > 1 — the writer announces its own block on band g * own_bands / n_groups (a global
// semaphore per band on every port core) instead of the single program semaphore sem_own_ready
constexpr uint32_t own_bands = get_compile_time_arg_val(34);
// RMS with gamma (OVERLAP): x * gamma is computed into cb_xg while the stats round runs (x * r * g = (x * g) * r),
// leaving one multiply per tile on the critical path after the stats arrive
constexpr bool xg_pre = get_compile_time_arg_val(35) != 0;
// EXPERIMENT own_entry_port_l1_staging: the own slot's walk on the ports (agrn_port_common.hpp), so a worker can place
// each of its pages at its position in its lane's own-entry packet sequence (cb_stage on the lane's two ports)
constexpr bool stage_on = get_compile_time_arg_val(36) != 0;
constexpr uint32_t st_L = get_compile_time_arg_val(37);  // links (lanes)
constexpr uint32_t st_nb = get_compile_time_arg_val(38);
constexpr uint32_t st_cols = get_compile_time_arg_val(39);  // walk cols_eff
constexpr uint32_t st_row_stride = get_compile_time_arg_val(40);
constexpr uint32_t st_slot_stride = get_compile_time_arg_val(41);
constexpr uint32_t st_nctr = get_compile_time_arg_val(42) & 0xFF;
constexpr bool st_by_row =
    ((get_compile_time_arg_val(42) >> 8) & 0xFF) != 0;  // sharded output: lane units are tile-rows
constexpr uint32_t ST_MAX_LANES = 4;
constexpr uint32_t ST_MAX_NB = 16;
constexpr uint32_t ST_LANES_IDX = 43;                              // staged units per lane
constexpr uint32_t ST_UNIT_OFF_IDX = ST_LANES_IDX + ST_MAX_LANES;  // per bank: its first page in the lane walk
constexpr uint32_t ST_CTR_IDX = ST_UNIT_OFF_IDX + ST_MAX_NB;       // counter L1 addresses (port cores)
constexpr uint32_t st_copies = (get_compile_time_arg_val(42) >> 16) & 0xFF;    // 2: fwd + bwd ports, 1: fwd only
constexpr bool st_split = ((get_compile_time_arg_val(42) >> 24) & 0xFF) != 0;  // bwd copies by the worker reader
constexpr uint32_t ST_M2_IDX = ST_CTR_IDX + ST_MAX_NB;
constexpr uint32_t st_mode = get_compile_time_arg_val(ST_M2_IDX) & 0xFF;  // 1: port L1, 2: distributed chunks
constexpr uint32_t st_rp = get_compile_time_arg_val(ST_M2_IDX + 1);       // pages per chunk (= run_pages)
constexpr uint32_t st_nch = get_compile_time_arg_val(ST_M2_IDX + 2);      // chunks per lane (max over lanes)
constexpr uint32_t st_nact = get_compile_time_arg_val(ST_M2_IDX + 3);     // staging cores (active workers)
constexpr uint32_t st_tbl = get_compile_time_arg_val(ST_M2_IDX + 4);      // common-arg index of their NoC coords
constexpr uint32_t st_ctr = get_compile_time_arg_val(ST_M2_IDX + 5);      // port chunk counters (L1 address)
constexpr bool st_m2_single = ((get_compile_time_arg_val(ST_M2_IDX) >> 8) & 1) != 0;  // one readiness counter per port
constexpr bool st_m2_ack = ((get_compile_time_arg_val(ST_M2_IDX) >> 9) & 1) != 0;  // atomic barrier before DRAM writes
constexpr uint32_t st_head = get_compile_time_arg_val(ST_M2_IDX + 6);      // walk pages staged on the ports themselves
constexpr uint32_t st_head_off = get_compile_time_arg_val(ST_M2_IDX + 7);  // their cb_stage page offset
constexpr uint32_t WORKER_CT_ARGS = ST_M2_IDX + 8;  // TensorAccessorArgs follow (reader / writer)

constexpr uint32_t KT = pre_combine ? 1 : K;  // members per (slot, row) in the stats table
constexpr uint32_t P = G * KT;                // stats partitions per row of the cross-chip combine
constexpr uint32_t gt_pc = (K + 31) / 32;     // gather tiles of the on-chip pre-combine (P = K)
constexpr uint32_t n_owners = pre_combine ? (R + rows_per_owner - 1) / rows_per_owner : 0;
constexpr uint32_t nfinal = is_ln ? 2 : 1;  // final stats per row returned by the owners (inv | mean, rstd)
constexpr uint32_t TILE = 32;
constexpr uint32_t FACE = 16;

// ---- common runtime args ----
constexpr uint32_t CRT_X_ADDR = 0;
constexpr uint32_t CRT_DST_ADDR = 1;  // own-slice write target: output (aligned) / staging (R_U)
constexpr uint32_t CRT_OUT_ADDR = 2;  // output (R_U assembly)
constexpr uint32_t CRT_GAMMA_ADDR = 3;
constexpr uint32_t CRT_BETA_ADDR = 4;
constexpr uint32_t CRT_EPS_BITS = 5;
constexpr uint32_t CRT_INV_W_BITS = 6;
constexpr uint32_t CRT_P = 7;
constexpr uint32_t CRT_HUB_MASK =
    8;  // bit q set: slot q lands in the bwd hub (host check; the table is multicast to every core)
constexpr uint32_t CRT_FWD_HUB = 9;  // NoC x | y << 16
constexpr uint32_t CRT_BWD_HUB = 10;
constexpr uint32_t CRT_PORTS = 11;                          // n_ports linear logical indices
constexpr uint32_t CRT_PORT_NOC = CRT_PORTS + n_ports;      // n_ports NoC coords (x | y << 16)
constexpr uint32_t CRT_INV_N = CRT_PORT_NOC + n_ports;      // K x fp32 bits of 1 / (valid columns of member m)
constexpr uint32_t CRT_INV_WL = CRT_INV_N + K;              // fp32 bits of 1 / W_local (pre-combined slice mean)
constexpr uint32_t CRT_BANDS = CRT_INV_WL + 1;              // own_bands band-counter L1 addresses (port cores)
constexpr uint32_t CRT_MEMBER_NOC = CRT_BANDS + own_bands;  // pre_combine: K member NoC coords (x | y << 16)
constexpr uint32_t CRT_WORKER_MAP = CRT_MEMBER_NOC + (pre_combine ? K : 0);  // sharded: byte [lin] = worker index
                                                                             // (0xFF: none), 4 per word

inline WorkerId my_worker_id() {
    uint32_t ports[n_ports];
    for (uint32_t i = 0; i < n_ports; ++i) {
        ports[i] = get_common_arg_val<uint32_t>(CRT_PORTS + i);
    }
    const uint32_t lin =
        static_cast<uint32_t>(get_absolute_logical_y()) * grid_x + static_cast<uint32_t>(get_absolute_logical_x());
    uint32_t idx;
    if constexpr (x_sharded) {
        // the shard core order is the worker order (host select_sharded_split)
        idx = (get_common_arg_val<uint32_t>(CRT_WORKER_MAP + lin / 4) >> (8 * (lin % 4))) & 0xFFu;
    } else {
        // row-major logical order with the port cores removed
        uint32_t before = 0;
        for (uint32_t i = 0; i < n_ports; ++i) {
            before += ports[i] < lin ? 1u : 0u;
        }
        idx = lin - before;
    }
    return worker_from_idx(idx, K, n_groups, rows_hi, big, cw_nom, Wlt, last_crt);
}

// pre_combine: the tile-rows [first, first + count) member m owns (count 0: not an owner).
inline uint32_t owner_rows(uint32_t m) {
    const uint32_t first = m * rows_per_owner;
    return first >= R ? 0u : ((R - first) < rows_per_owner ? R - first : rows_per_owner);
}

inline uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Valid columns of the member's last tile (the slice's W tail sits in the last member's last tile).
inline uint32_t member_owns_tail(const WorkerId& w) { return (w_valid_last < TILE) && (w.w_start + w.cw == Wlt); }

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
// EXPERIMENT own_entry_port_l1_staging: write the normalized block's pages (cb_out_block, waited page by page, never
// popped here) into cb_stage of the ports of their lane (fwd l = port l, bwd l = port L + l), each at its index in the
// lane's own-entry walk (bank-major, then slot row, then column; sharded output: lane row-major), then one increment
// per (lane, counter) of the selected ports. Uses the calling RISC's NoC.
// Position of the block page (tr, local column jc) in its lane's own-entry walk: returns false if not staged.
struct StagePos {
    uint32_t l, i, idx;
};
inline bool stage_pos(uint32_t p, uint32_t tr, uint32_t jc, StagePos& o) {
    const uint32_t base = p * st_slot_stride;
    const uint32_t base_mod = base % st_nb;
    const uint32_t rs_mod = st_row_stride % st_nb;
    const uint32_t rr = is_ru ? 0u : tr;
    const uint32_t jj = is_ru ? tr * Wlt + jc : jc;
    if constexpr (st_by_row) {
        o.l = rr % st_L;
        o.i = rr / st_L;
        if (o.i >= kernel_compile_time_args[ST_LANES_IDX + o.l]) {
            return false;
        }
        o.idx = o.i * st_cols + jj;
        return true;
    }
    uint32_t T = 1;  // rows per residue period of (base + rr * row_stride) % nb
    {
        uint32_t m = rs_mod;
        while (m != 0) {
            m += rs_mod;
            if (m >= st_nb) {
                m -= st_nb;
            }
            ++T;
        }
    }
    auto cnt_row = [&](uint32_t rbm, uint32_t b) -> uint32_t {
        const uint32_t j0 = b >= rbm ? b - rbm : b + st_nb - rbm;
        return j0 < st_cols ? (st_cols - 1 - j0) / st_nb + 1 : 0u;
    };
    const uint32_t P = base + rr * st_row_stride + jj;
    const uint32_t b = P % st_nb;
    o.l = b % st_L;
    o.i = b / st_L;
    if (o.i >= kernel_compile_time_args[ST_LANES_IDX + o.l]) {
        return false;
    }
    const uint32_t q = rr / T, rem = rr - q * T;
    uint32_t sum_t = 0, part = 0, m = base_mod;
    for (uint32_t u = 0; u < T; ++u) {
        const uint32_t c = cnt_row(m, b);
        sum_t += c;
        part += u < rem ? c : 0u;
        m += rs_mod;
        if (m >= st_nb) {
            m -= st_nb;
        }
    }
    const uint32_t rbm = (base_mod + rem * rs_mod) % st_nb;
    const uint32_t j0 = b >= rbm ? b - rbm : b + st_nb - rbm;
    o.idx = kernel_compile_time_args[ST_UNIT_OFF_IDX + b] + q * sum_t + part + (jj - j0) / st_nb;
    return true;
}

// STAGE_MODE 2: home of chunk g (of st_L * st_nch): spread evenly over the staging cores (one each while they suffice),
// else round-robin with slot g / n_act. Must match the port reader.
inline void stage_chunk_home(uint32_t g, uint32_t& core, uint32_t& slot) {
    constexpr uint32_t n_ch = st_L * st_nch;
    if constexpr (n_ch <= st_nact) {
        core = g * st_nact / n_ch;
        slot = 0;
    } else {
        core = g % st_nact;
        slot = g / st_nact;
    }
}

// STAGE_MODE 2: each page to the staging core of its walk chunk (one copy, shared by the lane's fwd and bwd port),
// then one increment of that chunk's counter on both ports per page (after the barrier).
inline void stage_dist_block(const WorkerId& w, uint32_t p) {
    const uint32_t blk = get_read_ptr(cb_out_block);
    const uint32_t stage = get_write_ptr(cb_stage);
    for (uint32_t r = 0; r < w.crt; ++r) {
        for (uint32_t j = 0; j < w.cw; ++j) {
            StagePos o;
            if (!stage_pos(p, w.row_start + r, w.w_start + j, o)) {
                continue;
            }
            const uint32_t t = r * w.cw + j;
            if (o.idx < st_head) {
                // the walk's head: straight into both ports of the lane (sent from there, no port read)
                const uint32_t d = stage + (st_head_off + o.idx) * out_page_bytes;
                const uint32_t f = get_common_arg_val<uint32_t>(CRT_PORT_NOC + o.l);
                const uint32_t bw = get_common_arg_val<uint32_t>(CRT_PORT_NOC + st_L + o.l);
                cb_wait_front(cb_out_block, t + 1);
                noc_async_write(blk + t * out_page_bytes, get_noc_addr(f & 0xFFFF, f >> 16, d), out_page_bytes);
                noc_async_write(blk + t * out_page_bytes, get_noc_addr(bw & 0xFFFF, bw >> 16, d), out_page_bytes);
                continue;
            }
            const uint32_t c = o.idx / st_rp;
            const uint32_t g = o.l * st_nch + c;
            uint32_t core, slot;
            stage_chunk_home(g, core, slot);
            const uint32_t xy = get_common_arg_val<uint32_t>(st_tbl + core);
            const uint32_t dst_l1 = stage + (slot * st_rp + (o.idx - c * st_rp)) * out_page_bytes;
            cb_wait_front(cb_out_block, t + 1);
            noc_async_write(blk + t * out_page_bytes, get_noc_addr(xy & 0xFFFF, xy >> 16, dst_l1), out_page_bytes);
        }
    }
    {
        AGRN_ZONE("wr_stage_bar");
        noc_async_write_barrier();
    }
    if constexpr (st_m2_single) {
        AGRN_ZONE("wr_stage_inc");
        // one counter (word 0) per port: one increment per worker on every port
        for (uint32_t i = 0; i < 2 * st_L; ++i) {
            const uint32_t x = get_common_arg_val<uint32_t>(CRT_PORT_NOC + i);
            noc_semaphore_inc(get_noc_addr(x & 0xFFFF, x >> 16, st_ctr), 1);
        }
        if constexpr (st_m2_ack) {
            noc_async_atomic_barrier();  // the increments land before the DRAM own-write burst starts on this NoC
        }
        return;
    }
    for (uint32_t r = 0; r < w.crt; ++r) {
        for (uint32_t j = 0; j < w.cw; ++j) {
            StagePos o;
            if (!stage_pos(p, w.row_start + r, w.w_start + j, o)) {
                continue;
            }
            const uint32_t a = st_ctr + (o.idx / st_rp) * 4;
            const uint32_t f = get_common_arg_val<uint32_t>(CRT_PORT_NOC + o.l);
            const uint32_t bw = get_common_arg_val<uint32_t>(CRT_PORT_NOC + st_L + o.l);
            noc_semaphore_inc(get_noc_addr(f & 0xFFFF, f >> 16, a), 1);
            noc_semaphore_inc(get_noc_addr(bw & 0xFFFF, bw >> 16, a), 1);
        }
    }
    noc_async_atomic_barrier();
}

inline void stage_own_block(const WorkerId& w, uint32_t p, bool to_fwd, bool to_bwd) {
    if (!to_fwd && !to_bwd) {
        return;
    }
    const uint32_t blk = get_read_ptr(cb_out_block);  // nothing popped yet on this RISC: the block's first page
    const uint32_t stage = get_write_ptr(cb_stage);   // same address on every core
    const uint32_t base = p * st_slot_stride;
    const uint32_t base_mod = base % st_nb;
    const uint32_t rs_mod = st_row_stride % st_nb;
    uint32_t T = 1;  // rows per residue period of (base + rr * row_stride) % nb
    {
        uint32_t m = rs_mod;
        while (m != 0) {
            m += rs_mod;
            if (m >= st_nb) {
                m -= st_nb;
            }
            ++T;
        }
    }
    auto cnt_row = [&](uint32_t rbm, uint32_t b) -> uint32_t {
        const uint32_t j0 = b >= rbm ? b - rbm : b + st_nb - rbm;
        return j0 < st_cols ? (st_cols - 1 - j0) / st_nb + 1 : 0u;
    };
    uint32_t cnt[ST_MAX_LANES][16];
    for (uint32_t l = 0; l < st_L; ++l) {
        for (uint32_t c = 0; c < st_nctr; ++c) {
            cnt[l][c] = 0;
        }
    }
    for (uint32_t r = 0; r < w.crt; ++r) {
        const uint32_t tr = w.row_start + r;
        for (uint32_t j = 0; j < w.cw; ++j) {
            const uint32_t t = r * w.cw + j;
            const uint32_t rr = is_ru ? 0u : tr;
            const uint32_t jj = is_ru ? tr * Wlt + w.w_start + j : w.w_start + j;
            uint32_t l, i, idx;
            if constexpr (st_by_row) {
                l = rr % st_L;
                i = rr / st_L;
                if (i >= kernel_compile_time_args[ST_LANES_IDX + l]) {
                    continue;
                }
                idx = i * st_cols + jj;
            } else {
                const uint32_t P = base + rr * st_row_stride + jj;
                const uint32_t b = P % st_nb;
                l = b % st_L;
                i = b / st_L;
                if (i >= kernel_compile_time_args[ST_LANES_IDX + l]) {
                    continue;
                }
                // pages of bank b in the slot rows before rr (periodic in rr with period T)
                const uint32_t q = rr / T, rem = rr - q * T;
                uint32_t sum_t = 0, part = 0, m = base_mod;
                for (uint32_t u = 0; u < T; ++u) {
                    const uint32_t c = cnt_row(m, b);
                    sum_t += c;
                    part += u < rem ? c : 0u;
                    m += rs_mod;
                    if (m >= st_nb) {
                        m -= st_nb;
                    }
                }
                const uint32_t rbm = (base_mod + rem * rs_mod) % st_nb;
                const uint32_t j0 = b >= rbm ? b - rbm : b + st_nb - rbm;
                idx = kernel_compile_time_args[ST_UNIT_OFF_IDX + b] + q * sum_t + part + (jj - j0) / st_nb;
            }
            cb_wait_front(cb_out_block, t + 1);
            const uint32_t src = blk + t * out_page_bytes;
            const uint32_t dst_l1 = stage + idx * out_page_bytes;
            if (to_fwd) {
                const uint32_t c = get_common_arg_val<uint32_t>(CRT_PORT_NOC + l);
                noc_async_write(src, get_noc_addr(c & 0xFFFF, c >> 16, dst_l1), out_page_bytes);
            }
            if (to_bwd) {
                const uint32_t c = get_common_arg_val<uint32_t>(CRT_PORT_NOC + st_L + l);
                noc_async_write(src, get_noc_addr(c & 0xFFFF, c >> 16, dst_l1), out_page_bytes);
            }
            ++cnt[l][i < st_nctr ? i : st_nctr - 1];
        }
    }
    noc_async_write_barrier();
    for (uint32_t l = 0; l < st_L; ++l) {
        for (uint32_t c = 0; c < st_nctr; ++c) {
            if (cnt[l][c] > 0) {
                const uint32_t a = kernel_compile_time_args[ST_CTR_IDX + c];
                if (to_fwd) {
                    const uint32_t x = get_common_arg_val<uint32_t>(CRT_PORT_NOC + l);
                    noc_semaphore_inc(get_noc_addr(x & 0xFFFF, x >> 16, a), cnt[l][c]);
                }
                if (to_bwd) {
                    const uint32_t x = get_common_arg_val<uint32_t>(CRT_PORT_NOC + st_L + l);
                    noc_semaphore_inc(get_noc_addr(x & 0xFFFF, x >> 16, a), cnt[l][c]);
                }
            }
        }
    }
    noc_async_atomic_barrier();
}
#endif

}  // namespace agrn
