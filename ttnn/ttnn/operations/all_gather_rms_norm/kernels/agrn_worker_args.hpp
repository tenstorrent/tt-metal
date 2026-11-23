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
constexpr uint32_t WORKER_CT_ARGS = 31;  // TensorAccessorArgs follow (reader / writer)

constexpr uint32_t P = G * K;  // stats partitions per row
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
constexpr uint32_t CRT_HUB_MASK = 8;  // bit q set: slot q is read from the bwd hub, else the fwd hub
constexpr uint32_t CRT_FWD_HUB = 9;   // NoC x | y << 16
constexpr uint32_t CRT_BWD_HUB = 10;
constexpr uint32_t CRT_PORTS = 11;                      // n_ports linear logical indices
constexpr uint32_t CRT_PORT_NOC = CRT_PORTS + n_ports;  // n_ports NoC coords (x | y << 16)
constexpr uint32_t CRT_INV_N = CRT_PORT_NOC + n_ports;  // K x fp32 bits of 1 / (valid columns of member m)
constexpr uint32_t CRT_WORKER_MAP = CRT_INV_N + K;      // sharded: byte [lin] = worker index (0xFF: none), 4 per word

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

// Valid columns of the member's last tile (the slice's W tail sits in the last member's last tile).
inline uint32_t member_owns_tail(const WorkerId& w) { return (w_valid_last < TILE) && (w.w_start + w.cw == Wlt); }

}  // namespace agrn
