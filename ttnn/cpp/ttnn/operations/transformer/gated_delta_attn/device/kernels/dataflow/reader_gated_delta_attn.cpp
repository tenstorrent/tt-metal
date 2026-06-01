// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader (Path A): pushes per-chunk inputs to CBs.
//
// Runtime args:
//   0  head_idx
//   1  num_chunks
//   2  L_unit_addr
//   3  v_beta_sc_addr
//   4  k_bd_sc_addr
//   5  intra_attn_addr
//   6  q_decay_addr
//   7  k_decay_t_addr
//   8  dl_exp_addr
//   9  L_inv_addr       [BH, NC, C, 32] — 4 diagonal block inverses per chunk
//  10  initial_state_addr (0 = zeros)
//
// Compile-time args: Ct, Kt, Vt

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);

    const uint32_t head_idx = get_arg_val<uint32_t>(0);
    const uint32_t NC = get_arg_val<uint32_t>(1);
    const uint32_t lu_addr = get_arg_val<uint32_t>(2);
    const uint32_t vbs_addr = get_arg_val<uint32_t>(3);
    const uint32_t kbs_addr = get_arg_val<uint32_t>(4);
    const uint32_t att_addr = get_arg_val<uint32_t>(5);
    const uint32_t qdec_addr = get_arg_val<uint32_t>(6);
    const uint32_t kdt_addr = get_arg_val<uint32_t>(7);
    const uint32_t dle_addr = get_arg_val<uint32_t>(8);
    const uint32_t linv_addr = get_arg_val<uint32_t>(9);
    const uint32_t s0_addr = get_arg_val<uint32_t>(10);
    // Multi-core-per-head value split: this core owns value-tile slice [v_off, v_off+Vt)
    // of the global value dim (Vt_global tiles). Vt (ct arg) = LOCAL value-tile count.
    // split_v=1 => v_off=0, Vt_global=Vt (identical to single-core behavior).
    const uint32_t v_off = get_arg_val<uint32_t>(11);
    const uint32_t Vt_global = get_arg_val<uint32_t>(12);

    constexpr uint32_t out_tiles = Ct * Vt;
    constexpr uint32_t in_kv_tiles = Ct * Kt;
    constexpr uint32_t attn_tiles = Ct * Ct;
    constexpr uint32_t kdt_tiles = Kt * Ct;
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr uint32_t cb_L_unit = tt::CBIndex::c_0;
    constexpr uint32_t cb_v_beta_sc = tt::CBIndex::c_1;
    constexpr uint32_t cb_k_bd_sc = tt::CBIndex::c_2;
    constexpr uint32_t cb_intra_att = tt::CBIndex::c_3;
    constexpr uint32_t cb_q_decay = tt::CBIndex::c_4;
    constexpr uint32_t cb_k_dt = tt::CBIndex::c_5;
    constexpr uint32_t cb_dl_exp = tt::CBIndex::c_6;
    constexpr uint32_t cb_S = tt::CBIndex::c_8;
    // L_inv CBs: c_14..c_17 (1 tile each, loaded per chunk)
    constexpr uint32_t cb_L_inv_0 = tt::CBIndex::c_14;
    constexpr uint32_t cb_L_inv_1 = tt::CBIndex::c_15;
    constexpr uint32_t cb_L_inv_2 = tt::CBIndex::c_16;
    constexpr uint32_t cb_L_inv_3 = tt::CBIndex::c_17;

    constexpr uint32_t f32_tile = get_tile_size(cb_S);

    const InterleavedAddrGenFast<true> lu_gen = {.bank_base_address = lu_addr, .page_size = f32_tile};
    const InterleavedAddrGenFast<true> vbs_gen = {.bank_base_address = vbs_addr, .page_size = f32_tile};
    const InterleavedAddrGenFast<true> kbs_gen = {.bank_base_address = kbs_addr, .page_size = f32_tile};
    const InterleavedAddrGenFast<true> att_gen = {.bank_base_address = att_addr, .page_size = f32_tile};
    const InterleavedAddrGenFast<true> qdec_gen = {.bank_base_address = qdec_addr, .page_size = f32_tile};
    const InterleavedAddrGenFast<true> kdt_gen = {.bank_base_address = kdt_addr, .page_size = f32_tile};
    const InterleavedAddrGenFast<true> dle_gen = {.bank_base_address = dle_addr, .page_size = f32_tile};
    const InterleavedAddrGenFast<true> linv_gen = {.bank_base_address = linv_addr, .page_size = f32_tile};
    const InterleavedAddrGenFast<true> s0_gen = {.bank_base_address = s0_addr, .page_size = f32_tile};

    // === Load initial state S [Kt, Vt_local] into CB8 ===
    // Global state is [BH, Dk, Dv] = head*(Kt*Vt_global) + kt*Vt_global + (v_off + vl).
    cb_reserve_back(cb_S, state_tiles);
    if (s0_addr != 0) {
        for (uint32_t t = 0; t < state_tiles; t++) {
            uint32_t kt = t / Vt;  // Vt = local value-tile count
            uint32_t vl = t % Vt;
            uint32_t gtile = head_idx * Kt * Vt_global + kt * Vt_global + v_off + vl;
            uint64_t na = get_noc_addr(gtile, s0_gen);
            noc_async_read(na, get_write_ptr(cb_S) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
    } else {
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_S));
        for (uint32_t w = 0; w < state_tiles * f32_tile / 4; w++) {
            ptr[w] = 0;
        }
    }
    cb_push_back(cb_S, state_tiles);

    // Per-head tile offsets in the flat 4D tensors [BH, NC, *, *].
    const uint32_t h_off_lu = head_idx * NC * attn_tiles;
    const uint32_t h_off_kbs = head_idx * NC * in_kv_tiles;
    const uint32_t h_off_att = head_idx * NC * attn_tiles;
    const uint32_t h_off_qdec = head_idx * NC * in_kv_tiles;
    const uint32_t h_off_kdt = head_idx * NC * kdt_tiles;
    const uint32_t h_off_dle = head_idx * NC * 1;
    // L_inv: [BH, NC, C, 32] => Ct tiles per (head, chunk) where Ct = C/32
    const uint32_t h_off_linv = head_idx * NC * Ct;

    for (uint32_t c = 0; c < NC; c++) {
        uint32_t lu_off = h_off_lu + c * attn_tiles;
        uint32_t kbs_off = h_off_kbs + c * in_kv_tiles;
        uint32_t att_off = h_off_att + c * attn_tiles;
        uint32_t qdec_off = h_off_qdec + c * in_kv_tiles;
        uint32_t kdt_off = h_off_kdt + c * kdt_tiles;
        uint32_t dle_off = h_off_dle + c;
        uint32_t linv_off = h_off_linv + c * Ct;

        // L_unit [C,C]
        cb_reserve_back(cb_L_unit, attn_tiles);
        for (uint32_t t = 0; t < attn_tiles; t++) {
            uint64_t na = get_noc_addr(lu_off + t, lu_gen);
            noc_async_read(na, get_write_ptr(cb_L_unit) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_L_unit, attn_tiles);

        // v_beta_sc [C, Vt_local slice] — global [BH,NC,C,Dv]; load this core's value slice.
        cb_reserve_back(cb_v_beta_sc, out_tiles);
        uint32_t vbs_chunk_base = head_idx * NC * Ct * Vt_global + c * Ct * Vt_global;
        for (uint32_t t = 0; t < out_tiles; t++) {
            uint32_t ct = t / Vt;  // Vt = local value-tile count
            uint32_t vl = t % Vt;
            uint32_t gtile = vbs_chunk_base + ct * Vt_global + v_off + vl;
            uint64_t na = get_noc_addr(gtile, vbs_gen);
            noc_async_read(na, get_write_ptr(cb_v_beta_sc) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_v_beta_sc, out_tiles);

        // k_bd_sc [C,Dk]
        cb_reserve_back(cb_k_bd_sc, in_kv_tiles);
        for (uint32_t t = 0; t < in_kv_tiles; t++) {
            uint64_t na = get_noc_addr(kbs_off + t, kbs_gen);
            noc_async_read(na, get_write_ptr(cb_k_bd_sc) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_bd_sc, in_kv_tiles);

        // intra_attn [C,C]
        cb_reserve_back(cb_intra_att, attn_tiles);
        for (uint32_t t = 0; t < attn_tiles; t++) {
            uint64_t na = get_noc_addr(att_off + t, att_gen);
            noc_async_read(na, get_write_ptr(cb_intra_att) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_intra_att, attn_tiles);

        // q_decay [C,Dk]
        cb_reserve_back(cb_q_decay, in_kv_tiles);
        for (uint32_t t = 0; t < in_kv_tiles; t++) {
            uint64_t na = get_noc_addr(qdec_off + t, qdec_gen);
            noc_async_read(na, get_write_ptr(cb_q_decay) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_q_decay, in_kv_tiles);

        // k_decay_t [Dk,C]
        cb_reserve_back(cb_k_dt, kdt_tiles);
        for (uint32_t t = 0; t < kdt_tiles; t++) {
            uint64_t na = get_noc_addr(kdt_off + t, kdt_gen);
            noc_async_read(na, get_write_ptr(cb_k_dt) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_dt, kdt_tiles);

        // dl_exp (fp32 scalar tile)
        cb_reserve_back(cb_dl_exp, 1);
        {
            uint64_t na = get_noc_addr(dle_off, dle_gen);
            noc_async_read(na, get_write_ptr(cb_dl_exp), f32_tile);
            noc_async_read_barrier();
        }
        cb_push_back(cb_dl_exp, 1);

        // L_inv diagonal block inverses: 4 tiles → CB14..17
        // L_inv shape: [BH, NC, C, 32] with Ct = C/32 tiles per chunk.
        // Tile i holds L^{-1}[i*32:(i+1)*32, 0:32] (the i-th diagonal block inverse).
        {
            uint64_t na0 = get_noc_addr(linv_off + 0, linv_gen);
            cb_reserve_back(cb_L_inv_0, 1);
            noc_async_read(na0, get_write_ptr(cb_L_inv_0), f32_tile);
            noc_async_read_barrier();
            cb_push_back(cb_L_inv_0, 1);
        }
        {
            uint64_t na1 = get_noc_addr(linv_off + 1, linv_gen);
            cb_reserve_back(cb_L_inv_1, 1);
            noc_async_read(na1, get_write_ptr(cb_L_inv_1), f32_tile);
            noc_async_read_barrier();
            cb_push_back(cb_L_inv_1, 1);
        }
        {
            uint64_t na2 = get_noc_addr(linv_off + 2, linv_gen);
            cb_reserve_back(cb_L_inv_2, 1);
            noc_async_read(na2, get_write_ptr(cb_L_inv_2), f32_tile);
            noc_async_read_barrier();
            cb_push_back(cb_L_inv_2, 1);
        }
        {
            uint64_t na3 = get_noc_addr(linv_off + 3, linv_gen);
            cb_reserve_back(cb_L_inv_3, 1);
            noc_async_read(na3, get_write_ptr(cb_L_inv_3), f32_tile);
            noc_async_read_barrier();
            cb_push_back(cb_L_inv_3, 1);
        }
    }
}
