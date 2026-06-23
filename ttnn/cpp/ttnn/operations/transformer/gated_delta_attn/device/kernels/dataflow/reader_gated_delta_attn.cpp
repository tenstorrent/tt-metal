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

    // TensorAccessors for the interleaved fp32 DRAM inputs. The per-tensor
    // TensorAccessorArgs compile-time blocks are appended (in this order) by the
    // program factory right after the {Ct, Kt, Vt} compile-time args, so the first
    // block starts at compile-time-arg offset 3 and each chains off the previous.
    constexpr auto lu_args = TensorAccessorArgs<3>();
    const auto lu_gen = TensorAccessor(lu_args, lu_addr, f32_tile);
    constexpr auto vbs_args = TensorAccessorArgs<lu_args.next_compile_time_args_offset()>();
    const auto vbs_gen = TensorAccessor(vbs_args, vbs_addr, f32_tile);
    constexpr auto kbs_args = TensorAccessorArgs<vbs_args.next_compile_time_args_offset()>();
    const auto kbs_gen = TensorAccessor(kbs_args, kbs_addr, f32_tile);
    constexpr auto att_args = TensorAccessorArgs<kbs_args.next_compile_time_args_offset()>();
    const auto att_gen = TensorAccessor(att_args, att_addr, f32_tile);
    constexpr auto qdec_args = TensorAccessorArgs<att_args.next_compile_time_args_offset()>();
    const auto qdec_gen = TensorAccessor(qdec_args, qdec_addr, f32_tile);
    constexpr auto kdt_args = TensorAccessorArgs<qdec_args.next_compile_time_args_offset()>();
    const auto kdt_gen = TensorAccessor(kdt_args, kdt_addr, f32_tile);
    constexpr auto dle_args = TensorAccessorArgs<kdt_args.next_compile_time_args_offset()>();
    const auto dle_gen = TensorAccessor(dle_args, dle_addr, f32_tile);
    constexpr auto linv_args = TensorAccessorArgs<dle_args.next_compile_time_args_offset()>();
    const auto linv_gen = TensorAccessor(linv_args, linv_addr, f32_tile);
    constexpr auto s0_args = TensorAccessorArgs<linv_args.next_compile_time_args_offset()>();
    const auto s0_gen = TensorAccessor(s0_args, s0_addr, f32_tile);

    // === Load initial state S into CB8 ===
    cb_reserve_back(cb_S, state_tiles);
    uint32_t s0_base_tile = head_idx * state_tiles;
    if (s0_addr != 0) {
        for (uint32_t t = 0; t < state_tiles; t++) {
            uint64_t na = s0_gen.get_noc_addr(s0_base_tile + t);
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
    const uint32_t h_off_vbs = head_idx * NC * out_tiles;
    const uint32_t h_off_kbs = head_idx * NC * in_kv_tiles;
    const uint32_t h_off_att = head_idx * NC * attn_tiles;
    const uint32_t h_off_qdec = head_idx * NC * in_kv_tiles;
    const uint32_t h_off_kdt = head_idx * NC * kdt_tiles;
    const uint32_t h_off_dle = head_idx * NC * 1;
    // L_inv: [BH, NC, C, 32] => Ct tiles per (head, chunk) where Ct = C/32
    const uint32_t h_off_linv = head_idx * NC * Ct;

    for (uint32_t c = 0; c < NC; c++) {
        uint32_t lu_off = h_off_lu + c * attn_tiles;
        uint32_t vbs_off = h_off_vbs + c * out_tiles;
        uint32_t kbs_off = h_off_kbs + c * in_kv_tiles;
        uint32_t att_off = h_off_att + c * attn_tiles;
        uint32_t qdec_off = h_off_qdec + c * in_kv_tiles;
        uint32_t kdt_off = h_off_kdt + c * kdt_tiles;
        uint32_t dle_off = h_off_dle + c;
        uint32_t linv_off = h_off_linv + c * Ct;

        // L_unit [C,C]
        cb_reserve_back(cb_L_unit, attn_tiles);
        for (uint32_t t = 0; t < attn_tiles; t++) {
            uint64_t na = lu_gen.get_noc_addr(lu_off + t);
            noc_async_read(na, get_write_ptr(cb_L_unit) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_L_unit, attn_tiles);

        // v_beta_sc [C,Dv]
        cb_reserve_back(cb_v_beta_sc, out_tiles);
        for (uint32_t t = 0; t < out_tiles; t++) {
            uint64_t na = vbs_gen.get_noc_addr(vbs_off + t);
            noc_async_read(na, get_write_ptr(cb_v_beta_sc) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_v_beta_sc, out_tiles);

        // k_bd_sc [C,Dk]
        cb_reserve_back(cb_k_bd_sc, in_kv_tiles);
        for (uint32_t t = 0; t < in_kv_tiles; t++) {
            uint64_t na = kbs_gen.get_noc_addr(kbs_off + t);
            noc_async_read(na, get_write_ptr(cb_k_bd_sc) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_bd_sc, in_kv_tiles);

        // intra_attn [C,C]
        cb_reserve_back(cb_intra_att, attn_tiles);
        for (uint32_t t = 0; t < attn_tiles; t++) {
            uint64_t na = att_gen.get_noc_addr(att_off + t);
            noc_async_read(na, get_write_ptr(cb_intra_att) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_intra_att, attn_tiles);

        // q_decay [C,Dk]
        cb_reserve_back(cb_q_decay, in_kv_tiles);
        for (uint32_t t = 0; t < in_kv_tiles; t++) {
            uint64_t na = qdec_gen.get_noc_addr(qdec_off + t);
            noc_async_read(na, get_write_ptr(cb_q_decay) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_q_decay, in_kv_tiles);

        // k_decay_t [Dk,C]
        cb_reserve_back(cb_k_dt, kdt_tiles);
        for (uint32_t t = 0; t < kdt_tiles; t++) {
            uint64_t na = kdt_gen.get_noc_addr(kdt_off + t);
            noc_async_read(na, get_write_ptr(cb_k_dt) + t * f32_tile, f32_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_dt, kdt_tiles);

        // dl_exp (fp32 scalar tile)
        cb_reserve_back(cb_dl_exp, 1);
        {
            uint64_t na = dle_gen.get_noc_addr(dle_off);
            noc_async_read(na, get_write_ptr(cb_dl_exp), f32_tile);
            noc_async_read_barrier();
        }
        cb_push_back(cb_dl_exp, 1);

        // L_inv diagonal block inverses: 4 tiles → CB14..17
        // L_inv shape: [BH, NC, C, 32] with Ct = C/32 tiles per chunk.
        // Tile i holds L^{-1}[i*32:(i+1)*32, 0:32] (the i-th diagonal block inverse).
        {
            uint64_t na0 = linv_gen.get_noc_addr(linv_off + 0);
            cb_reserve_back(cb_L_inv_0, 1);
            noc_async_read(na0, get_write_ptr(cb_L_inv_0), f32_tile);
            noc_async_read_barrier();
            cb_push_back(cb_L_inv_0, 1);
        }
        {
            uint64_t na1 = linv_gen.get_noc_addr(linv_off + 1);
            cb_reserve_back(cb_L_inv_1, 1);
            noc_async_read(na1, get_write_ptr(cb_L_inv_1), f32_tile);
            noc_async_read_barrier();
            cb_push_back(cb_L_inv_1, 1);
        }
        {
            uint64_t na2 = linv_gen.get_noc_addr(linv_off + 2);
            cb_reserve_back(cb_L_inv_2, 1);
            noc_async_read(na2, get_write_ptr(cb_L_inv_2), f32_tile);
            noc_async_read_barrier();
            cb_push_back(cb_L_inv_2, 1);
        }
        {
            uint64_t na3 = linv_gen.get_noc_addr(linv_off + 3);
            cb_reserve_back(cb_L_inv_3, 1);
            noc_async_read(na3, get_write_ptr(cb_L_inv_3), f32_tile);
            noc_async_read_barrier();
            cb_push_back(cb_L_inv_3, 1);
        }
    }
}
