// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader (Path A): pushes per-chunk inputs to DFBs.
//
// Runtime args: head_idx, num_chunks
//
// Tensor bindings: L_unit, v_beta_sc, k_bd_sc, intra_attn, q_decay, k_decay_t, dl_exp,
//   L_inv [BH, NC, C, 32] — 4 diagonal block inverses per chunk,
//   initial_state (bound only when present, with HAS_INITIAL_STATE defined; zeros if absent)
//
// Compile-time args: Ct, Kt, Vt

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Ct = get_arg(args::Ct);
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Vt = get_arg(args::Vt);

    const uint32_t head_idx = get_arg(args::head_idx);
    const uint32_t NC = get_arg(args::num_chunks);

    constexpr uint32_t out_tiles = Ct * Vt;
    constexpr uint32_t in_kv_tiles = Ct * Kt;
    constexpr uint32_t attn_tiles = Ct * Ct;
    constexpr uint32_t kdt_tiles = Kt * Ct;
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr auto dfb_L_unit = dfb::L_unit;
    constexpr auto dfb_v_beta_sc = dfb::v_beta_sc;
    constexpr auto dfb_k_bd_sc = dfb::k_bd_sc;
    constexpr auto dfb_intra_att = dfb::intra_att;
    constexpr auto dfb_q_decay = dfb::q_decay;
    constexpr auto dfb_k_dt = dfb::k_dt;
    constexpr auto dfb_dl_exp = dfb::dl_exp;
    constexpr auto dfb_S0 = dfb::S0;  // initial state for chunk 0
    // L_inv DFBs (1 tile each, loaded per chunk)
    constexpr auto dfb_L_inv_0 = dfb::L_inv_0;
    constexpr auto dfb_L_inv_1 = dfb::L_inv_1;
    constexpr auto dfb_L_inv_2 = dfb::L_inv_2;
    constexpr auto dfb_L_inv_3 = dfb::L_inv_3;

    constexpr uint32_t f32_tile = get_tile_size(dfb_S0);

    // TensorAccessors for the interleaved fp32 DRAM inputs, one per tensor binding.
    const auto lu_gen = TensorAccessor(tensor::L_unit);
    const auto vbs_gen = TensorAccessor(tensor::v_beta_sc);
    const auto kbs_gen = TensorAccessor(tensor::k_bd_sc);
    const auto att_gen = TensorAccessor(tensor::intra_attn);
    const auto qdec_gen = TensorAccessor(tensor::q_decay);
    const auto kdt_gen = TensorAccessor(tensor::k_decay_t);
    const auto dle_gen = TensorAccessor(tensor::dl_exp);
    const auto linv_gen = TensorAccessor(tensor::L_inv);
#ifdef HAS_INITIAL_STATE
    const auto s0_gen = TensorAccessor(tensor::initial_state);
#endif

    Noc noc;
    DataflowBuffer dfb_L_unit_o(dfb_L_unit);
    DataflowBuffer dfb_v_beta_sc_o(dfb_v_beta_sc);
    DataflowBuffer dfb_k_bd_sc_o(dfb_k_bd_sc);
    DataflowBuffer dfb_intra_att_o(dfb_intra_att);
    DataflowBuffer dfb_q_decay_o(dfb_q_decay);
    DataflowBuffer dfb_k_dt_o(dfb_k_dt);
    DataflowBuffer dfb_dl_exp_o(dfb_dl_exp);
    DataflowBuffer dfb_S0_o(dfb_S0);
    DataflowBuffer dfb_L_inv_0_o(dfb_L_inv_0);
    DataflowBuffer dfb_L_inv_1_o(dfb_L_inv_1);
    DataflowBuffer dfb_L_inv_2_o(dfb_L_inv_2);
    DataflowBuffer dfb_L_inv_3_o(dfb_L_inv_3);

    // === Load initial state S0 into dfb_S0 ===
    dfb_S0_o.reserve_back(state_tiles);
#ifdef HAS_INITIAL_STATE
    uint32_t s0_base_tile = head_idx * state_tiles;
    for (uint32_t t = 0; t < state_tiles; t++) {
        noc.async_read(s0_gen, dfb_S0_o, f32_tile, {.page_id = s0_base_tile + t}, {.offset_bytes = t * f32_tile});
    }
    noc.async_read_barrier();
#else
    noc.async_write_zeros(dfb_S0_o, state_tiles * f32_tile);
    noc.write_zeros_l1_barrier();
#endif
    dfb_S0_o.push_back(state_tiles);

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
        dfb_L_unit_o.reserve_back(attn_tiles);
        for (uint32_t t = 0; t < attn_tiles; t++) {
            noc.async_read(lu_gen, dfb_L_unit_o, f32_tile, {.page_id = lu_off + t}, {.offset_bytes = t * f32_tile});
        }
        noc.async_read_barrier();
        dfb_L_unit_o.push_back(attn_tiles);

        // v_beta_sc [C,Dv]
        dfb_v_beta_sc_o.reserve_back(out_tiles);
        for (uint32_t t = 0; t < out_tiles; t++) {
            noc.async_read(
                vbs_gen, dfb_v_beta_sc_o, f32_tile, {.page_id = vbs_off + t}, {.offset_bytes = t * f32_tile});
        }
        noc.async_read_barrier();
        dfb_v_beta_sc_o.push_back(out_tiles);

        // k_bd_sc [C,Dk]
        dfb_k_bd_sc_o.reserve_back(in_kv_tiles);
        for (uint32_t t = 0; t < in_kv_tiles; t++) {
            noc.async_read(kbs_gen, dfb_k_bd_sc_o, f32_tile, {.page_id = kbs_off + t}, {.offset_bytes = t * f32_tile});
        }
        noc.async_read_barrier();
        dfb_k_bd_sc_o.push_back(in_kv_tiles);

        // intra_attn [C,C]
        dfb_intra_att_o.reserve_back(attn_tiles);
        for (uint32_t t = 0; t < attn_tiles; t++) {
            noc.async_read(
                att_gen, dfb_intra_att_o, f32_tile, {.page_id = att_off + t}, {.offset_bytes = t * f32_tile});
        }
        noc.async_read_barrier();
        dfb_intra_att_o.push_back(attn_tiles);

        // q_decay [C,Dk]
        dfb_q_decay_o.reserve_back(in_kv_tiles);
        for (uint32_t t = 0; t < in_kv_tiles; t++) {
            noc.async_read(
                qdec_gen, dfb_q_decay_o, f32_tile, {.page_id = qdec_off + t}, {.offset_bytes = t * f32_tile});
        }
        noc.async_read_barrier();
        dfb_q_decay_o.push_back(in_kv_tiles);

        // k_decay_t [Dk,C]
        dfb_k_dt_o.reserve_back(kdt_tiles);
        for (uint32_t t = 0; t < kdt_tiles; t++) {
            noc.async_read(kdt_gen, dfb_k_dt_o, f32_tile, {.page_id = kdt_off + t}, {.offset_bytes = t * f32_tile});
        }
        noc.async_read_barrier();
        dfb_k_dt_o.push_back(kdt_tiles);

        // dl_exp (fp32 scalar tile)
        dfb_dl_exp_o.reserve_back(1);
        {
            noc.async_read(dle_gen, dfb_dl_exp_o, f32_tile, {.page_id = dle_off}, {.offset_bytes = 0});
            noc.async_read_barrier();
        }
        dfb_dl_exp_o.push_back(1);

        // L_inv diagonal block inverses: 4 tiles → dfb_L_inv_0..3
        // L_inv shape: [BH, NC, C, 32] with Ct = C/32 tiles per chunk.
        // Tile i holds L^{-1}[i*32:(i+1)*32, 0:32] (the i-th diagonal block inverse).
        {
            dfb_L_inv_0_o.reserve_back(1);
            noc.async_read(linv_gen, dfb_L_inv_0_o, f32_tile, {.page_id = linv_off + 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_L_inv_0_o.push_back(1);
        }
        {
            dfb_L_inv_1_o.reserve_back(1);
            noc.async_read(linv_gen, dfb_L_inv_1_o, f32_tile, {.page_id = linv_off + 1}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_L_inv_1_o.push_back(1);
        }
        {
            dfb_L_inv_2_o.reserve_back(1);
            noc.async_read(linv_gen, dfb_L_inv_2_o, f32_tile, {.page_id = linv_off + 2}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_L_inv_2_o.push_back(1);
        }
        {
            dfb_L_inv_3_o.reserve_back(1);
            noc.async_read(linv_gen, dfb_L_inv_3_o, f32_tile, {.page_id = linv_off + 3}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_L_inv_3_o.push_back(1);
        }
    }
}
