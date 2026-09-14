// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// M0a scratch reader: one core = (user u, value head h). Everything is read ONCE per core: constants, the norm weight,
// the dt_bias / -exp(A) scalars, the head's whole 32-row q/k/v/z tiles and the a|b tile of the projection rows (rows
// u*T + t are this user's T tokens), the one-hot selector tiles, and the 64 KiB fp32 state from ring block
// (s0_slot*BH + bh). No per-token reads: the T-loop in compute runs on L1-resident operands.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"

using namespace gdn_step_df;

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Nk,
    uint32_t Nv,
    uint32_t T,
    uint32_t B,
    uint32_t z_tile0,
    uint32_t ab_page,
    uint32_t s0_slot,
    uint32_t dtb_fp32,
    uint32_t nea_fp32,
    uint32_t l2_eps_bits,
    uint32_t norm_eps_bits,
    uint32_t row_batched,
    uint32_t w_tiles,
    uint32_t opt_flags>
TT_KERNEL void reader(uint32_t u, uint32_t h) {
    const auto qkv_acc = TensorAccessor(tensor::qkv);
    const auto dtb_acc = TensorAccessor(tensor::dtb);
    const auto nea_acc = TensorAccessor(tensor::nea);
    const auto ring_acc = TensorAccessor(tensor::ring);
    const auto w_acc = TensorAccessor(tensor::weight);
    DataflowBuffer q_in(dfb::q_in);
    DataflowBuffer k_in(dfb::k_in);
    DataflowBuffer v_in(dfb::v_in);
    DataflowBuffer z_in(dfb::z_in);
    DataflowBuffer ab_in(dfb::ab_in);
    DataflowBuffer mask_T(dfb::mask_T);
    DataflowBuffer e_t(dfb::e_t);
    DataflowBuffer rsel(dfb::rsel);
    DataflowBuffer csel(dfb::csel);
    DataflowBuffer state_in(dfb::state_in);
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t rf = Nv / Nk;
    constexpr uint32_t BH = B * Nv;
    const uint32_t hk = h / rf;
    const uint32_t bh = u * Nv + h;
    const uint32_t tr = (u * T) / 32;  // the user's rows live in qkv tile row tr ...
    const uint32_t r0 = (u * T) % 32;  // ... at rows r0 .. r0+T-1 of that tile row
    const uint32_t pg = tr * w_tiles;  // page of tile column 0 of that tile row
    constexpr bool two_phase = (opt_flags & 1u) != 0;

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    generate_bcast_col_scalar(eps_l2, l2_eps_bits);
    generate_bcast_col_scalar(eps_norm, norm_eps_bits);
    // per-head scalars (each is a small read with its own barrier; done before the bulk reads are in flight)
    load_head_scalar<dtb_fp32 != 0>(dtb_acc, dtb_s, noc, h);
    load_head_scalar<nea_fp32 != 0>(nea_acc, nea_s, noc, h);

    // every bulk DRAM read of this core, issued up front; one barrier at the end (two_phase: the small reads get their
    // own barrier + push before the 64 KiB state read is issued, so compute's PRE overlaps the state load)
    if constexpr (!two_phase) {
        state_in.reserve_back(KV);
        read_tiles_at(ring_acc, state_in, noc, (s0_slot * BH + bh) * KV, KV, 0);
    }
    q_in.reserve_back(Kt);
    read_tiles_at(qkv_acc, q_in, noc, pg + hk * Kt, Kt, 0);
    k_in.reserve_back(Kt);
    read_tiles_at(qkv_acc, k_in, noc, pg + Nk * Kt + hk * Kt, Kt, 0);
    v_in.reserve_back(Vt);
    read_tiles_at(qkv_acc, v_in, noc, pg + 2 * Nk * Kt + h * Vt, Vt, 0);
    z_in.reserve_back(Vt);
    read_tiles_at(qkv_acc, z_in, noc, pg + z_tile0 + h * Vt, Vt, 0);
    ab_in.reserve_back(1);
    read_tiles_at(qkv_acc, ab_in, noc, pg + ab_page, 1, 0);
    w_in.reserve_back(Vt);
    read_tiles_at(w_acc, w_in, noc, 0, Vt, 0);

    // selector tiles (bf16 one-hots), built while the reads are in flight
    // csel[0]: row h all ones (column extractor of a), csel[1]: row Nv + h all ones (of b)
    csel.reserve_back(2);
    zero_reserved(csel, noc, 2);
    {
        auto lock = csel.scoped_write_lock(2);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t j = 0; j < 32; ++j) {
            p16[tile_elem_index(h, j)] = 0x3F80;
            p16[1024 + tile_elem_index(Nv + h, j)] = 0x3F80;
        }
    }
    csel.push_back(2);
    // rsel[t]: 1.0 at column r0 + t in every row (row extractor: rsel[t] @ X = every row equal to row r0 + t of X)
    rsel.reserve_back(T);
    zero_reserved(rsel, noc, T);
    {
        auto lock = rsel.scoped_write_lock(T);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t i = 0; i < 32; ++i) {
                p16[t * 1024 + tile_elem_index(i, r0 + t)] = 0x3F80;
            }
        }
    }
    rsel.push_back(T);
    if constexpr (row_batched != 0) {
        // mask_T: 1.0 at (r0 + t, 0) for t < T  (the user's rows, for the row-batched l2norm masks)
        mask_T.reserve_back(1);
        zero_reserved(mask_T, noc, 1);
        {
            auto lock = mask_T.scoped_write_lock(1);
            auto p16 = lock.template get_ptr<volatile uint16_t>();
            for (uint32_t t = 0; t < T; ++t) {
                p16[tile_elem_index(r0 + t, 0)] = 0x3F80;
            }
        }
        mask_T.push_back(1);
    }
    // e_t[t]: 1.0 at (r0 + t, 0)  (per-token row mask)
    e_t.reserve_back(T);
    zero_reserved(e_t, noc, T);
    {
        auto lock = e_t.scoped_write_lock(T);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t t = 0; t < T; ++t) {
            p16[t * 1024 + tile_elem_index(r0 + t, 0)] = 0x3F80;
        }
    }
    e_t.push_back(T);

    noc.async_read_barrier();
    q_in.push_back(Kt);
    k_in.push_back(Kt);
    v_in.push_back(Vt);
    z_in.push_back(Vt);
    ab_in.push_back(1);
    w_in.push_back(Vt);
    if constexpr (two_phase) {
        state_in.reserve_back(KV);
        read_tiles_at(ring_acc, state_in, noc, (s0_slot * BH + bh) * KV, KV, 0);
        noc.async_read_barrier();
    }
    state_in.push_back(KV);
}
