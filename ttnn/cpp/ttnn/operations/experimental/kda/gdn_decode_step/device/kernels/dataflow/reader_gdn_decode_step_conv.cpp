// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for the fused-conv variant (packed layout): per head, 4 packed tap tiles + 3 packed history slots (one tile
// each, row 2c = channel chunk c), the new token packed from the projection row, z, the a/b/dt_bias/-exp(A) scalars,
// the state and the norm weight; plus the constant scaler/eps/mask tiles and the 0/1 selector tiles used to scatter the
// conv result.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"

using namespace gdn_step_df;

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Nk,
    uint32_t Nv,
    uint32_t z_tile0,
    uint32_t ab_page,
    uint32_t dtb_fp32,
    uint32_t nea_fp32,
    uint32_t l2_eps_bits,
    uint32_t norm_eps_bits>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    const auto qkv_acc = TensorAccessor(tensor::qkv);
    const auto dtb_acc = TensorAccessor(tensor::dtb);
    const auto nea_acc = TensorAccessor(tensor::nea);
    const auto state_acc = TensorAccessor(tensor::state);
    const auto w_acc = TensorAccessor(tensor::weight);
    const auto hist_acc = TensorAccessor(tensor::hist);
    const auto taps_acc = TensorAccessor(tensor::taps);
    DataflowBuffer hist(dfb::hist);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer cur(dfb::cur);
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer z_in(dfb::z_in);
    DataflowBuffer a_s(dfb::a_s);
    DataflowBuffer b_s(dfb::b_s);
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
    DataflowBuffer state_in(dfb::state_in);
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
    DataflowBuffer mask(dfb::mask);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Ct = 2 * Kt + Vt;
    constexpr uint32_t rf = Nv / Nk;

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    generate_bcast_col_scalar(eps_l2, l2_eps_bits);
    generate_bcast_col_scalar(eps_norm, norm_eps_bits);
    build_row0_mask(mask, noc);
    read_tiles(w_acc, w_in, noc, 0, Vt);
    // selector tiles: sel[c] has a single 1.0 at (row 0, col c)  ->  sel[c] @ P picks row c of P into row 0
    sel.reserve_back(Ct);
    zero_reserved(sel, noc, Ct);
    {
        auto lock = sel.scoped_write_lock(Ct);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t c = 0; c < Ct; ++c) {
            p16[c * 1024 + tile_elem_index(0, 2 * c)] = 0x3F80;  // chunk c lives in row 2c of the packed tiles
        }
    }
    sel.push_back(Ct);

    for (uint32_t i = 0; i < wi_count; ++i) {
        const uint32_t h = wi_start + i;
        const uint32_t hk = h / rf;
        if (i == 0) {
            read_tiles(taps_acc, taps, noc, h * 4, 4);
        }
        read_tiles(hist_acc, hist, noc, h * 4 + 1, 3);  // slots 1..3 (oldest kept .. newest)
        cur.reserve_back(1);
        zero_reserved(cur, noc, 1);
        pack_head_tile<Kt, Vt, Nk>(qkv_acc, cur, noc, hk, h, 0);
        noc.async_read_barrier();
        cur.push_back(1);
        read_tiles(qkv_acc, z_in, noc, z_tile0 + h * Vt, Vt);
        // a[h], b[h] live in the same bf16 tile of the projection row: columns h and Nv + h
        a_s.reserve_back(1);
        const uint32_t a_bits = load_tile_scalar<false>(qkv_acc, a_s, noc, ab_page, h);
        uint32_t b_bits;
        {
            auto lock = a_s.scoped_write_lock(1);
            auto p16 = lock.template get_ptr<volatile uint16_t>();
            b_bits = static_cast<uint32_t>(p16[tile_elem_index(0, Nv + h)]) << 16;
        }
        fill_scalar_tile(a_s, a_bits);
        a_s.push_back(1);
        b_s.reserve_back(1);
        fill_scalar_tile(b_s, b_bits);
        b_s.push_back(1);
        load_head_scalar<dtb_fp32 != 0>(dtb_acc, dtb_s, noc, h);
        load_head_scalar<nea_fp32 != 0>(nea_acc, nea_s, noc, h);
        read_tiles(state_acc, state_in, noc, h * KV, KV);
    }
}
