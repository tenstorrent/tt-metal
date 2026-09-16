// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for the batched fused-conv variant: one core = value head `head`, users [u0, u0 + nu). Per core once: taps,
// norm weight, z tiles (rows = users), dt_bias / -exp(A) scalars, constants. Per user: 3 packed history slots, the new
// token packed from row b of the projection tile, a/b scalars from row b, per-user selector + mask tiles, the state.
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
TT_KERNEL void reader(uint32_t head, uint32_t u0, uint32_t nu) {
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
    const uint32_t h = head;
    const uint32_t hk = h / rf;

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    generate_bcast_col_scalar(eps_l2, l2_eps_bits);
    generate_bcast_col_scalar(eps_norm, norm_eps_bits);
    read_tiles(w_acc, w_in, noc, 0, Vt);
    read_tiles(taps_acc, taps, noc, h * 4, 4);
    read_tiles(qkv_acc, z_in, noc, z_tile0 + h * Vt, Vt);  // all users' rows of this head's z
    load_head_scalar<dtb_fp32 != 0>(dtb_acc, dtb_s, noc, h);
    load_head_scalar<nea_fp32 != 0>(nea_acc, nea_s, noc, h);

    for (uint32_t ui = 0; ui < nu; ++ui) {
        const uint32_t b = u0 + ui;
        const uint32_t bh = b * Nv + h;
        build_user_selectors(sel, mask, noc, b, Ct);
        read_tiles(hist_acc, hist, noc, bh * 4 + 1, 3);  // slots 1..3
        cur.reserve_back(1);
        zero_reserved(cur, noc, 1);
        pack_head_tile_user<Kt, Vt, Nk>(qkv_acc, cur, noc, hk, h, b, 0);
        noc.async_read_barrier();
        cur.push_back(1);
        // a[b,h], b[b,h]: row b of the a|b tile, columns h and Nv + h
        a_s.reserve_back(1);
        noc.async_read(qkv_acc, a_s, 2048, {.page_id = ab_page, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        uint32_t a_bits, b_bits;
        {
            auto lock = a_s.scoped_write_lock(1);
            auto p16 = lock.template get_ptr<volatile uint16_t>();
            a_bits = static_cast<uint32_t>(p16[tile_elem_index(b, h)]) << 16;
            b_bits = static_cast<uint32_t>(p16[tile_elem_index(b, Nv + h)]) << 16;
        }
        fill_scalar_tile(a_s, a_bits);
        a_s.push_back(1);
        b_s.reserve_back(1);
        fill_scalar_tile(b_s, b_bits);
        b_s.push_back(1);
        read_tiles(state_acc, state_in, noc, bh * KV, KV);
    }
}
