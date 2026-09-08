// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
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
    const auto cs1_acc = TensorAccessor(tensor::cs1);
    const auto cs2_acc = TensorAccessor(tensor::cs2);
    const auto cs3_acc = TensorAccessor(tensor::cs3);
    const auto tap0_acc = TensorAccessor(tensor::tap0);
    const auto tap1_acc = TensorAccessor(tensor::tap1);
    const auto tap2_acc = TensorAccessor(tensor::tap2);
    const auto tap3_acc = TensorAccessor(tensor::tap3);
    DataflowBuffer hist1(dfb::hist1);
    DataflowBuffer hist2(dfb::hist2);
    DataflowBuffer hist3(dfb::hist3);
    DataflowBuffer cur(dfb::cur);
    DataflowBuffer tap0(dfb::tap0);
    DataflowBuffer tap1(dfb::tap1);
    DataflowBuffer tap2(dfb::tap2);
    DataflowBuffer tap3(dfb::tap3);
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
    constexpr uint32_t rf = Nv / Nk;

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    generate_bcast_col_scalar(eps_l2, l2_eps_bits);
    generate_bcast_col_scalar(eps_norm, norm_eps_bits);
    build_row0_mask(mask, noc);
    read_tiles(w_acc, w_in, noc, 0, Vt);

    for (uint32_t i = 0; i < wi_count; ++i) {
        const uint32_t h = wi_start + i;
        const uint32_t hk = h / rf;
        if (i == 0) {  // taps are per-channel constants: one head slice per core
            read_head_row<Kt, Vt, Nk>(tap0_acc, tap0, noc, hk, h);
            read_head_row<Kt, Vt, Nk>(tap1_acc, tap1, noc, hk, h);
            read_head_row<Kt, Vt, Nk>(tap2_acc, tap2, noc, hk, h);
            read_head_row<Kt, Vt, Nk>(tap3_acc, tap3, noc, hk, h);
        }
        read_head_row<Kt, Vt, Nk>(cs1_acc, hist1, noc, hk, h);
        read_head_row<Kt, Vt, Nk>(cs2_acc, hist2, noc, hk, h);
        read_head_row<Kt, Vt, Nk>(cs3_acc, hist3, noc, hk, h);
        read_head_row<Kt, Vt, Nk>(qkv_acc, cur, noc, hk, h);
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
