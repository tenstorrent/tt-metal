// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"

namespace {

template <typename Accessor>
FORCE_INLINE void read_tiles(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count) {
    dfb.reserve_back(count);
    const uint32_t entry = dfb.get_entry_size();
    for (uint32_t t = 0; t < count; ++t) {
        noc.async_read(acc, dfb, entry, {.page_id = first_page + t}, {.offset_bytes = t * entry});
    }
    noc.async_read_barrier();
    dfb.push_back(count);
}

// Read tile 0 of a [1, 1, H] tensor and broadcast element (row 0, col `col`) over a whole fp32 tile.
template <bool src_fp32, typename Accessor>
FORCE_INLINE void load_head_scalar(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t col) {
    dfb.reserve_back(1);
    constexpr uint32_t src_bytes = src_fp32 ? 4096 : 2048;
    noc.async_read(acc, dfb, src_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    {
        auto lock = dfb.scoped_write_lock(1);
        const uint32_t idx = (col < 16 ? 0 : 256) + (col & 15);  // face-major 16x16 tile layout, row 0
        uint32_t value;
        if constexpr (src_fp32) {
            auto p = lock.template get_ptr<volatile uint32_t>();
            value = p[idx];
        } else {
            auto p16 = lock.template get_ptr<volatile uint16_t>();
            value = static_cast<uint32_t>(p16[idx]) << 16;
        }
        auto p32 = lock.template get_ptr<volatile uint32_t>();
        for (uint32_t i = 0; i < 1024; ++i) {
            p32[i] = value;
        }
    }
    dfb.push_back(1);
}

// bf16 column mask: 1.0 in row 0 (column 0), 0 elsewhere.
FORCE_INLINE void build_row0_mask(DataflowBuffer& dfb, Noc& noc) {
    dfb.reserve_back(1);
    noc.async_write_zeros(dfb, dfb.get_entry_size());
    noc.write_zeros_l1_barrier();
    {
        auto lock = dfb.scoped_write_lock(1);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        p16[0] = 0x3F80;
    }
    dfb.push_back(1);
}

}  // namespace

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Nk,
    uint32_t Nv,
    uint32_t beta_fp32,
    uint32_t g_fp32,
    uint32_t l2_eps_bits,
    uint32_t norm_eps_bits>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    const auto qkv_acc = TensorAccessor(tensor::qkv);
    const auto beta_acc = TensorAccessor(tensor::beta);
    const auto g_acc = TensorAccessor(tensor::g);
    const auto state_acc = TensorAccessor(tensor::state);
    const auto w_acc = TensorAccessor(tensor::weight);
    DataflowBuffer q_in(dfb::q_in);
    DataflowBuffer k_in(dfb::k_in);
    DataflowBuffer v_in(dfb::v_in);
    DataflowBuffer beta_s(dfb::beta_s);
    DataflowBuffer g_s(dfb::g_s);
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
        read_tiles(qkv_acc, q_in, noc, hk * Kt, Kt);
        read_tiles(qkv_acc, k_in, noc, Nk * Kt + hk * Kt, Kt);
        read_tiles(qkv_acc, v_in, noc, 2 * Nk * Kt + h * Vt, Vt);
        load_head_scalar<beta_fp32 != 0>(beta_acc, beta_s, noc, h);
        load_head_scalar<g_fp32 != 0>(g_acc, g_s, noc, h);
        read_tiles(state_acc, state_in, noc, h * KV, KV);
    }
}
