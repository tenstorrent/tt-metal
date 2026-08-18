// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count, uint32_t NC, uint32_t HV, uint32_t Hk) {
    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;

    const auto q_accessor = TensorAccessor(tensor::q);
    const auto k_accessor = TensorAccessor(tensor::k);
    const auto v_accessor = TensorAccessor(tensor::v);
    const auto g_accessor = TensorAccessor(tensor::g);
    const auto beta_accessor = TensorAccessor(tensor::beta);
    const auto eye_accessor = TensorAccessor(tensor::eye);
    const auto tril_accessor = TensorAccessor(tensor::tril);
    const auto ones_accessor = TensorAccessor(tensor::ones);
    const auto masks_accessor = TensorAccessor(tensor::masks);
    DataflowBuffer q(dfb::q);
    DataflowBuffer k(dfb::k);
    DataflowBuffer v(dfb::v);
    DataflowBuffer g(dfb::g);
    DataflowBuffer beta(dfb::beta);
    DataflowBuffer eye(dfb::eye);
    DataflowBuffer tril(dfb::tril);
    DataflowBuffer ones(dfb::ones);
    DataflowBuffer masks(dfb::u);
    Noc noc;

    auto read_into = [&](const auto& accessor, DataflowBuffer& buffer, uint32_t base, uint32_t tiles) {
        buffer.reserve_back(tiles);
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            noc.async_read(
                accessor,
                buffer,
                buffer.get_entry_size(),
                {.page_id = base + tile},
                {.offset_bytes = tile * buffer.get_entry_size()});
        }
        noc.async_read_barrier();
        buffer.push_back(tiles);
    };
    read_into(eye_accessor, eye, 0, cc);
    read_into(tril_accessor, tril, 0, cc);
    read_into(ones_accessor, ones, 0, cc);
    read_into(masks_accessor, masks, 0, 3);

    auto read_v_flat = [&](uint32_t hc) {
        const uint32_t bh = hc / NC;
        const uint32_t chunk = hc % NC;
        const uint32_t value_head = bh % HV;
        const uint32_t batch = bh / HV;
        const uint32_t row_stride = HV * Vt;
        const uint32_t batch_base = batch * NC * Ct * row_stride;
        v.reserve_back(cv);
        for (uint32_t row = 0; row < Ct; ++row) {
            for (uint32_t col = 0; col < Vt; ++col) {
                const uint32_t page = batch_base + (chunk * Ct + row) * row_stride + value_head * Vt + col;
                noc.async_read(
                    v_accessor,
                    v,
                    v.get_entry_size(),
                    {.page_id = page},
                    {.offset_bytes = (row * Vt + col) * v.get_entry_size()});
            }
        }
        noc.async_read_barrier();
        v.push_back(cv);
    };
    auto read_qk_flat = [&](const auto& accessor, DataflowBuffer& buffer, uint32_t hc) {
        const uint32_t group_size = HV / Hk;
        const uint32_t bh = hc / NC;
        const uint32_t chunk = hc % NC;
        const uint32_t value_head = bh % HV;
        const uint32_t batch = bh / HV;
        const uint32_t key_head = value_head / group_size;
        const uint32_t row_stride = Hk * Kt;
        const uint32_t batch_base = batch * NC * Ct * row_stride;
        buffer.reserve_back(ck);
        for (uint32_t row = 0; row < Ct; ++row) {
            for (uint32_t col = 0; col < Kt; ++col) {
                const uint32_t page = batch_base + (chunk * Ct + row) * row_stride + key_head * Kt + col;
                noc.async_read(
                    accessor,
                    buffer,
                    buffer.get_entry_size(),
                    {.page_id = page},
                    {.offset_bytes = (row * Kt + col) * buffer.get_entry_size()});
            }
        }
        noc.async_read_barrier();
        buffer.push_back(ck);
    };
    auto read_g_flat = [&](uint32_t hc) {
        const uint32_t bh = hc / NC;
        const uint32_t chunk = hc % NC;
        const uint32_t value_head = bh % HV;
        const uint32_t batch = bh / HV;
        const uint32_t row_stride = HV * Kt;
        const uint32_t batch_base = batch * NC * Ct * row_stride;
        g.reserve_back(ck);
        for (uint32_t row = 0; row < Ct; ++row) {
            for (uint32_t col = 0; col < Kt; ++col) {
                const uint32_t page = batch_base + (chunk * Ct + row) * row_stride + value_head * Kt + col;
                noc.async_read(
                    g_accessor,
                    g,
                    g.get_entry_size(),
                    {.page_id = page},
                    {.offset_bytes = (row * Kt + col) * g.get_entry_size()});
            }
        }
        noc.async_read_barrier();
        g.push_back(ck);
    };

    for (uint32_t index = 0; index < wi_count; ++index) {
        const uint32_t hc = wi_start + index;
        read_qk_flat(q_accessor, q, hc);
        read_qk_flat(k_accessor, k, hc);
        read_v_flat(hc);
        read_g_flat(hc);
        read_into(beta_accessor, beta, hc * Ct, Ct);
    }
}
