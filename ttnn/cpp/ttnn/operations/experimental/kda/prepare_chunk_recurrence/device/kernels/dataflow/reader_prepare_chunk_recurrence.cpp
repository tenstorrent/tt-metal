// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

inline void fill_constant_tiles(DataflowBuffer& eye, DataflowBuffer& tril, DataflowBuffer& ones) {
    constexpr uint32_t one = 0x3F800000;
    constexpr uint32_t face_width = 16;
    constexpr uint32_t face_elements = face_width * face_width;
    constexpr uint32_t row_bytes = face_width * sizeof(uint32_t);
    constexpr uint32_t face_bytes = face_elements * sizeof(uint32_t);

    eye.reserve_back(1);
    tril.reserve_back(1);
    ones.reserve_back(1);
    Noc noc;
    noc.async_write_zeros(eye, eye.get_entry_size());
    noc.async_write_zeros(tril, tril.get_entry_size());
    noc.write_zeros_l1_barrier();

    volatile tt_l1_ptr uint32_t* eye_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eye.get_write_ptr());
    volatile tt_l1_ptr uint32_t* tril_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tril.get_write_ptr());
    volatile tt_l1_ptr uint32_t* ones_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones.get_write_ptr());

    // Seed one aligned row, then use local NoC reads to replicate it across the dense regions.
    for (uint32_t column = 0; column < face_width; ++column) {
        ones_tile[column] = one;
    }
    UnicastEndpoint self;
    const auto ones_row = noc_traits_t<UnicastEndpoint>::src_args_type{
        .noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = ones.get_write_ptr()};
    for (uint32_t row = 1; row < face_width; ++row) {
        noc.async_read(self, ones, row_bytes, ones_row, {.offset_bytes = row * row_bytes});
    }
    noc.async_read_barrier();
    const auto ones_face = noc_traits_t<UnicastEndpoint>::src_args_type{
        .noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = ones.get_write_ptr()};
    for (uint32_t face = 1; face < 4; ++face) {
        noc.async_read(self, ones, face_bytes, ones_face, {.offset_bytes = face * face_bytes});
    }
    noc.async_read(self, tril, face_bytes, ones_face, {.offset_bytes = 2 * face_bytes});
    noc.async_read_barrier();

    for (uint32_t row = 0; row < face_width; ++row) {
        for (uint32_t column = 0; column <= row; ++column) {
            tril_tile[row * face_width + column] = one;
        }
        eye_tile[row * face_width + row] = one;
    }
    const auto eye_face = noc_traits_t<UnicastEndpoint>::src_args_type{
        .noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = eye.get_write_ptr()};
    const auto tril_face = noc_traits_t<UnicastEndpoint>::src_args_type{
        .noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = tril.get_write_ptr()};
    noc.async_read(self, eye, face_bytes, eye_face, {.offset_bytes = 3 * face_bytes});
    noc.async_read(self, tril, face_bytes, tril_face, {.offset_bytes = 3 * face_bytes});
    noc.async_read_barrier();

    eye.push_back(1);
    tril.push_back(1);
    ones.push_back(1);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count, uint32_t NC, uint32_t HV, uint32_t Hk) {
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;

    const auto q_accessor = TensorAccessor(tensor::q);
    const auto k_accessor = TensorAccessor(tensor::k);
    const auto v_accessor = TensorAccessor(tensor::v);
    const auto g_accessor = TensorAccessor(tensor::g);
    const auto beta_accessor = TensorAccessor(tensor::beta);
    DataflowBuffer q(dfb::q);
    DataflowBuffer k(dfb::k);
    DataflowBuffer v(dfb::v);
    DataflowBuffer g(dfb::g);
    DataflowBuffer beta(dfb::beta);
    DataflowBuffer eye(dfb::eye);
    DataflowBuffer tril(dfb::tril);
    DataflowBuffer ones(dfb::ones);
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
    fill_constant_tiles(eye, tril, ones);

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
