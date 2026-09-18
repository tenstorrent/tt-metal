// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "tt-metalium/constants.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

inline void fill_constant_tiles(
    DataflowBuffer& eye, DataflowBuffer& tril, DataflowBuffer& ones, DataflowBuffer& block_masks) {
    constexpr uint32_t fp32_one_bits = __builtin_bit_cast(uint32_t, 1.0F);
    constexpr uint32_t face_width = tt::constants::FACE_WIDTH;
    constexpr uint32_t face_elements = tt::constants::FACE_HW;
    constexpr uint32_t row_bytes = face_width * sizeof(uint32_t);
    constexpr uint32_t face_bytes = face_elements * sizeof(uint32_t);

    eye.reserve_back(1);
    tril.reserve_back(1);
    ones.reserve_back(1);
    block_masks.reserve_back(2);
    Noc noc;
    noc.async_write_zeros(eye, eye.get_entry_size());
    noc.async_write_zeros(tril, tril.get_entry_size());
    noc.async_write_zeros(block_masks, 2 * block_masks.get_entry_size());
    noc.write_zeros_l1_barrier();

    volatile tt_l1_ptr uint32_t* eye_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eye.get_write_ptr());
    volatile tt_l1_ptr uint32_t* tril_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tril.get_write_ptr());
    volatile tt_l1_ptr uint32_t* ones_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones.get_write_ptr());

    // Seed one aligned row, then use local NoC reads to replicate it across the dense regions.
    for (uint32_t column = 0; column < face_width; ++column) {
        ones_tile[column] = fp32_one_bits;
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
    // Tile 0 selects the two diagonal 16x16 faces; tile 1 selects the bottom-left face.
    noc.async_read(self, block_masks, face_bytes, ones_face, {.offset_bytes = 0});
    noc.async_read(self, block_masks, face_bytes, ones_face, {.offset_bytes = 3 * face_bytes});
    noc.async_read(
        self, block_masks, face_bytes, ones_face, {.offset_bytes = block_masks.get_entry_size() + 2 * face_bytes});

    noc.async_read_barrier();

    for (uint32_t row = 0; row < face_width; ++row) {
        for (uint32_t column = 0; column <= row; ++column) {
            tril_tile[row * face_width + column] = fp32_one_bits;
        }
        eye_tile[row * face_width + row] = fp32_one_bits;
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
    block_masks.push_back(2);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
TT_KERNEL void reader(uint32_t work_item_start, uint32_t work_item_count, uint32_t num_chunks, uint32_t num_heads) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t chunk_value_tiles = Ct * Vt;

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
    DataflowBuffer block_masks(dfb::block_masks);
    Noc noc;

    auto enqueue_contiguous_read = [&](const auto& accessor, DataflowBuffer& buffer, uint32_t base, uint32_t tiles) {
        buffer.reserve_back(tiles);
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            noc.async_read(
                accessor,
                buffer,
                buffer.get_entry_size(),
                {.page_id = base + tile},
                {.offset_bytes = tile * buffer.get_entry_size()});
        }
    };
    fill_constant_tiles(eye, tril, ones, block_masks);

    auto enqueue_value_read = [&](uint32_t head_chunk_index) {
        const uint32_t head = head_chunk_index / num_chunks;
        const uint32_t chunk = head_chunk_index % num_chunks;
        const uint32_t row_stride = num_heads * Vt;
        v.reserve_back(chunk_value_tiles);
        for (uint32_t row = 0; row < Ct; ++row) {
            for (uint32_t col = 0; col < Vt; ++col) {
                const uint32_t page = (chunk * Ct + row) * row_stride + head * Vt + col;
                noc.async_read(
                    v_accessor,
                    v,
                    v.get_entry_size(),
                    {.page_id = page},
                    {.offset_bytes = (row * Vt + col) * v.get_entry_size()});
            }
        }
    };
    auto enqueue_key_width_read = [&](const auto& accessor, DataflowBuffer& buffer, uint32_t head_chunk_index) {
        const uint32_t head = head_chunk_index / num_chunks;
        const uint32_t chunk = head_chunk_index % num_chunks;
        const uint32_t row_stride = num_heads * Kt;
        buffer.reserve_back(chunk_key_tiles);
        for (uint32_t row = 0; row < Ct; ++row) {
            for (uint32_t col = 0; col < Kt; ++col) {
                const uint32_t page = (chunk * Ct + row) * row_stride + head * Kt + col;
                noc.async_read(
                    accessor,
                    buffer,
                    buffer.get_entry_size(),
                    {.page_id = page},
                    {.offset_bytes = (row * Kt + col) * buffer.get_entry_size()});
            }
        }
    };

    for (uint32_t index = 0; index < work_item_count; ++index) {
        const uint32_t head_chunk_index = work_item_start + index;
        enqueue_key_width_read(q_accessor, q, head_chunk_index);
        enqueue_key_width_read(k_accessor, k, head_chunk_index);
        enqueue_value_read(head_chunk_index);
        enqueue_key_width_read(g_accessor, g, head_chunk_index);
        enqueue_contiguous_read(beta_accessor, beta, head_chunk_index * Ct, Ct);
        // All five inputs are independent reads on the same NoC. One barrier lets them overlap, then publishes
        // the complete work item atomically to compute.
        noc.async_read_barrier();
        q.push_back(chunk_key_tiles);
        k.push_back(chunk_key_tiles);
        v.push_back(chunk_value_tiles);
        g.push_back(chunk_key_tiles);
        beta.push_back(Ct);
    }
}
