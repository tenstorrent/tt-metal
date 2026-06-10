// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for indexer_score: walks this core's flat span of causal-valid
// output tiles in row-major order. On a new q-tile-row pushes the resident
// q row (Hi*Dt tiles) and w row (Hi tiles); per output tile pushes the k
// column (Dt tiles). Also builds the diagonal -inf mask tile once.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

#include "indexer_score_common.hpp"

constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_w = tt::CBIndex::c_2;
constexpr uint32_t cb_mask = tt::CBIndex::c_3;

constexpr uint32_t tile_bytes = get_tile_size(cb_q);

/**
 * Stamp the strictly-upper-triangle -inf mask tile (chunk_start is tile-aligned,
 * so one diagonal tile suffices). Pushed once and permanently fronted.
 */
inline void build_mask_tile(Noc noc) {
    CircularBuffer cb(cb_mask);
    cb.reserve_back(1);
    fill_causal_diagonal_tile_bf16<tile_bytes>(noc, cb_mask, /*tile_id=*/0);
    cb.push_back(1);
}

/**
 * Read the resident q row (tile id = h*Sqt*Dt + s*Dt + d) and w row (tile id = h*Sqt + s)
 * for q-tile-row s. One barrier covers both reads; rows stay fronted across the whole row.
 */
template <typename QAcc, typename WAcc>
inline void read_qw_row(Noc noc, const QAcc& q_acc, const WAcc& w_acc, uint32_t s) {
    CircularBuffer cb_q_in(cb_q);
    cb_q_in.reserve_back(Hi * Dt);
    uint32_t q_ptr = cb_q_in.get_write_ptr();
    for (uint32_t h = 0; h < Hi; ++h) {
        const uint32_t base = h * Sqt * Dt + s * Dt;
        for (uint32_t d = 0; d < Dt; ++d) {
            noc.async_read(q_acc, CoreLocalMem<uint32_t>(q_ptr), tile_bytes, {.page_id = base + d}, {});
            q_ptr += tile_bytes;
        }
    }
    CircularBuffer cb_w_in(cb_w);
    cb_w_in.reserve_back(Hi);
    uint32_t w_ptr = cb_w_in.get_write_ptr();
    for (uint32_t h = 0; h < Hi; ++h) {
        noc.async_read(w_acc, CoreLocalMem<uint32_t>(w_ptr), tile_bytes, {.page_id = h * Sqt + s}, {});
        w_ptr += tile_bytes;
    }
    noc.async_read_barrier();
    cb_q_in.push_back(Hi * Dt);
    cb_w_in.push_back(Hi);
}

/** Read k column t (tile id = t*Dt + d). */
template <typename KAcc>
inline void read_k_column(Noc noc, const KAcc& k_acc, uint32_t t) {
    CircularBuffer cb_k_in(cb_k);
    cb_k_in.reserve_back(Dt);
    uint32_t k_ptr = cb_k_in.get_write_ptr();
    for (uint32_t d = 0; d < Dt; ++d) {
        noc.async_read(k_acc, CoreLocalMem<uint32_t>(k_ptr), tile_bytes, {.page_id = t * Dt + d}, {});
        k_ptr += tile_bytes;
    }
    noc.async_read_barrier();
    cb_k_in.push_back(Dt);
}

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t flat_start = get_arg_val<uint32_t>(3);
    const uint32_t flat_count = get_arg_val<uint32_t>(4);

    constexpr auto q_args = TensorAccessorArgs<5>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, tile_bytes);

    Noc noc;

    build_mask_tile(noc);

    ValidTileSpan span;
    span.start(flat_start);

    bool need_row = true;
    for (uint32_t i = 0; i < flat_count; ++i) {
        if (need_row) {
            read_qw_row(noc, q_acc, w_acc, span.s);
            need_row = false;
        }
        read_k_column(noc, k_acc, span.t);
        need_row = span.advance();
    }
}
