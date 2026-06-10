// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for indexer_score: walks this core's flat span of causal-valid
// output tiles in row-major order. On a new q-tile-row pushes the resident
// q row (Hi*Dt tiles) and w row (Hi tiles); per output tile pushes the k
// column (Dt tiles). Also builds the diagonal -inf mask tile once.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

#include "indexer_score_common.hpp"

constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_w = tt::CBIndex::c_2;
constexpr uint32_t cb_mask = tt::CBIndex::c_3;

constexpr uint32_t tile_bytes = get_tile_size(cb_q);

// strictly-upper-triangle -inf, zero elsewhere (chunk_start is tile-aligned)
inline void build_mask_tile() {
    cb_reserve_back(cb_mask, 1);
    Noc noc;
    fill_causal_diagonal_tile_bf16<tile_bytes>(noc, cb_mask, /*tile_id=*/0);
    cb_push_back(cb_mask, 1);
}

// resident q row (tile id = h*Sqt*Dt + s*Dt + d) and w row (tile id = h*Sqt + s)
template <typename QAcc, typename WAcc>
inline void read_qw_row(const QAcc& q_acc, const WAcc& w_acc, uint32_t s) {
    cb_reserve_back(cb_q, Hi * Dt);
    uint32_t wr = get_write_ptr(cb_q);
    for (uint32_t h = 0; h < Hi; ++h) {
        uint32_t base = h * Sqt * Dt + s * Dt;
        for (uint32_t d = 0; d < Dt; ++d) {
            noc_async_read_tile(base + d, q_acc, wr);
            wr += tile_bytes;
        }
    }
    cb_reserve_back(cb_w, Hi);
    uint32_t wwr = get_write_ptr(cb_w);
    for (uint32_t h = 0; h < Hi; ++h) {
        noc_async_read_tile(h * Sqt + s, w_acc, wwr);
        wwr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_q, Hi * Dt);
    cb_push_back(cb_w, Hi);
}

// k column t: tile id = t*Dt + d
template <typename KAcc>
inline void read_k_column(const KAcc& k_acc, uint32_t t) {
    cb_reserve_back(cb_k, Dt);
    uint32_t kwr = get_write_ptr(cb_k);
    for (uint32_t d = 0; d < Dt; ++d) {
        noc_async_read_tile(t * Dt + d, k_acc, kwr);
        kwr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_k, Dt);
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

    build_mask_tile();

    ValidTileSpan span;
    span.start(flat_start);

    bool need_row = true;
    for (uint32_t i = 0; i < flat_count; ++i) {
        if (need_row) {
            read_qw_row(q_acc, w_acc, span.s);
            need_row = false;
        }
        read_k_column(k_acc, span.t);
        need_row = span.advance();
    }
}
