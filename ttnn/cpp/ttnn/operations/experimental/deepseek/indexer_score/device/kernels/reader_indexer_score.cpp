// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for indexer_score: walks this core's flat span of causal-valid work
// units (QC q-tile-rows x kw k-tiles). On a new q-row-group pushes the
// resident w group (Hi*QC tiles) and, when all heads fit, the resident q
// group (Hi*QC*Dt). With HB < Hi the q head-group blocks (HB*QC*Dt) stream
// per unit instead. Per unit pushes the k chunk (kw*Dt). Builds the [diag,
// full] -inf mask tiles once.

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
constexpr bool stream_heads = HB < Hi;

/**
 * Stamp the diag (strict-upper) and full -inf mask tiles (chunk_start is
 * tile-aligned, so one diagonal tile suffices). Pushed once, permanently fronted.
 */
inline void build_mask_tiles(Noc noc) {
    CircularBuffer cb(cb_mask);
    cb.reserve_back(2);
    fill_causal_diagonal_tile_bf16<tile_bytes>(noc, cb_mask, /*tile_id=*/0);
    fill_neginf_tile<tile_bytes>(cb_mask, /*tile_id=*/1);
    cb.push_back(2);
}

/** Read q head-group block hb of group g: [QC][HB][Dt] (heads contiguous per
 *  row so HP head rows stride Dt for matmul_block), tile id = h*Sqt*Dt + s*Dt + d. */
template <typename QAcc>
inline void read_q_block(Noc noc, const QAcc& q_acc, uint32_t s0, uint32_t hb) {
    CircularBuffer cb(cb_q);
    cb.reserve_back(HB * QC * Dt);
    uint32_t ptr = cb.get_write_ptr();
    for (uint32_t r = 0; r < QC; ++r) {
        for (uint32_t h = hb; h < hb + HB; ++h) {
            const uint32_t base = h * Sqt * Dt + (s0 + r) * Dt;
            for (uint32_t d = 0; d < Dt; ++d) {
                noc.async_read(q_acc, CoreLocalMem<uint32_t>(ptr), tile_bytes, {.page_id = base + d}, {});
                ptr += tile_bytes;
            }
        }
    }
    noc.async_read_barrier();
    cb.push_back(HB * QC * Dt);
}

/** Read the resident w group: [QC][Hi], tile id = h*Sqt + s. */
template <typename WAcc>
inline void read_w_group(Noc noc, const WAcc& w_acc, uint32_t s0) {
    CircularBuffer cb(cb_w);
    cb.reserve_back(Hi * QC);
    uint32_t ptr = cb.get_write_ptr();
    for (uint32_t r = 0; r < QC; ++r) {
        for (uint32_t h = 0; h < Hi; ++h) {
            noc.async_read(w_acc, CoreLocalMem<uint32_t>(ptr), tile_bytes, {.page_id = h * Sqt + s0 + r}, {});
            ptr += tile_bytes;
        }
    }
    noc.async_read_barrier();
    cb.push_back(Hi * QC);
}

/** Read k chunk [kw][Dt] starting at k-tile c0, tile id = t*Dt + d.
 *  Always reserves/pushes the full KC*Dt so the 2-chunk ring stays half-aligned
 *  (a kw-sized push would wrap mid-block and overflow the CB); the kw..KC tail
 *  is left unread and never consumed. */
template <typename KAcc>
inline void read_k_chunk(Noc noc, const KAcc& k_acc, uint32_t c0, uint32_t kw) {
    CircularBuffer cb(cb_k);
    cb.reserve_back(KC * Dt);
    uint32_t ptr = cb.get_write_ptr();
    for (uint32_t c = 0; c < kw; ++c) {
        for (uint32_t d = 0; d < Dt; ++d) {
            noc.async_read(k_acc, CoreLocalMem<uint32_t>(ptr), tile_bytes, {.page_id = (c0 + c) * Dt + d}, {});
            ptr += tile_bytes;
        }
    }
    noc.async_read_barrier();
    cb.push_back(KC * Dt);
}

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t flat_start = get_arg_val<uint32_t>(3);
    const uint32_t flat_count = get_arg_val<uint32_t>(4);

    constexpr auto q_args = TensorAccessorArgs<8>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, tile_bytes);

    Noc noc;

    build_mask_tiles(noc);

    WorkUnitSpan span;
    span.start(flat_start);

    bool need_group = true;
    for (uint32_t i = 0; i < flat_count; ++i) {
        if (need_group) {
            read_w_group(noc, w_acc, span.s0());
            if constexpr (!stream_heads) {
                read_q_block(noc, q_acc, span.s0(), 0);
            }
            need_group = false;
        }
        read_k_chunk(noc, k_acc, span.c0(), span.kw());
        if constexpr (stream_heads) {
            // compute completes one tile at a time, streaming all head blocks per tile
            for (uint32_t rc = 0; rc < QC * span.kw(); ++rc) {
                for (uint32_t hb = 0; hb < Hi; hb += HB) {
                    read_q_block(noc, q_acc, span.s0(), hb);
                }
            }
        }
        need_group = span.advance();
    }
}
