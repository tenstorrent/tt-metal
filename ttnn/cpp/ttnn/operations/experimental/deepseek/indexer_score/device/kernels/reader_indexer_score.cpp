// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for indexer_score: walks this core's flat span of causal-valid work
// units (q_tiles_per_unit q-tile-rows x up-to-k_tiles_per_unit k-tiles). On a
// new q-row-group pushes the resident w group and, when all heads fit, the
// resident q group. With heads_per_group < num_heads the q head-group blocks
// stream per tile instead. Per unit pushes the k chunk. Builds the [diag,
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

constexpr uint32_t tile_bytes = get_tile_size(cb_q);    // q/w: bf16
constexpr uint32_t k_tile_bytes = get_tile_size(cb_k);  // k: bf16 or bfp8_b (smaller tile)

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

/** Read the q head-group block starting at head first_head for the group at q-tile-row q_row_start:
 *  [q_tiles_per_unit][heads_per_group][head_dim_tiles] (heads contiguous per row so a DEST pass's
 *  head rows stride head_dim_tiles for matmul_block), tile id = h*q_len_tiles*head_dim_tiles + s*head_dim_tiles + d. */
template <bool dma_off, typename QAcc>
inline void read_q_block(Noc noc, const QAcc& q_acc, uint32_t q_row_start, uint32_t first_head) {
    CircularBuffer cb(cb_q);
    cb.reserve_back(q_group_tiles);
    if constexpr (!dma_off) {  // compute-ceiling toggle: skip the NoC reads, still push the block
        uint32_t ptr = cb.get_write_ptr();
        for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
            for (uint32_t h = first_head; h < first_head + heads_per_group; ++h) {
                const uint32_t base = h * q_len_tiles * head_dim_tiles + (q_row_start + r) * head_dim_tiles;
                for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                    noc.async_read(q_acc, CoreLocalMem<uint32_t>(ptr), tile_bytes, {.page_id = base + d}, {});
                    ptr += tile_bytes;
                }
            }
        }
        noc.async_read_barrier();
    }
    cb.push_back(q_group_tiles);
}

/** Read the resident w group: [q_tiles_per_unit][num_heads], tile id = h*q_len_tiles + s. */
template <bool dma_off, typename WAcc>
inline void read_w_group(Noc noc, const WAcc& w_acc, uint32_t q_row_start) {
    CircularBuffer cb(cb_w);
    cb.reserve_back(w_group_tiles);
    if constexpr (!dma_off) {  // compute-ceiling toggle: skip the NoC reads, still push the group
        uint32_t ptr = cb.get_write_ptr();
        for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
            for (uint32_t h = 0; h < num_heads; ++h) {
                noc.async_read(
                    w_acc, CoreLocalMem<uint32_t>(ptr), tile_bytes, {.page_id = h * q_len_tiles + q_row_start + r}, {});
                ptr += tile_bytes;
            }
        }
        noc.async_read_barrier();
    }
    cb.push_back(w_group_tiles);
}

/** Read k chunk [k_tiles_in_unit][head_dim_tiles] starting at k_tile_start, tile id = t*head_dim_tiles + d.
 *  Always reserves/pushes the full k_tiles_per_unit*head_dim_tiles so the 2-chunk ring stays
 *  half-aligned (a partial push would wrap mid-block and overflow the CB); the unread tail
 *  is never consumed. */
template <bool dma_off, typename KAcc>
inline void read_k_chunk(Noc noc, const KAcc& k_acc, uint32_t k_tile_start, uint32_t k_tiles_in_unit) {
    CircularBuffer cb(cb_k);
    cb.reserve_back(k_chunk_tiles);
    if constexpr (!dma_off) {  // compute-ceiling toggle: skip the NoC reads, still push the chunk
        uint32_t ptr = cb.get_write_ptr();
        for (uint32_t c = 0; c < k_tiles_in_unit; ++c) {
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                noc.async_read(
                    k_acc,
                    CoreLocalMem<uint32_t>(ptr),
                    k_tile_bytes,
                    {.page_id = (k_tile_start + c) * head_dim_tiles + d},
                    {});
                ptr += k_tile_bytes;
            }
        }
        noc.async_read_barrier();
    }
    cb.push_back(k_chunk_tiles);
}

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t flat_start = get_arg_val<uint32_t>(3);
    const uint32_t flat_count = get_arg_val<uint32_t>(4);

    constexpr auto q_args = TensorAccessorArgs<num_common_ct_args>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    // DMA-off bitmask, appended last in the CT args (after the three TensorAccessors):
    // bit0=q, bit1=k, bit2=w. Each read still pushes its CB so compute runs; only the NoC
    // read is skipped, isolating that input's contribution to the reader's exposed time.
    constexpr uint32_t dma_mask = get_compile_time_arg_val(w_args.next_compile_time_args_offset());
    constexpr bool q_off = (dma_mask & 0b001u) != 0;
    constexpr bool k_off = (dma_mask & 0b010u) != 0;
    constexpr bool w_off = (dma_mask & 0b100u) != 0;
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, tile_bytes);

    Noc noc;

    build_mask_tiles(noc);

    WorkUnitSpan span;
    span.start(flat_start);

    bool need_group = true;
    for (uint32_t i = 0; i < flat_count; ++i) {
        if (need_group) {
            read_w_group<w_off>(noc, w_acc, span.q_tile_start());
            if constexpr (!stream_heads) {
                read_q_block<q_off>(noc, q_acc, span.q_tile_start(), 0);
            }
            need_group = false;
        }
        read_k_chunk<k_off>(noc, k_acc, span.k_tile_start(), span.k_tiles());
        if constexpr (stream_heads) {
            // one q-block per (r, c) output tile per head group; must match compute's
            // (r outer, c inner) tile order so each block lands in the tile that consumes it
            for (uint32_t tile_idx = 0; tile_idx < q_tiles_per_unit * span.k_tiles(); ++tile_idx) {
                for (uint32_t first_head = 0; first_head < num_heads; first_head += heads_per_group) {
                    read_q_block<q_off>(noc, q_acc, span.q_tile_start(), first_head);
                }
            }
        }
        need_group = span.advance();
    }
}
