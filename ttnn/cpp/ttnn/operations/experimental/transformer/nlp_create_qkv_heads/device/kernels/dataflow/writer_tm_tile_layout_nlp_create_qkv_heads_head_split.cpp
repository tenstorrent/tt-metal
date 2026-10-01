// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1 writer (pairs with the head-split reader): per (sequence block,
// KV group) work unit, writes the group's Q heads, then its K head and V head.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Drains one tile from the qv buffer to output page `page_id`.
template <typename Accessor>
inline void write_tile(Noc& noc, DataflowBuffer& dfb, const Accessor& s, uint32_t page_id) {
    dfb.wait_front(1);
    noc.async_write(dfb, s, dfb.get_entry_size(), {}, {.page_id = page_id});
    noc.async_write_barrier();
    dfb.pop_front(1);
}

void kernel_main() {
    Noc noc;

    const uint32_t num_work_units = get_arg(args::num_work_units);
    const uint32_t work_unit_start = get_arg(args::work_unit_start);

    constexpr uint32_t q_out_h_tiles = get_arg(args::q_out_h_tiles);
    constexpr uint32_t q_out_w_tiles = get_arg(args::q_out_w_tiles);
    constexpr uint32_t q_out_HtWt = get_arg(args::q_out_HtWt);
    constexpr uint32_t num_q_heads = get_arg(args::num_q_heads);
    constexpr uint32_t num_kv_heads = get_arg(args::num_kv_heads);
    constexpr uint32_t q_heads_per_kv = get_arg(args::q_heads_per_kv);

    DataflowBuffer dfb_qv(dfb::qv);
    const auto sq = TensorAccessor(tensor::q);
    const auto sk = TensorAccessor(tensor::k);
    const auto sv = TensorAccessor(tensor::v);

    constexpr uint32_t q_tiles_per_group = q_heads_per_kv * q_out_w_tiles;
    constexpr uint32_t q_out_CHtWt = num_q_heads * q_out_HtWt;
    constexpr uint32_t kv_out_CHtWt = num_kv_heads * q_out_HtWt;

    for (uint32_t work = 0; work < num_work_units; ++work) {
        const uint32_t work_unit = work_unit_start + work;
        const uint32_t block = work_unit / num_kv_heads;
        const uint32_t kv_group = work_unit - block * num_kv_heads;
        const uint32_t batch = block / q_out_h_tiles;
        const uint32_t h_tile = block - batch * q_out_h_tiles;
        const uint32_t q_head_start = kv_group * q_heads_per_kv;

        const uint32_t q_tile_base = batch * q_out_CHtWt + q_head_start * q_out_HtWt + h_tile * q_out_w_tiles;
        for (uint32_t i = 0; i < q_tiles_per_group; ++i) {
            const uint32_t q_head_offset = i / q_out_w_tiles;
            const uint32_t w = i - q_head_offset * q_out_w_tiles;
            write_tile(noc, dfb_qv, sq, q_tile_base + q_head_offset * q_out_HtWt + w);
        }

        const uint32_t kv_tile_base = batch * kv_out_CHtWt + kv_group * q_out_HtWt + h_tile * q_out_w_tiles;
        for (uint32_t i = 0; i < q_out_w_tiles; ++i) {
            write_tile(noc, dfb_qv, sk, kv_tile_base + i);
        }
        for (uint32_t i = 0; i < q_out_w_tiles; ++i) {
            write_tile(noc, dfb_qv, sv, kv_tile_base + i);
        }
    }
}
