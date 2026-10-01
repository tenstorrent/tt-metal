// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1 reader. Work unit = (sequence block, KV group): reads the group's
// q_heads_per_kv Q heads, then its K head and V head, so the work splits across num_kv_heads x more cores.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Reads `count` consecutive input tiles starting at `first_tile` into the qv buffer, one tile per barrier.
template <typename Accessor>
inline void read_tiles(Noc& noc, DataflowBuffer& dfb, const Accessor& s, uint32_t first_tile, uint32_t count) {
    const uint32_t tile_bytes = dfb.get_entry_size();
    for (uint32_t i = 0; i < count; ++i) {
        dfb.reserve_back(1);
        noc.async_read(s, dfb, tile_bytes, {.page_id = first_tile + i}, {});
        noc.async_read_barrier();
        dfb.push_back(1);
    }
}

void kernel_main() {
    Noc noc;

    const uint32_t num_work_units = get_arg(args::num_work_units);
    const uint32_t work_unit_start = get_arg(args::work_unit_start);

    constexpr uint32_t q_heads_per_kv = get_arg(args::q_heads_per_kv);
    constexpr uint32_t num_kv_heads = get_arg(args::num_kv_heads);
    constexpr uint32_t head_tiles = get_arg(args::head_tiles);
    constexpr uint32_t in0_w_tiles = get_arg(args::in0_w_tiles);

    DataflowBuffer dfb_qv(dfb::qv);
    const auto s0 = TensorAccessor(tensor::input_q);

    constexpr uint32_t q_tiles_per_group = q_heads_per_kv * head_tiles;
    constexpr uint32_t q_tiles_total = q_heads_per_kv * num_kv_heads * head_tiles;
    constexpr uint32_t kv_tiles_total = num_kv_heads * head_tiles;

    for (uint32_t work = 0; work < num_work_units; ++work) {
        const uint32_t work_unit = work_unit_start + work;
        const uint32_t block = work_unit / num_kv_heads;
        const uint32_t kv_group = work_unit - block * num_kv_heads;
        const uint32_t block_base = block * in0_w_tiles;

        read_tiles(noc, dfb_qv, s0, block_base + kv_group * q_tiles_per_group, q_tiles_per_group);
        read_tiles(noc, dfb_qv, s0, block_base + q_tiles_total + kv_group * head_tiles, head_tiles);
        read_tiles(noc, dfb_qv, s0, block_base + q_tiles_total + kv_tiles_total + kv_group * head_tiles, head_tiles);
    }
}
