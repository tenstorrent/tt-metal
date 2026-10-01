// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// PI0_MQA_HEAD_SPLIT=1 reader (MQA/GQA, Q-head parallel). Work unit = (sequence block, Q head):
//   kv_group = q_head_idx / q_heads_per_kv;  the first Q head of each KV group also reads its K and V.
// K and V are shared by the group's Q heads, so they are read (and written) once, by that first head;
// num work units scales with num_q_heads instead of num_kv_heads.
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

    // Same compile-time args as the head-split reader.
    constexpr uint32_t q_heads_per_kv = get_arg(args::q_heads_per_kv);
    constexpr uint32_t num_kv_heads = get_arg(args::num_kv_heads);
    constexpr uint32_t head_tiles = get_arg(args::head_tiles);
    constexpr uint32_t in0_w_tiles = get_arg(args::in0_w_tiles);

    DataflowBuffer dfb_qv(dfb::qv);
    const auto s0 = TensorAccessor(tensor::input_q);

    // Input tile layout per block: [Q_0..Q_{nq-1} K_0..K_{nkv-1} V_0..V_{nkv-1}], head_tiles tiles each.
    constexpr uint32_t num_q_heads = q_heads_per_kv * num_kv_heads;
    constexpr uint32_t q_tiles_total = num_q_heads * head_tiles;
    constexpr uint32_t kv_tiles_total = num_kv_heads * head_tiles;

    for (uint32_t work = 0; work < num_work_units; ++work) {
        const uint32_t work_unit = work_unit_start + work;
        const uint32_t block = work_unit / num_q_heads;
        const uint32_t q_head_idx = work_unit - block * num_q_heads;
        const uint32_t kv_group = q_head_idx / q_heads_per_kv;
        const bool is_first_in_kv = (q_head_idx - kv_group * q_heads_per_kv) == 0;
        const uint32_t block_base = block * in0_w_tiles;

        read_tiles(noc, dfb_qv, s0, block_base + q_head_idx * head_tiles, head_tiles);
        if (is_first_in_kv) {
            read_tiles(noc, dfb_qv, s0, block_base + q_tiles_total + kv_group * head_tiles, head_tiles);
            read_tiles(
                noc, dfb_qv, s0, block_base + q_tiles_total + kv_tiles_total + kv_group * head_tiles, head_tiles);
        }
    }
}
