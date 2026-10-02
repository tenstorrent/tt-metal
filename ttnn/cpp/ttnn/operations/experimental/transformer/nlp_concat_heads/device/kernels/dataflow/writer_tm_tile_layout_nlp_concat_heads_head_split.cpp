// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1 writer: drains one (batch, h_tile, head group) work unit —
// group_tiles contiguous output tiles — per iteration (pairs with the head-split reader).
void kernel_main() {
    Noc noc;

    // Runtime args
    const uint32_t num_work_units = get_arg(args::num_work_units);
    const uint32_t work_unit_start = get_arg(args::work_unit_start);

    // Compile-time args
    constexpr uint32_t head_groups = get_arg(args::head_groups);
    constexpr uint32_t heads_per_group = get_arg(args::heads_per_group);
    constexpr uint32_t in0_w_tiles = get_arg(args::in0_w_tiles);
    constexpr uint32_t per_tensor_tiles = get_arg(args::per_tensor_tiles);

    DataflowBuffer dfb_in0(dfb::in0);
    const uint32_t single_tile_size_bytes = dfb_in0.get_entry_size();
    const auto s0 = TensorAccessor(tensor::dst);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t group_tiles = heads_per_group * in0_w_tiles;

    for (uint32_t work = 0; work < num_work_units; ++work) {
        const uint32_t work_unit = work_unit_start + work;
        const uint32_t block = work_unit / head_groups;
        const uint32_t group = work_unit - block * head_groups;
        const uint32_t out_tile_base = block * per_tensor_tiles + group * group_tiles;

        for (uint32_t i = 0; i < group_tiles; ++i) {
            dfb_in0.wait_front(onetile);
            noc.async_write(dfb_in0, s0, single_tile_size_bytes, {}, {.page_id = out_tile_base + i});
            noc.async_write_barrier();
            dfb_in0.pop_front(onetile);
        }
    }
}
