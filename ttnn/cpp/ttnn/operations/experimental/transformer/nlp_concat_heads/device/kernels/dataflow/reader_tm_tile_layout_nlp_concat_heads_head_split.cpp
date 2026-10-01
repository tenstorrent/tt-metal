// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1 reader: each work unit is one (batch, h_tile, head group) and
// reads heads_per_group heads x in0_w_tiles tiles, so the work splits across head_groups x more cores.
void kernel_main() {
    Noc noc;

    // Runtime args
    const uint32_t num_work_units = get_arg(args::num_work_units);
    const uint32_t work_unit_start = get_arg(args::work_unit_start);

    // Compile-time args
    constexpr uint32_t in0_h_tiles = get_arg(args::in0_h_tiles);
    constexpr uint32_t in0_w_tiles = get_arg(args::in0_w_tiles);
    constexpr uint32_t in0_c = get_arg(args::in0_c);
    constexpr uint32_t in0_HtWt = get_arg(args::in0_HtWt);
    constexpr uint32_t head_groups = get_arg(args::head_groups);
    constexpr uint32_t heads_per_group = get_arg(args::heads_per_group);

    DataflowBuffer dfb_in0(dfb::in0);
    const uint32_t single_tile_size_bytes = dfb_in0.get_entry_size();
    const auto s0 = TensorAccessor(tensor::src);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t group_tiles = heads_per_group * in0_w_tiles;
    constexpr uint32_t in0_CHtWt = in0_c * in0_HtWt;

    for (uint32_t work = 0; work < num_work_units; ++work) {
        const uint32_t work_unit = work_unit_start + work;
        const uint32_t block = work_unit / head_groups;
        const uint32_t group = work_unit - block * head_groups;
        const uint32_t batch = block / in0_h_tiles;
        const uint32_t h_tile = block - batch * in0_h_tiles;
        const uint32_t head_start = group * heads_per_group;

        const uint32_t base_tile = batch * in0_CHtWt + head_start * in0_HtWt + h_tile * in0_w_tiles;
        for (uint32_t i = 0; i < group_tiles; ++i) {
            const uint32_t head_offset = i / in0_w_tiles;
            const uint32_t w = i - head_offset * in0_w_tiles;
            dfb_in0.reserve_back(onetile);
            noc.async_read(
                s0, dfb_in0, single_tile_size_bytes, {.page_id = base_tile + head_offset * in0_HtWt + w}, {});
            noc.async_read_barrier();
            dfb_in0.push_back(onetile);
        }
    }
}
