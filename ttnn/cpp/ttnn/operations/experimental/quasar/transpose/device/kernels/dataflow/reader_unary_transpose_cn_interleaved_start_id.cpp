// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t N = get_arg(args::N);
    const uint32_t C = get_arg(args::C);
    const uint32_t HtWt = get_arg(args::HtWt);
    const uint32_t batch_step = get_arg(args::batch_step);      // CHtWt - HtWt
    const uint32_t channel_step = get_arg(args::channel_step);  // NCHtWt - HtWt
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);
    uint32_t hw = get_arg(args::hw);
    uint32_t n = get_arg(args::n);

    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t read_size = get_arg(args::read_size);

    // ublocks size defined in tiles
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer cb(dfb::in0);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t page_idx = start_id;
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb.reserve_back(onepage);
#ifdef CN_RM
        // Restored native sharded multi-page split (see common.hpp helper).
        // RM-only: the helper splits a logical "row page" across multiple shards laterally
        // when the buffer is BLOCK/WIDTH-sharded. TILE-layout path below must keep the
        // original single-page transfer since each tile is one indivisible NOC unit.
        const uint32_t cb_write_ptr = cb.get_write_ptr();
        tt::data_movement::common::noc_async_read_sharded(noc, cb_write_ptr, s, page_idx, 0, read_size);
#else
        noc.async_read(s, cb, page_size, {.page_id = page_idx}, {.offset_bytes = 0});
#endif
        noc.async_read_barrier();
        cb.push_back(onepage);
        page_idx++;
        hw++;
        if (hw == HtWt) {
            hw = 0;
            n++;
            page_idx += batch_step;
            if (n == N) {
                n = 0;
                page_idx -= channel_step;
            }
        }
    }
}
