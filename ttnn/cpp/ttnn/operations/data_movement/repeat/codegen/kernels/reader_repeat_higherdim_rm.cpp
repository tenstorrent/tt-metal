// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Repeat-local reader for HIGHER-DIM replication on RM interleaved tensors.
//
// This mirrors the shared reader_repeat_higherdim_rm.cpp mapping, with
// compile-time shortcuts for size-1 repeated dimensions. Tiny repeat cases often
// broadcast a singleton dim; avoiding the full generic div/mod map keeps those
// cases from losing to TTNN's collapsed RM path.
//
// Named CT args: xfer_size, l1_stride, num_repeats, lower_pages, rep_dim_pages, batch
// Bindings:      tensor::src (input tensor), dfb::in (staging buffer this reader fills)
// Named RT args: num_out_pages, out_start_page
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_out_pages = get_arg(args::num_out_pages);
    uint32_t out_start_page = get_arg(args::out_start_page);

    constexpr uint32_t xfer_size = get_arg(args::xfer_size);
    constexpr uint32_t l1_stride = get_arg(args::l1_stride);
    constexpr uint32_t NUM_REPEATS = get_arg(args::num_repeats);
    constexpr uint32_t LOWER_PAGES = get_arg(args::lower_pages);
    constexpr uint32_t REP_DIM_PAGES = get_arg(args::rep_dim_pages);
    constexpr uint32_t BATCH = get_arg(args::batch);

    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    // dfb::in — one l1_stride slot per output page, filled here, drained by the writer.
    DataflowBuffer dfb_in(dfb::in);

    constexpr uint32_t SRC_LOWER = REP_DIM_PAGES * LOWER_PAGES;
    constexpr uint32_t DST_LOWER = NUM_REPEATS * SRC_LOWER;

    uint32_t out_page = out_start_page;
    uint32_t pages_left = num_out_pages;

    while (pages_left > 0) {
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        dfb_in.reserve_back(batch);
        uint32_t l1_offset = 0;

        for (uint32_t t = 0; t < batch; t++) {
            uint32_t src_page;

            if constexpr (REP_DIM_PAGES == 1 && LOWER_PAGES == 1) {
                src_page = out_page / NUM_REPEATS;
            } else if constexpr (REP_DIM_PAGES == 1) {
                uint32_t block = out_page / (NUM_REPEATS * LOWER_PAGES);
                uint32_t within = out_page % (NUM_REPEATS * LOWER_PAGES);
                src_page = block * LOWER_PAGES + (within % LOWER_PAGES);
            } else {
                uint32_t block = out_page / DST_LOWER;
                uint32_t within = out_page % DST_LOWER;
                uint32_t lower_in_rep = within % SRC_LOWER;
                src_page = block * SRC_LOWER + lower_in_rep;
            }

            noc.async_read(s, dfb_in, xfer_size, {.page_id = src_page, .offset_bytes = 0}, {.offset_bytes = l1_offset});
            l1_offset += l1_stride;
            out_page++;
        }
        noc.async_read_barrier();
        dfb_in.push_back(batch);
        pages_left -= batch;
    }
}
