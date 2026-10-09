// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar nearest-neighbour upsample, sharded (height / block) row-major input: PRODUCER side.
//
// SPMD data-movement kernel (num_threads = T). Thread t owns output sticks t, t+T, t+2T, ... of this core's
// output shard. For each one it computes the source stick (nearest neighbour: ih = oh / scale_h,
// iw = ow / scale_w), turns it into a page id of the sharded INPUT tensor (one page = one shard-wide row; for
// block sharding the row is split into ncols pages and this core reads the column it owns) and issues ONE
// implicit-sync NoC read into the staging DFB `stage`:
//
//     noc.async_read<NocOptions::TXN_ID>(input, stage, {.page_id = src_page}, {});
//
// The read lands in this thread's next ring entry (STRIDED producer), is stamped with one of the DFB's
// transaction ids, and the DM0 ISR posts the entry's credit when the NoC reports completion. No reserve /
// push / barrier; finish() hands over the tail credits. writer_upsample_sharded_implicit.cpp consumes the
// ring with matching thread ownership (entry i <-> stick i, both sides run T threads).
//
// Compile args: scale_h, scale_w, batch, in_h, in_w, out_h, out_w, pages_per_row, out_nsticks_per_core
// Runtime args: out_row_start (first output row of this core's shard), col (block-sharded column index)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t scale_h = get_arg(args::scale_h);
    constexpr uint32_t scale_w = get_arg(args::scale_w);
    constexpr uint32_t in_h = get_arg(args::in_h);
    constexpr uint32_t in_w = get_arg(args::in_w);
    constexpr uint32_t out_h = get_arg(args::out_h);
    constexpr uint32_t out_w = get_arg(args::out_w);
    constexpr uint32_t pages_per_row = get_arg(args::pages_per_row);
    constexpr uint32_t out_nsticks_per_core = get_arg(args::out_nsticks_per_core);
    const uint32_t out_row_start = get_arg(args::out_row_start);
    const uint32_t col = get_arg(args::col);

    constexpr uint32_t out_hw = out_h * out_w;

    const uint32_t num_threads = get_num_threads();
    const uint32_t thread_id = get_my_thread_id();

    const auto input = TensorAccessor(tensor::input);
    DataflowBuffer stage(dfb::stage);
    Noc noc;

    for (uint32_t i = thread_id; i < out_nsticks_per_core; i += num_threads) {
        const uint32_t out_row = out_row_start + i;
        const uint32_t n = out_row / out_hw;
        const uint32_t rem = out_row - n * out_hw;
        const uint32_t oh = rem / out_w;
        const uint32_t ow = rem - oh * out_w;
        const uint32_t in_row = (n * in_h + oh / scale_h) * in_w + ow / scale_w;
        const uint32_t src_page = in_row * pages_per_row + col;
        noc.async_read<NocOptions::TXN_ID>(input, stage, {.page_id = src_page}, {});
    }
    stage.finish();
}
