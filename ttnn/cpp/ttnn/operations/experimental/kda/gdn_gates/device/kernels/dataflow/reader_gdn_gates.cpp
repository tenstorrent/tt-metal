// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// One 32-token tile row per work item: read the gab tile holding a and the tile holding b.
template <uint32_t gab_row_tiles, uint32_t a_col_tile, uint32_t b_col_tile>
TT_KERNEL void reader(uint32_t mt_start, uint32_t mt_count) {
    const auto gab_acc = TensorAccessor(tensor::gab);
    const auto dt_acc = TensorAccessor(tensor::dt_bias);
    const auto an_acc = TensorAccessor(tensor::a_neg);
    DataflowBuffer a(dfb::a);
    DataflowBuffer b(dfb::b);
    DataflowBuffer dt(dfb::dt);
    DataflowBuffer aneg(dfb::aneg);
    Noc noc;

    // dt_bias / a_neg tiles (one page each); row-broadcast in the compute kernel.
    dt.reserve_back(1);
    aneg.reserve_back(1);
    noc.async_read(dt_acc, dt, dt.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(an_acc, aneg, aneg.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    dt.push_back(1);
    aneg.push_back(1);

    for (uint32_t i = 0; i < mt_count; i++) {
        const uint32_t mt = mt_start + i;
        a.reserve_back(1);
        b.reserve_back(1);
        noc.async_read(
            gab_acc, a, a.get_entry_size(), {.page_id = mt * gab_row_tiles + a_col_tile}, {.offset_bytes = 0});
        noc.async_read(
            gab_acc, b, b.get_entry_size(), {.page_id = mt * gab_row_tiles + b_col_tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        a.push_back(1);
        b.push_back(1);
    }
}
