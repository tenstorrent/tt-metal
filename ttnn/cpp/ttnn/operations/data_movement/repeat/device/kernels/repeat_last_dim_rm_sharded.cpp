// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RM last-dim sharded; noc_async_*_sharded with per-replica write offset.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

using namespace tt::data_movement::common;

void kernel_main() {
    const auto page_start = get_arg(args::page_start);
    const auto page_end = get_arg(args::page_end);
    const auto nop = get_arg(args::nop);

    constexpr auto original_page_size_bytes = get_arg(args::original_page_size_bytes);
    constexpr auto num_repeats = get_arg(args::num_repeats);

    if (nop == 1) {
        return;
    }

    const auto s = TensorAccessor(tensor::src);
    const auto d = TensorAccessor(tensor::dst);

    DataflowBuffer dfb(dfb::in0);
    dfb.reserve_back(1);
    const uint32_t cb_slot = dfb.get_write_ptr();
    dfb.push_back(1);

    Noc noc;

    for (uint32_t i = page_start; i < page_end; i++) {
        noc_async_read_sharded(noc, cb_slot, s, i, 0, original_page_size_bytes);
        noc.async_read_barrier();
        for (uint32_t k = 0; k < num_repeats; k++) {
            noc_async_write_sharded(noc, cb_slot, d, i, k * original_page_size_bytes, original_page_size_bytes);
        }
        noc.async_write_barrier();
    }
}
