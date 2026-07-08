// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t stick_size = get_arg(args::stick_size);
    uint32_t stick_size_offset = get_arg(args::stick_size_offset);
    uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    uint32_t num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    uint32_t num_read_per_barrier = get_arg(args::num_read_per_barrier);
    uint32_t start_id = get_arg(args::start_id);
    // Per-shard page size: shard_W on B/W-sharded outputs, full row otherwise.
    // Feeds `noc_async_write_sharded`'s multi-shard split via `get_aligned_page_size()`.
    uint32_t page_size_override = get_arg(args::page_size_override);

    // Override the binding's page size (per-shard width); keep the auto-patched binding address
    // (cache-hit safe).
    const uint32_t base_addr = get_common_arg_val<uint32_t>(decltype(tensor::out)::addr_crta_offset / sizeof(uint32_t));
    const auto s0 = TensorAccessor(decltype(tensor::out)::args, base_addr, page_size_override);

    Noc noc;
    // Create DataflowBuffer for Device 2.0 API
    DataflowBuffer cb_out0(dfb::cb_out);

    uint32_t i_stick = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_out0.wait_front(num_read_per_barrier);
        uint32_t l1_read_addr = cb_out0.get_read_ptr();

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            // noc_async_write_sharded splits the write across shards for B/W-sharded outputs;
            // falls through to a single noc_async_write for interleaved / HEIGHT-sharded.
            tt::data_movement::common::noc_async_write_sharded(
                noc, l1_read_addr, s0, i_stick, /*offset=*/0, /*size=*/stick_size);
            l1_read_addr += stick_size_offset;
            i_stick += 1;
        }
        noc.async_write_barrier();
        cb_out0.pop_front(num_read_per_barrier);
    }
}
