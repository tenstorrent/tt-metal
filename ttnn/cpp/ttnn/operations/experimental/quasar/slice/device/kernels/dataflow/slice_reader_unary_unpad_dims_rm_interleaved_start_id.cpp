// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_dims = get_arg(args::num_dims);

    // Common (shared) runtime args.
    // addr_offset = begins_bytes - misalignment: a constant byte offset into the input buffer
    // (folded into the program hash). The buffer base address is read from the auto-patched
    // tensor binding so the read survives buffer reallocation on a program-cache hit.
    const uint32_t addr_offset = get_arg(args::addr_offset);
    const uint32_t padded_stick_size = get_arg(args::padded_stick_size);
    const uint32_t unpadded_stick_size = get_arg(args::unpadded_stick_size);
    const uint32_t stick_size_offset = get_arg(args::stick_size_offset);
    const uint32_t misalignment = get_arg(args::misalignment);

    // Per-core runtime args.
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const uint32_t num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    const uint32_t num_read_per_barrier = get_arg(args::num_read_per_barrier);

    // num_unpadded_sticks / num_padded_sticks are per-dim arrays read in the inner loop by a
    // runtime-varying index, so they arrive as common runtime varargs: [0, num_dims) is
    // num_unpadded_sticks and [num_dims, 2*num_dims) is num_padded_sticks.
    uint32_t num_unpadded_sticks[num_dims];
    uint32_t num_padded_sticks[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        num_unpadded_sticks[j] = get_common_vararg(j);
        num_padded_sticks[j] = get_common_vararg(num_dims + j);
    }

    // id_per_dim is a per-core array advanced in the inner loop by a runtime-varying index → runtime varargs.
    uint32_t id_per_dim[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(j);
    }

    uint32_t read_size = unpadded_stick_size + misalignment;

    // padded_stick_size = per-shard page size (shard_W on B/W-sharded, full row otherwise);
    // feeds `noc_async_read_sharded`'s multi-shard split via `get_aligned_page_size()`.
    // Override the binding's address (aligned-down: base + addr_offset) and page size (per-shard
    // width). The base address is the auto-patched binding address (cache-hit safe).
    const uint32_t base_addr = get_common_arg_val<uint32_t>(decltype(tensor::in)::addr_crta_offset / sizeof(uint32_t));
    const auto s0 = TensorAccessor(decltype(tensor::in)::args, base_addr + addr_offset, padded_stick_size);

    // Create objects for Device 2.0 API
    DataflowBuffer cb_in0(dfb::cb_in);
    Noc noc;

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_in0.reserve_back(num_read_per_barrier);
        uint32_t src_buffer_l1_addr = cb_in0.get_write_ptr();

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            // noc_async_read_sharded splits the read across shards for B/W-sharded inputs;
            // falls through to a single noc_async_read for interleaved / HEIGHT-sharded.
            tt::data_movement::common::noc_async_read_sharded(
                noc, src_buffer_l1_addr, s0, src_stick_id, /*offset=*/0, /*size=*/read_size);
            if (misalignment != 0) {
                noc.async_read_barrier();
                tt::data_movement::common::tt_memmove<false, false, false, 0>(
                    noc, src_buffer_l1_addr, src_buffer_l1_addr + misalignment, unpadded_stick_size);
            }
            src_buffer_l1_addr += stick_size_offset;
            src_stick_id++;
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks[j];
                } else {
                    break;
                }
            }
        }
        noc.async_read_barrier();
        cb_in0.push_back(num_read_per_barrier);
    }
}
