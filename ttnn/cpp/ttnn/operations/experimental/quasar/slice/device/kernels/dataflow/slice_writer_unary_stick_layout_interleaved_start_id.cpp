// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Named runtime args (formerly writer slots [1..7]; slot [0] output buffer address and
    // slot [7] per-shard page-size override are dropped under the spec ABI — the output buffer
    // is bound via the OUTPUT TensorParameter and the per-shard page size is derived from its
    // TensorSpec).
    const uint32_t stick_size = get_arg(args::unpadded_row_size_bytes);
    const uint32_t stick_size_offset = get_arg(args::unpadded_row_size_bytes_offset);
    const uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const uint32_t num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    const uint32_t num_read_per_barrier = get_arg(args::num_read_per_barrier);
    const uint32_t start_id = get_arg(args::start_id);

    // Output buffer is bound via the OUTPUT TensorParameter; the per-shard page size used by
    // noc_async_write_sharded's multi-shard split is derived from the tensor's TensorSpec.
    const auto s0 = TensorAccessor(tensor::out);

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
