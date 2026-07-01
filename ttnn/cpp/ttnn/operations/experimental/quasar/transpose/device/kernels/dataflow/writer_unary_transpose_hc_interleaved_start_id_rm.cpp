// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    uint32_t num_read_per_barrier = get_arg(args::num_read_per_barrier);
    uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t W_size_bytes = get_arg(args::W_size_bytes);

    const uint32_t stick_size_bytes = W_size_bytes;

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer cb(dfb::cb_out0);

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        cb.wait_front(num_read_per_barrier);
        const uint32_t cb_read_ptr = cb.get_read_ptr();
        uint32_t l1_read_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            // Restored native sharded multi-page split (see common.hpp helper).
            tt::data_movement::common::noc_async_write_sharded(
                noc, cb_read_ptr + l1_read_offset, s, i_stick, 0, stick_size_bytes);
            l1_read_offset += stick_size_bytes;
            i_stick += 1;
        }
        noc.async_write_barrier();
        cb.pop_front(num_read_per_barrier);
    }
}
