// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
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
    uint32_t curr_c = get_arg(args::curr_c);
    uint32_t curr_h = get_arg(args::curr_h);
    uint32_t curr_n = get_arg(args::curr_n);

    constexpr uint32_t N = get_arg(args::N);
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t C = get_arg(args::C);
    constexpr uint32_t W_size_bytes = get_arg(args::W_size_bytes);

    constexpr uint32_t CH = C * H;

    const uint32_t stick_size_bytes = W_size_bytes;

    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer cb(dfb::cb_in0);

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        cb.reserve_back(num_read_per_barrier);
        const uint32_t cb_write_ptr = cb.get_write_ptr();
        uint32_t l1_write_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            // Restored native sharded multi-page split (see common.hpp helper).
            tt::data_movement::common::noc_async_read_sharded(
                noc, cb_write_ptr + l1_write_offset, s, i_stick, 0, stick_size_bytes);
            l1_write_offset += stick_size_bytes;

            curr_c++;
            i_stick += H;
            if (curr_c == C) {  // end of channel dim
                curr_h++;
                curr_c = 0;
                if (curr_h == H) {  // end of H dim
                    curr_n++;
                    curr_c = 0;
                    curr_h = 0;
                    i_stick = i_stick - H + 1;
                } else {
                    i_stick = i_stick - CH + 1;
                }
            }
        }
        noc.async_read_barrier();
        cb.push_back(num_read_per_barrier);
    }
}
