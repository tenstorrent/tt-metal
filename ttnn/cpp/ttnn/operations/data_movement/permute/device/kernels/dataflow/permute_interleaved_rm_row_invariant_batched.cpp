// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t N = get_arg(args::N);
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t batch = get_arg(args::batch);

    const uint32_t start_row = get_arg(args::start_row);
    const uint32_t end_row = get_arg(args::end_row);

    const auto s_in = TensorAccessor(tensor::input);
    const auto s_out = TensorAccessor(tensor::output);
    Scratchpad<uint8_t> scratch_pad(scratch::scratch);
    Noc noc;

    uint32_t input_shape[N], perm[N], dest_strides[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_vararg(i);
        perm[i] = get_vararg(i + N);
        dest_strides[i] = get_vararg(i + 2 * N);
    }
    uint32_t dest_step[N];
    for (uint32_t i = 0; i < N; ++i) {
        dest_step[i] = 0;
    }
    for (uint32_t i = 0; i < N - 1; ++i) {
        dest_step[perm[i]] = dest_strides[i];
    }
    uint32_t idx[N];
    uint32_t dest = 0;
    uint32_t remaining = start_row;
    for (uint32_t i = 0; i < N - 1; ++i) {
        const uint32_t d = N - 2 - i;
        idx[d] = remaining % input_shape[d];
        remaining /= input_shape[d];
        dest += idx[d] * dest_step[d];
    }
    idx[N - 1] = 0;

    const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&scratch_pad[0]));
    for (uint32_t row = start_row; row < end_row;) {
        const uint32_t left = end_row - row;
        const uint32_t n = left < batch ? left : batch;
        uint32_t addr = base;
        for (uint32_t i = 0; i < n; ++i) {
            tt::data_movement::common::noc_async_read_sharded(noc, addr, s_in, row + i, 0, page_size);
            addr += page_size;
        }
        noc.async_read_barrier();
        addr = base;
        for (uint32_t i = 0; i < n; ++i) {
            tt::data_movement::common::noc_async_write_sharded(noc, addr, s_out, dest, 0, page_size);
            addr += page_size;
            int32_t d = static_cast<int32_t>(N) - 2;
            idx[d]++;
            dest += dest_step[d];
            while (d > 0 && idx[d] == input_shape[d]) {
                dest -= input_shape[d] * dest_step[d];
                idx[d] = 0;
                --d;
                idx[d]++;
                dest += dest_step[d];
            }
        }
        noc.async_writes_flushed();
        row += n;
    }
    noc.async_write_barrier();
}
