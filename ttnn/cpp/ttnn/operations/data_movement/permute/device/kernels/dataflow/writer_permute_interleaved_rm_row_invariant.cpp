// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t N = get_arg(args::N);
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t num_rows = get_arg(args::num_rows);

    const uint32_t start_row = get_arg(args::start_row);
    const uint32_t end_row = get_arg(args::end_row);

    const auto s0 = TensorAccessor(tensor::output);
    Noc noc;
    DataflowBuffer dfb(dfb::cb_src);

    // shape / permutation / destination-stride arrays (rank-length; count = N, a compile-time arg).
    // Delivered as runtime varargs: input_shape occupies varargs [0, N), perm [N, 2N), dest_strides [2N, 3N).
    uint32_t input_shape[N], perm[N], dest_strides[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_vararg(i);
        perm[i] = get_vararg(i + N);
        dest_strides[i] = get_vararg(i + 2 * N);
    }

    for (uint32_t row = start_row; row < end_row; ++row) {
        // Compute multi-dimensional index for the source row
        uint32_t src_multi_idx[N];
        size_t remaining = row;
        for (uint32_t i = 0; i < N - 1; ++i) {
            size_t dim = N - 2 - i;  // Start from the second last dimension
            src_multi_idx[dim] = remaining % input_shape[dim];
            remaining /= input_shape[dim];
        }
        src_multi_idx[N - 1] = 0;  // Row dimension index

        // Apply permutation to get destination multi-dimensional index
        uint32_t dest_multi_idx[N];
        for (uint32_t i = 0; i < N; ++i) {
            dest_multi_idx[i] = src_multi_idx[perm[i]];
        }

        // Convert destination multi-dimensional index to linear index
        uint32_t dest_linear_idx = 0;
        for (uint32_t i = 0; i < N - 1; ++i) {
            dest_linear_idx += dest_multi_idx[i] * dest_strides[i];
        }
        dfb.wait_front(1);
        uint32_t l1_read_addr = dfb.get_read_ptr();
        tt::data_movement::common::noc_async_write_sharded(noc, l1_read_addr, s0, dest_linear_idx, 0, page_size);
        noc.async_write_barrier();
        dfb.pop_front(1);
    }
}
