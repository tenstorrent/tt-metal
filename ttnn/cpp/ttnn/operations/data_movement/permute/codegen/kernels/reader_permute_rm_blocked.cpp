// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads x_block_size rows along x_dim (the last output axis), each a w_block_size-element chunk of
// the input row, for the compute transpose.
// Common RT: src_addr, input_shape[N], src_strides[N]. RT: start_block, end_block.
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t N = get_named_compile_time_arg_val("N");
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    constexpr uint32_t num_rows = get_named_compile_time_arg_val("num_rows");
    constexpr uint32_t x_dim = get_named_compile_time_arg_val("x_dim");
    constexpr uint32_t x_blocks = get_named_compile_time_arg_val("x_blocks");
    constexpr uint32_t w_blocks = get_named_compile_time_arg_val("w_blocks");
    constexpr uint32_t x_block_size = get_named_compile_time_arg_val("x_block_size");
    constexpr uint32_t w_block_size = get_named_compile_time_arg_val("w_block_size");
    constexpr uint32_t element_size = get_named_compile_time_arg_val("element_size");
    constexpr uint32_t in_page_size = get_named_compile_time_arg_val("in_page_size");
    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t w_dim = N - 1;
    constexpr uint32_t h_dim = N - 2;

    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(0);
    const uint32_t end_block = get_arg_val<uint32_t>(1);

    uint32_t input_shape[N], src_strides[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_common_arg_val<uint32_t>(1 + i);
        src_strides[i] = get_common_arg_val<uint32_t>(1 + N + i);
    }

    const uint32_t X = input_shape[x_dim];
    const uint32_t X_stride = src_strides[x_dim];
    const uint32_t W = input_shape[w_dim];
    const uint32_t non_x_rows = num_rows / X;

    const auto s = TensorAccessor(src_args, src_addr, in_page_size);

    Noc noc;
    CircularBuffer cb_in(cb_id);

    uint32_t idxs[N];

    for (uint32_t block = start_block; block < end_block; ++block) {
        uint32_t rem = block;
        const uint32_t w_block = rem % w_blocks;
        rem /= w_blocks;
        const uint32_t x_block = rem % x_blocks;
        rem /= x_blocks;
        const uint32_t xw_block = rem % non_x_rows;

        const uint32_t x_start = x_block * x_block_size;
        const uint32_t x_end = std::min(x_start + x_block_size, X);

        const uint32_t w_start = w_block * w_block_size;
        const uint32_t w_end = std::min(w_start + w_block_size, W);
        const uint32_t w_offset = w_start * element_size;
        const uint32_t w_read_size_bytes = (w_end - w_start) * element_size;

        uint32_t remainder = xw_block;
        for (int32_t d = (int32_t)h_dim; d >= 0; --d) {
            if (d == (int32_t)x_dim) {
                idxs[d] = 0;
                continue;
            }
            idxs[d] = remainder % input_shape[d];
            remainder /= input_shape[d];
        }
        idxs[w_dim] = 0;

        uint32_t base_addr_offset = 0;
        for (uint32_t d = 0; d < N; ++d) {
            if (d != x_dim) {
                base_addr_offset += idxs[d] * src_strides[d];
            }
        }

        cb_in.reserve_back(x_block_size);
        uint32_t page_offset = 0;
        for (uint32_t x = x_start; x < x_end; ++x) {
            uint32_t row = base_addr_offset + x * X_stride;
            noc.async_read(
                s, cb_in, w_read_size_bytes, {.page_id = row, .offset_bytes = w_offset}, {.offset_bytes = page_offset});
            page_offset += page_size;
        }
        noc.async_read_barrier();
        cb_in.push_back(x_block_size);
    }
}
