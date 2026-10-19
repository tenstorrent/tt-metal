// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scatters the transposed [w_block_size x x_block_size] blocks from the compute kernel to their output
// rows. Compute swapped x_dim and W, so the shape and permutation are swapped on them before addressing.
// Common RT: dst_addr, input_shape[N], perm[N], dest_strides[N]. RT: start_block, end_block.
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t N = get_named_compile_time_arg_val("N");
    constexpr uint32_t output_page_size = get_named_compile_time_arg_val("output_page_size");
    constexpr uint32_t x_dim = get_named_compile_time_arg_val("x_dim");
    constexpr uint32_t x_blocks = get_named_compile_time_arg_val("x_blocks");
    constexpr uint32_t w_blocks = get_named_compile_time_arg_val("w_blocks");
    constexpr uint32_t x_block_size = get_named_compile_time_arg_val("x_block_size");
    constexpr uint32_t w_block_size = get_named_compile_time_arg_val("w_block_size");
    constexpr uint32_t X = get_named_compile_time_arg_val("X");
    constexpr uint32_t W = get_named_compile_time_arg_val("W");
    constexpr uint32_t num_rows = get_named_compile_time_arg_val("num_rows");
    constexpr uint32_t element_size = get_named_compile_time_arg_val("element_size");
    constexpr uint32_t out_page_size = get_named_compile_time_arg_val("out_page_size");
    constexpr auto dst_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_id = 2;
    constexpr uint32_t w_dim = N - 1;
    constexpr uint32_t h_dim = N - 2;
    constexpr uint32_t non_x_rows = num_rows / X;

    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(0);
    const uint32_t end_block = get_arg_val<uint32_t>(1);

    uint32_t input_shape[N], perm[N], dest_strides[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_common_arg_val<uint32_t>(1 + i);
        perm[i] = get_common_arg_val<uint32_t>(1 + N + i);
        dest_strides[i] = get_common_arg_val<uint32_t>(1 + 2 * N + i);
    }

    {
        uint32_t t = input_shape[x_dim];
        input_shape[x_dim] = input_shape[w_dim];
        input_shape[w_dim] = t;
    }
    for (uint32_t i = 0; i < N; i++) {
        if (perm[i] == x_dim) {
            perm[i] = w_dim;
        } else if (perm[i] == w_dim) {
            perm[i] = x_dim;
        }
    }

    uint32_t x_dim_in_dest = N;
    for (uint32_t i = 0; i < N; ++i) {
        if (perm[i] == x_dim) {
            x_dim_in_dest = i;
            break;
        }
    }

    const auto s = TensorAccessor(dst_args, dst_addr, out_page_size);

    Noc noc;
    CircularBuffer cb_out(cb_id);

    uint32_t src_multi_idx[N] = {0};
    uint32_t dest_multi_idx[N] = {0};

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

        const uint32_t x_read_size_bytes = (x_end - x_start) * element_size;
        const uint32_t x_offset = x_start * element_size;

        uint32_t remainder = xw_block;
        for (int32_t d = (int32_t)h_dim; d >= 0; --d) {
            if (d == (int32_t)x_dim) {
                continue;
            }
            src_multi_idx[d] = remainder % input_shape[d];
            remainder /= input_shape[d];
        }

        uint32_t dest_linear_idx_base = 0;
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t src_idx = perm[i];
            if (src_idx != x_dim) {
                dest_multi_idx[i] = src_multi_idx[src_idx];
                if (i < w_dim) {
                    dest_linear_idx_base += dest_multi_idx[i] * dest_strides[i];
                }
            }
        }

        cb_out.wait_front(w_block_size);

        for (uint32_t w = w_start; w < w_end; ++w) {
            uint32_t dest_linear_idx = dest_linear_idx_base;
            if (x_dim_in_dest < w_dim) {
                dest_linear_idx += w * dest_strides[x_dim_in_dest];
            }
            uint32_t l1_offset = (w - w_start) * output_page_size;
            noc.async_write(
                cb_out,
                s,
                x_read_size_bytes,
                {.offset_bytes = l1_offset},
                {.page_id = dest_linear_idx, .offset_bytes = x_offset});
        }
        noc.async_writes_flushed();
        cb_out.pop_front(w_block_size);
    }
    noc.async_write_barrier();
}
