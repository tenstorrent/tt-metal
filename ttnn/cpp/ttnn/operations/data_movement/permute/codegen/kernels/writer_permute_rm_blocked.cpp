// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the single-pass blocked-generic RM permute (W-changing class).
//
// Consumes the compute kernel's transposed [w_block_size x x_block_size] blocks
// from cb_2 and scatters each of the w_block_size output rows (x_read_size bytes)
// to its permuted output page. The compute transposed input axes x_dim <-> W, so
// perm is swapped on those two axes here before addressing.
//
// Named CT: N, output_page_size(=x_block_size*elem), num_rows, X, x_dim,
//           x_blocks, w_blocks, x_block_size, w_block_size, W, element_size,
//           out_page_size(=aligned output row bytes)
// Positional CT: TensorAccessorArgs (index 0)
// RT: [dst_addr, start_block, end_block, input_shape[N], perm[N], dest_strides[N]]
#include <algorithm>
#include "api/dataflow/dataflow_api.h"

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

    constexpr uint32_t cb_id = 2;  // compute output CB (c_2)
    constexpr uint32_t w_dim = N - 1;
    constexpr uint32_t non_x_rows = num_rows / X;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t end_block = get_arg_val<uint32_t>(2);

    uint32_t input_shape[N], perm[N], dest_strides[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_arg_val<uint32_t>(3 + i);
        perm[i] = get_arg_val<uint32_t>(3 + N + i);
        dest_strides[i] = get_arg_val<uint32_t>(3 + 2 * N + i);
    }

    // The compute kernel transposed x_dim <-> W. Reflect that in perm.
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
        for (int32_t d = (int32_t)N - 2; d >= 0; --d) {
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

        cb_wait_front(cb_id, w_block_size);
        uint32_t read_addr = get_read_ptr(cb_id);

        for (uint32_t w = w_start; w < w_end; ++w) {
            uint32_t dest_linear_idx = dest_linear_idx_base;
            if (x_dim_in_dest < w_dim) {
                dest_linear_idx += w * dest_strides[x_dim_in_dest];
            }
            uint32_t l1_addr = read_addr + (w - w_start) * output_page_size;
            uint64_t noc_addr = get_noc_addr(dest_linear_idx, s) + x_offset;
            noc_async_write(l1_addr, noc_addr, x_read_size_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id, w_block_size);
    }
}
