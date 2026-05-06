// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer for YUV conversion (multicore).
//
// Each cb_out page = one uint8 T-chunk (32 bytes for full, partial_elems for last).
// Pages arrive in order: Y pass, then Cb, then Cr — for this core's slice only.
//
// Compile-time args:
//   [0] cb_out
//   [1] num_full_chunks, [2] has_partial, [3] full_chunk_elems, [4] partial_elems
//   [5..] TensorAccessorArgs for Y, U, V buffers
//
// Runtime args:
//   [0] y_addr, [1] u_addr, [2] v_addr
//   [3] y_start, [4] y_count    — this core's Y spatial slice
//   [5] uv_start, [6] uv_count  — this core's UV spatial slice

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t y_start = get_arg_val<uint32_t>(3);
    const uint32_t y_count = get_arg_val<uint32_t>(4);
    const uint32_t uv_start = get_arg_val<uint32_t>(5);
    const uint32_t uv_count = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t has_partial = get_compile_time_arg_val(2);
    constexpr uint32_t full_chunk_elems = get_compile_time_arg_val(3);  // 32
    constexpr uint32_t partial_elems = get_compile_time_arg_val(4);
    constexpr auto y_args = TensorAccessorArgs<5>();
    constexpr auto u_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();

    constexpr uint32_t num_chunks = num_full_chunks + has_partial;

    const auto sy = TensorAccessor(y_args, y_addr);
    const auto su = TensorAccessor(u_args, u_addr);
    const auto sv = TensorAccessor(v_args, v_addr);

    auto write_plane = [&](const auto& dst, uint32_t start, uint32_t count) {
        for (uint32_t spatial = start; spatial < start + count; spatial++) {
            for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
                const bool is_partial = has_partial && (chunk == num_full_chunks);
                const uint32_t write_elems = is_partial ? partial_elems : full_chunk_elems;
                const uint32_t byte_off = chunk * full_chunk_elems;

                cb_wait_front(cb_out, 1);
                uint32_t l1_src = get_read_ptr(cb_out);

                noc_async_write(l1_src, dst.get_noc_addr(spatial, byte_off), write_elems);
                noc_async_writes_flushed();
                cb_pop_front(cb_out, 1);
            }
        }
    };

    write_plane(sy, y_start, y_count);
    write_plane(su, uv_start, uv_count);
    write_plane(sv, uv_start, uv_count);

    noc_async_write_barrier();
}
