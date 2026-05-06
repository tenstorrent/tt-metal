// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer for YUV conversion (degenerate-tile approach).
//
// Each cb_out page = one uint8 T-chunk (32 bytes for full, partial_elems for last).
// Pages arrive in order: Y pass (H×W × num_chunks), then Cb, then Cr.
//
// Compile-time args:
//   [0] cb_out
//   [1] num_full_chunks, [2] has_partial, [3] full_chunk_elems, [4] partial_elems
//   [5] H, [6] W, [7] T, [8] H2, [9] W2
//   [10..] TensorAccessorArgs for Y, U, V buffers
// Runtime args: [0] y_addr, [1] u_addr, [2] v_addr

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t has_partial = get_compile_time_arg_val(2);
    constexpr uint32_t full_chunk_elems = get_compile_time_arg_val(3);  // 32
    constexpr uint32_t partial_elems = get_compile_time_arg_val(4);
    constexpr uint32_t H = get_compile_time_arg_val(5);
    constexpr uint32_t W = get_compile_time_arg_val(6);
    constexpr uint32_t T = get_compile_time_arg_val(7);
    constexpr uint32_t H2 = get_compile_time_arg_val(8);
    constexpr uint32_t W2 = get_compile_time_arg_val(9);
    constexpr auto y_args = TensorAccessorArgs<10>();
    constexpr auto u_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();

    constexpr uint32_t HW = H * W;
    constexpr uint32_t HW2 = H2 * W2;
    constexpr uint32_t num_chunks = num_full_chunks + has_partial;

    const auto sy = TensorAccessor(y_args, y_addr);
    const auto su = TensorAccessor(u_args, u_addr);
    const auto sv = TensorAccessor(v_args, v_addr);

    // Write all chunks for one spatial plane to the given TensorAccessor.
    auto write_plane = [&](const auto& dst, uint32_t spatial_total) {
        for (uint32_t spatial = 0; spatial < spatial_total; spatial++) {
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

    write_plane(sy, HW);   // Y: H×W sticks
    write_plane(su, HW2);  // U (Cb): H/2×W/2 sticks
    write_plane(sv, HW2);  // V (Cr)

    noc_async_write_barrier();
}
