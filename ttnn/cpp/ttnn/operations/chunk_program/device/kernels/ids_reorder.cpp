// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// tt/page_copy.IdsReorder's kernel, verbatim (NC.7).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t WU = get_compile_time_arg_val(0);    // 32-id units a mesh row (W / 32)
constexpr uint32_t SP = get_compile_time_arg_val(1);    // mesh rows
constexpr uint32_t VMAX = get_compile_time_arg_val(2);  // vocab - 1 (the clamp and the pad id)
constexpr uint32_t UB = 128;                            // bytes a unit (32 uint32)
constexpr auto s_args = TensorAccessorArgs<3>();
constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0);    // [SP, 1, W] the server's order (every mesh row's), a page a row
    const uint32_t dst = get_arg_val<uint32_t>(1);    // [1, W] this chip's natural-order ids
    const uint32_t iu = get_arg_val<uint32_t>(2);     // (start mod W) / 32 (per-run value)
    const uint32_t cs = get_arg_val<uint32_t>(3);     // (start / W) mod SP: the mesh row holding the chunk's start
    const uint32_t valid = get_arg_val<uint32_t>(4);  // natural positions below it are ids, the rest pad
    const uint32_t me = get_arg_val<uint32_t>(5);     // this mesh row (frozen)
    const auto S = TensorAccessor(s_args, src, WU * UB);
    const auto D = TensorAccessor(d_args, dst, WU * UB);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t SU = WU * SP;  // units a chunk
    for (uint32_t u = 0; u < WU; ++u) {
        const uint32_t d = me * WU + u;  // the natural unit
        uint32_t row, j;
        if (d + iu < WU) {  // the start row's head (tt/attention.WindowGeometry.server_rows, k == 0)
            row = cs;
            j = d;
        } else if (d + iu >= SU) {  // the start row's tail: the chunk's last iu units
            row = cs;
            j = d + WU - SU;
        } else {
            const uint32_t e = d + iu;
            row = (e / WU + cs) % SP;
            j = e % WU;
        }
        noc_async_read(S.get_noc_addr(row, j * UB), buf + u * UB, UB);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    const uint32_t p0 = me * WU * 32;
    for (uint32_t i = 0; i < WU * 32; ++i) {
        const uint32_t x = v[i];
        v[i] = (p0 + i >= valid || x > VMAX) ? VMAX : x;
    }
    noc_async_write(buf, D.get_noc_addr(0), WU * UB);
    noc_async_write_barrier();
}
