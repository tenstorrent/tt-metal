// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// tt/page_copy.GluedIdsReorder's kernel, verbatim (NC.7).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t W = get_compile_time_arg_val(0);     // ids a mesh row (chunk / SP)
constexpr uint32_t SP = get_compile_time_arg_val(1);    // mesh rows
constexpr uint32_t VMAX = get_compile_time_arg_val(2);  // vocab - 1 (the clamp and the pad id)
constexpr uint32_t HDR = get_compile_time_arg_val(3);   // header words after each mesh row's ids
constexpr uint32_t BLOCK =
    get_compile_time_arg_val(4);                       // decode's kv block (PREFILL_END=cap: cap = end / BLOCK x BLOCK)
constexpr uint32_t CAP = get_compile_time_arg_val(5);  // 1: PREFILL_END=cap, 0: end (every row computed)
constexpr uint32_t ALIGN = get_compile_time_arg_val(6);  // PB.9G: rows a chip of chip-aligned lanes (0: back to back)
constexpr uint32_t NSEG = 6;
constexpr uint32_t PAGE = (W + HDR) * 4;  // bytes a mesh row of the gathered input
constexpr uint32_t NR = ALIGN ? 1 : SP;   // (PB.9G: engine row = model row: this mesh row's own input page only)
constexpr auto s_args = TensorAccessorArgs<7>();
constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0);  // [SP, 1, W + HDR] the engine input of every mesh row (gathered)
    const uint32_t dst = get_arg_val<uint32_t>(1);  // [1, W] this chip's model-row ids
    const uint32_t me = get_arg_val<uint32_t>(2);   // this mesh row (frozen)
    const auto S = TensorAccessor(s_args, src, PAGE);
    const auto D = TensorAccessor(d_args, dst, W * 4);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;  // SP pages, then W output words
    for (uint32_t r = 0; r < NR; ++r) {
        noc_async_read(S.get_noc_addr(r, 0), buf + r * PAGE, PAGE);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    volatile tt_l1_ptr uint32_t* h = in + W;  // mesh row 0's header: n, then (slot, start, end, gws) a segment
    uint32_t n = h[0];
    if (n > NSEG) {
        n = NSEG;
    }
    uint32_t in_off[NSEG], lane_off[NSEG], rows[NSEG];
    uint32_t ci = 0, cl = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t start = h[1 + 4 * k + 1], end = h[1 + 4 * k + 2];
        const uint32_t run_end = CAP ? (end / BLOCK) * BLOCK : end;
        in_off[k] = ci;
        lane_off[k] = cl;
        rows[k] = run_end > start ? run_end - start : 0;
        const uint32_t len = end > start ? end - start : 1;
        constexpr uint32_t AL = ALIGN ? ALIGN : 1;  // (no division by a zero constant)
        const uint32_t ext = ALIGN ? (len + AL - 1) / AL * AL : 0;
        ci += ALIGN ? ext : end - start;
        cl += ALIGN ? ext : rows[k];
    }
    volatile tt_l1_ptr uint32_t* out = in + NR * (W + HDR);
    uint32_t k = 0;
    for (uint32_t i = 0; i < W; ++i) {
        const uint32_t j = me * W + i;  // the model row
        while (k < n && j >= lane_off[k] + rows[k]) {
            ++k;
        }
        uint32_t x = VMAX;
        if (k < n && j >= lane_off[k]) {
            const uint32_t e = in_off[k] + (j - lane_off[k]);  // its engine row
            x = ALIGN ? in[e - me * W] : in[(e / W) * (W + HDR) + (e % W)];
            if (x > VMAX) {
                x = VMAX;
            }
        }
        out[i] = x;
    }
    noc_async_write(reinterpret_cast<uint32_t>(out), D.get_noc_addr(0), W * 4);
    noc_async_write_barrier();
}
