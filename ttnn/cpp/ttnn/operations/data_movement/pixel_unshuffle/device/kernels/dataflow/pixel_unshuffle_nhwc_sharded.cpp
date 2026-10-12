// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pixel_unshuffle, channels_last: NCHW ROW_MAJOR input -> NHWC [N, Ho, Wo, Cp] ROW_MAJOR,
// HEIGHT_SHARDED in L1, written into this core's own shard.
//
// This core owns output pixels [pix0, pix0 + npix) in flattened (n, ho, wo) order; that is
// exactly its shard, so output stick j of the shard is pixel pix0 + j. The two dataflow
// RISCs run this same source; each image row touched by the range is split between them
// at a DRAM-aligned column, and each RISC reads only the r*C input half-rows its columns
// need (page = input row n*C*H + c*H + ho*r + dy, a byte span of it) into a page of its
// private staging CB. The gather into the shard - plain 32-bit L1 loads and stores, no NOC
// writes - is split between this reader (the first (24 - helper24)/24 of its span) and the compute
// UNPACK, MATH and PACK RISCs (pixel_unshuffle_nhwc_gather.cpp), which share the remaining helper24/24.
// The staging CB (2 pages, one image row each) is the only synchronisation: a page is pushed
// once its reads have landed, and a page is reused only after the helpers popped it.
//
// Reads run one image row ahead of the gather (depth 2), with plain barriers only: the
// barrier at the top of an iteration waits for exactly the reads issued in the previous
// one. No transaction ids (see pixel_unshuffle_nchw.cpp for why).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/pixel_unshuffle_nhwc_common.hpp"

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t r = get_compile_time_arg_val(3);
    constexpr uint32_t Ho = get_compile_time_arg_val(4);
    constexpr uint32_t Wo = get_compile_time_arg_val(5);
    constexpr uint32_t datum = get_compile_time_arg_val(6);
    constexpr uint32_t Cp = get_compile_time_arg_val(7);
    constexpr uint32_t out_row_nbytes = get_compile_time_arg_val(8);
    constexpr uint32_t aligned_row_nbytes_in = get_compile_time_arg_val(9);
    constexpr uint32_t cb_in = get_compile_time_arg_val(10);
    constexpr uint32_t cb_out = get_compile_time_arg_val(11);
    constexpr uint32_t channel_order = get_compile_time_arg_val(12);  // 0=CHANNEL_MAJOR, 1=SPATIAL_MAJOR
    constexpr uint32_t risc = get_compile_time_arg_val(13);           // 0: first half of each row, 1: second
    constexpr uint32_t helper24 = get_compile_time_arg_val(14);       // 24ths of each span gathered by the compute helpers together
    constexpr auto src_args = TensorAccessorArgs<15, 0>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t pix0 = get_arg_val<uint32_t>(1);
    const uint32_t npix = get_arg_val<uint32_t>(2);
    if (npix == 0) {
        return;
    }

    using G = pixel_unshuffle_nhwc::Geom<W, C, H, r, Ho, Wo, datum, Cp, out_row_nbytes, aligned_row_nbytes_in, channel_order>;
    using pixel_unshuffle_nhwc::Item;
    constexpr uint32_t own_hi = 24 - helper24;

    const auto s_in = TensorAccessor(src_args, src_addr);
    const uint32_t out_base = get_write_ptr(cb_out);
    const uint32_t end = pix0 + npix;

    auto issue_reads = [&](const Item& it, uint32_t dst) {
        const uint32_t nbytes = (it.cb_al - it.ca_al) * datum;
        const uint32_t src_off = it.ca_al * datum;
        const uint32_t page_base = it.n * G::CH + it.ho * r;  // + c*H + dy
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t dy = 0; dy < r; dy++) {
                noc_async_read(s_in.get_noc_addr(page_base + c * H + dy, src_off), dst, nbytes);
                dst += aligned_row_nbytes_in;
            }
        }
    };

    uint32_t cursor = pix0;
    Item cur;
    Item nxt;
    if (!G::advance(risc, pix0, end, cursor, cur)) {
        return;
    }
    cb_reserve_back(cb_in, 1);
    uint32_t page = get_write_ptr(cb_in);
    issue_reads(cur, page);
    while (true) {
        const bool has_next = G::advance(risc, pix0, end, cursor, nxt);
        noc_async_read_barrier();  // only cur's reads are outstanding
        asm volatile("" ::: "memory");
        cb_push_back(cb_in, 1);  // the helpers may start on cur
        uint32_t next_page = 0;
        if (has_next) {
            cb_reserve_back(cb_in, 1);  // waits until the helpers are done with the row before last
            next_page = get_write_ptr(cb_in);
            issue_reads(nxt, next_page);  // overlaps the gather below
        }
        uint32_t a;
        uint32_t b;
        G::part(cur, 0, own_hi, a, b);
        G::gather(cur, a, b, page, out_base);
        if (!has_next) {
            break;
        }
        cur = nxt;
        page = next_page;
    }
}
