// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.block_max_bf16(strip_major=True) (block_select.py): one data-movement kernel. in [S NB / 8, 256] bf16
// ROW_MAJOR, strip-major in 512-byte pages of 8 blocks (the indexer score's block_max_compute output: query strip s,
// block b's 32 row values one 64-byte run at byte (s NB + b) 64), out [S * 32, NB] bf16 ROW_MAJOR:
//   out[32 s + i, b] = in(s, b)[i]                                          for b < n (the limit)
// (a transpose of each strip's [NB, 32] to [32, NB]; exact copies). n (blocks) is a runtime value: a common arg (value
// slot) or word 0 of a [1] uint32 tensor (HAS_LEN); rounded up to 8 (16-byte writes) and capped at NB. Blocks from
// there on are not written. Work units (strip, piece of CHB blocks) split evenly over both RISCs of NCORES cores at
// run time (no per-core args); double-buffered reads.
// compile-time args: cb, NB, GX, NCORES, CHB, HAS_LEN, HALF, in / out / len accessors; common args: in, out, len
// addresses, n, strips.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

/** m blocks (even) of a strip, [m][32] bf16 at iv -> [32][m] bf16 at ov (row pitch m values). Two words in (rows 2j,
 *  2j + 1 of blocks b, b + 1), two words out. */
inline void transpose_piece(const uint32_t* iv, uint32_t* ov, uint32_t m) {
    const uint32_t pitch = m / 2;  // output row pitch in words
    for (uint32_t b = 0; b < m; b += 2) {
        const uint32_t* w0p = iv + b * 16;
        const uint32_t* w1p = w0p + 16;
        uint32_t* o = ov + b / 2;
#pragma GCC unroll 16
        for (uint32_t j = 0; j < 16; ++j) {
            const uint32_t w0 = w0p[j], w1 = w1p[j];
            o[(2 * j) * pitch] = (w0 & 0xFFFFu) | (w1 << 16);
            o[(2 * j + 1) * pitch] = (w0 >> 16) | (w1 & 0xFFFF0000u);
        }
    }
}

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr uint32_t GX = get_compile_time_arg_val(2);
    constexpr uint32_t NCORES = get_compile_time_arg_val(3);
    constexpr uint32_t CHB = get_compile_time_arg_val(4);
    constexpr bool HAS_LEN = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t HALF = get_compile_time_arg_val(6);  // this RISC's half of the core's units (0 / 1)
    constexpr auto i_args = TensorAccessorArgs<7>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto l_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    static_assert(NB % 8 == 0, "NB: whole 16-byte output runs");
    static_assert(CHB % 8 == 0, "CHB: whole 16-byte output runs");
    constexpr uint32_t RUN = 64;         // a block's 32 bf16 row values
    constexpr uint32_t BUF = CHB * RUN;  // one piece in (== one piece out)
    const auto in = TensorAccessor(i_args, get_common_arg_val<uint32_t>(0));
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1));
    const uint32_t strips = get_common_arg_val<uint32_t>(4);
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();

    // this RISC's scratch: two input + two output buffers, 64-byte aligned
    cb_reserve_back(cb, 1);
    const uint32_t base = ((get_write_ptr(cb) + 63u) & ~63u) + HALF * 4u * BUF;
    const uint32_t in_l1[2] = {base, base + BUF};
    const uint32_t out_l1[2] = {base + 2u * BUF, base + 3u * BUF};
    uint32_t n = get_common_arg_val<uint32_t>(3);
    if constexpr (HAS_LEN) {
        const auto len = TensorAccessor(l_args, get_common_arg_val<uint32_t>(2));
        noc_async_read(len.get_noc_addr(0), out_l1[0], 16);
        noc_async_read_barrier();
        n = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1[0]);
    }
    n = (n + 7u) & ~7u;
    n = n < NB ? n : NB;
    const uint32_t pieces = (n + CHB - 1) / CHB;
    const uint32_t units = strips * pieces;
    // 2 NCORES workers (each core's two RISCs), worker w = 2 c + HALF
    const uint32_t nw = 2 * NCORES, w = 2 * c + HALF;
    const uint32_t per = units / nw, extra = units % nw;
    const uint32_t count = per + (w < extra ? 1u : 0u);
    const uint32_t first = w * per + (w < extra ? w : extra);
    auto issue = [&](uint32_t u, uint32_t slot) {
        const uint32_t s = u / pieces, b0 = (u % pieces) * CHB;
        const uint32_t m = (n - b0) < CHB ? (n - b0) : CHB;
        // input pages of 8 blocks (512 B): strip s's blocks [b0, b0 + m) are pages s NB / 8 + b0 / 8 + [0, m / 8)
        const uint32_t p0 = s * (NB / 8) + b0 / 8;
        for (uint32_t k = 0; k < m / 8; ++k) {
            noc_async_read(in.get_noc_addr(p0 + k), in_l1[slot] + k * 8 * RUN, 8 * RUN);
        }
    };
    if (count) {
        issue(first, 0);
    }
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t u = first + i, slot = i & 1u;
        noc_async_read_barrier();
        if (i + 1 < count) {
            noc_async_writes_flushed();  // (the other slot's output may still be leaving)
            issue(u + 1, slot ^ 1u);
        }
        const uint32_t s = u / pieces, b0 = (u % pieces) * CHB;
        const uint32_t m = (n - b0) < CHB ? (n - b0) : CHB;
        transpose_piece(reinterpret_cast<const uint32_t*>(in_l1[slot]), reinterpret_cast<uint32_t*>(out_l1[slot]), m);
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_write(out_l1[slot] + r * m * 2u, out.get_noc_addr(32 * s + r, b0 * 2u), m * 2u);
        }
    }
    noc_async_write_barrier();
}
