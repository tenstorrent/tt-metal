// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.block_max_bf16 (block_select.py): one data-movement kernel. in [rows, NI] fp32 ROW_MAJOR (bf16 values
// widened, e.g. the indexer score's block_max_out), out [rows, NI / F] bf16 ROW_MAJOR:
//   out[r, b] = max(in[r, b F .. b F + F)) as bf16 (exact: every input is a bf16 value)   for b < n (the limit)
// n (output columns) is a runtime value: a common arg (value slot) or word 0 of a [1] uint32 tensor (HAS_LEN); it is
// rounded up to a multiple of 8 (16-byte writes) and capped at NI / F. Columns from there on are not written. Work
// units (row, piece of CH input values) are split evenly over NCORES cores at run time (no per-core args).
// compile-time args: cb, F, NI, GX, NCORES, CH, HAS_LEN, in / out (/ len) accessors; common args: in, out, len
// addresses, n, rows.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

template <uint32_t F>
inline void reduce_piece(const int32_t* iv, uint16_t* ov, uint32_t nout) {
    // fp32 bits as int32: the largest float is the signed max when it is >= 0, else the signed min
    for (uint32_t b = 0; b < nout; ++b) {
        int32_t mx = iv[b * F], mn = mx;
#pragma GCC unroll 16
        for (uint32_t e = 1; e < F; ++e) {
            const int32_t v = iv[b * F + e];
            mx = mx > v ? mx : v;
            mn = mn < v ? mn : v;
        }
        ov[b] = static_cast<uint16_t>(static_cast<uint32_t>(mx >= 0 ? mx : mn) >> 16);
    }
}

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t F = get_compile_time_arg_val(1);
    constexpr uint32_t NI = get_compile_time_arg_val(2);
    constexpr uint32_t GX = get_compile_time_arg_val(3);
    constexpr uint32_t NCORES = get_compile_time_arg_val(4);
    constexpr uint32_t CH = get_compile_time_arg_val(5);
    constexpr bool HAS_LEN = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t HALF = get_compile_time_arg_val(7);  // this RISC's half of the core's units (0 / 1)
    constexpr auto i_args = TensorAccessorArgs<8>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto l_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr uint32_t NO = NI / F;
    static_assert(F >= 1 && F <= 16 && (F & (F - 1)) == 0, "F: a power of two <= 16");
    static_assert(NI % (8 * F) == 0, "NI: whole 16-byte output runs");
    static_assert(CH % (8 * F) == 0 && CH % 16 == 0, "CH: whole output runs, 64-byte aligned reads");
    const auto in = TensorAccessor(i_args, get_common_arg_val<uint32_t>(0));
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1));
    const uint32_t rows = get_common_arg_val<uint32_t>(4);
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();

    // this RISC's scratch: two input buffers + two output buffers (CH values each), 64-byte aligned
    cb_reserve_back(cb, 1);
    constexpr uint32_t SPAN = CH * 4u + CH / F * 2u + 64u;
    const uint32_t base = ((get_write_ptr(cb) + 63u) & ~63u) + HALF * 2u * ((SPAN + 63u) & ~63u);
    const uint32_t in_l1[2] = {base, base + ((SPAN + 63u) & ~63u)};
    const uint32_t out_l1[2] = {in_l1[0] + CH * 4u, in_l1[1] + CH * 4u};
    uint32_t n = get_common_arg_val<uint32_t>(3);
    if constexpr (HAS_LEN) {
        const auto len = TensorAccessor(l_args, get_common_arg_val<uint32_t>(2));
        noc_async_read(len.get_noc_addr(0), out_l1[0], 16);
        noc_async_read_barrier();
        n = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1[0]);
    }
    n = (n + 7u) & ~7u;
    n = n < NO ? n : NO;
    const uint32_t n_in = n * F;
    const uint32_t pieces = (n_in + CH - 1) / CH;
    const uint32_t units = rows * pieces;
    // 2 NCORES workers (each core's two RISCs), worker w = 2 c + HALF
    const uint32_t nw = 2 * NCORES, w = 2 * c + HALF;
    const uint32_t per = units / nw, extra = units % nw;
    const uint32_t count = per + (w < extra ? 1u : 0u);
    const uint32_t first = w * per + (w < extra ? w : extra);
    auto issue = [&](uint32_t u, uint32_t slot) {
        const uint32_t r = u / pieces, c0 = (u % pieces) * CH;
        const uint32_t m = (n_in - c0) < CH ? (n_in - c0) : CH;
        noc_async_read(in.get_noc_addr(r, c0 * 4u), in_l1[slot], m * 4u);
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
        const uint32_t r = u / pieces, c0 = (u % pieces) * CH;
        const uint32_t m = (n_in - c0) < CH ? (n_in - c0) : CH;
        reduce_piece<F>(
            reinterpret_cast<const int32_t*>(in_l1[slot]), reinterpret_cast<uint16_t*>(out_l1[slot]), m / F);
        noc_async_write(out_l1[slot], out.get_noc_addr(r, (c0 / F) * 2u), (m / F) * 2u);
    }
    noc_async_write_barrier();
}
