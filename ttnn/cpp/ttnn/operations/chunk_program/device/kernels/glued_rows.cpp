// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// tt/lane_rows.GluedRows' kernel (its GR_COMMON form: the pieces are common runtime args), NC.7: NPAIR (table,
// output) pairs a program (the RoPE cos and sin tables of one start, fused: one program, the same pieces). Every pair
// is copied exactly as the single-pair kernel copies it (the same pages, the same order), one pair after the other.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a page
constexpr uint32_t PPU = get_compile_time_arg_val(1);   // pages a unit (a row: 1; a tile row: Wt)
constexpr uint32_t B = get_compile_time_arg_val(2);     // pages a NoC barrier
constexpr uint32_t NC = get_compile_time_arg_val(3);    // cores
constexpr uint32_t GW = get_compile_time_arg_val(4);    // values an SP row
constexpr uint32_t SLOT = (PAGE + 63) & ~63u;
constexpr auto s0_args = TensorAccessorArgs<5>();
constexpr auto d0_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
#if NPAIR > 1
constexpr auto s1_args = TensorAccessorArgs<d0_args.next_compile_time_args_offset()>();
constexpr auto d1_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();
#endif

#define VAL(i) get_common_arg_val<uint32_t>(i)

template <typename SA, typename DA>
FORCE_INLINE void copy_pair(
    const SA& S, const DA& D, uint32_t buf, uint32_t w0, uint32_t np, uint32_t lo, uint32_t hi) {
    uint32_t g = lo;
    while (g < hi) {
        uint32_t nb = 0;
        const uint32_t d0 = g;
        for (; nb < B && g < hi; ++nb, ++g) {
            uint32_t sp = 0;
            for (uint32_t p = 0; p < np; ++p) {
                const uint32_t s0 = VAL(w0 + 1 + 2 * p);
                const uint32_t cd = VAL(w0 + 2 + 2 * p);
                const uint32_t cnt = cd & 0xFFFFu, dd = cd >> 16;
                if (g >= dd * PPU && g < (dd + cnt) * PPU) {
                    sp = s0 * PPU + (g - dd * PPU);
                    break;
                }
            }
            noc_async_read(S.get_noc_addr(sp), buf + nb * SLOT, PAGE);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < nb; ++t) {
            noc_async_write(buf + t * SLOT, D.get_noc_addr(d0 + t), PAGE);
        }
        noc_async_write_barrier();
    }
}

void kernel_main() {
    // [src_0, dst_0, (src_1, dst_1,) k, r]
    const uint32_t k = get_arg_val<uint32_t>(2 * NPAIR);      // this core's index of NC
    const uint32_t r = get_arg_val<uint32_t>(2 * NPAIR + 1);  // this chip's SP row (frozen: one program an SP row)
    const uint32_t w0 = r * GW;
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t np = VAL(w0);
    uint32_t total = 0;
    for (uint32_t p = 0; p < np; ++p) {
        total += (VAL(w0 + 2 + 2 * p) & 0xFFFFu) * PPU;
    }
    const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(total) * k) / NC);
    const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(total) * (k + 1)) / NC);
    {
        const auto S = TensorAccessor(s0_args, get_arg_val<uint32_t>(0), PAGE);
        const auto D = TensorAccessor(d0_args, get_arg_val<uint32_t>(1), PAGE);
        copy_pair(S, D, buf, w0, np, lo, hi);
    }
#if NPAIR > 1
    {
        const auto S = TensorAccessor(s1_args, get_arg_val<uint32_t>(2), PAGE);
        const auto D = TensorAccessor(d1_args, get_arg_val<uint32_t>(3), PAGE);
        copy_pair(S, D, buf, w0, np, lo, hi);
    }
#endif
}
