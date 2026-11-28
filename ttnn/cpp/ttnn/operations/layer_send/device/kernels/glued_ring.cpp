// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Moved verbatim from models/demos/deepseek_v41_flash_d_p/tt/packed_glued.py _GRING_KERNEL (GluedSend) (NC.3).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a row (a page)
constexpr uint32_t B = get_compile_time_arg_val(1);     // rows read a NoC barrier
constexpr uint32_t RP = get_compile_time_arg_val(2);    // ring pieces a segment
constexpr uint32_t NS = get_compile_time_arg_val(3);    // segments
constexpr uint32_t QR = get_compile_time_arg_val(4);    // the chip's own rows
constexpr uint32_t W0 = get_compile_time_arg_val(5);    // the region row of own row 0 (128)
constexpr auto o_args = TensorAccessorArgs<6>();
constexpr auto r_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
constexpr auto g_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();

template <typename Src, typename Dst>
inline void copy_rows(const Src& S, const Dst& D, uint32_t buf, uint32_t s0, uint32_t d0, uint32_t n) {
    for (uint32_t r = 0; r < n; r += B) {
        const uint32_t nb = (n - r) < B ? (n - r) : B;
        for (uint32_t i = 0; i < nb; ++i) {
            noc_async_read(S.get_noc_addr(s0 + r + i), buf + i * PAGE, PAGE);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < nb; ++i) {
            noc_async_write(buf + i * PAGE, D.get_noc_addr(d0 + r + i), PAGE);
        }
        noc_async_write_barrier();
    }
}

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t region = get_arg_val<uint32_t>(1);
    const auto O = TensorAccessor(o_args, rows, PAGE);
    const auto R = TensorAccessor(r_args, region, PAGE);
    const uint32_t buf = (get_write_ptr(1) + 63) & ~63u;
    copy_rows(O, R, buf, 0, W0, QR);     // the chip's own rows into the region
    for (uint32_t s = 0; s < NS; ++s) {  // every segment's ring pieces into its slot's ring (from this chip's rows)
        const auto G = TensorAccessor(g_args, get_arg_val<uint32_t>(2 + s), PAGE);
        for (uint32_t p = 0; p < RP; ++p) {
            const uint32_t x = 2 + NS + 3 * (s * RP + p);
            const uint32_t n = get_arg_val<uint32_t>(x + 2);
            if (n) {
                copy_rows(O, G, buf, get_arg_val<uint32_t>(x), get_arg_val<uint32_t>(x + 1), n);
            }
        }
    }
}
