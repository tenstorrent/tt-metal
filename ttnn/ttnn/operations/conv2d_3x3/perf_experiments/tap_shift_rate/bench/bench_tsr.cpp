// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated bench (compute only, one core, Blackhole, fp32 DEST half-sync, LoFi): conv2d_3x3's tap-shift prelude, the
// kw = 0 / 2 taps of one M-block's N-slice in0 block built from the input ring (TILE slab, bf8_b).
//
// The ring (CB_IN) holds the plan's slab rows (ct tiles each) resident (a reader pushes them once). Each rep runs one
// whole prelude call into the [rows, 6 ct] in0 block (CB_OUT), which the kernel then pops.
//
// variant 0  : HEAD's conv2d_3x3::tap_shift::run (op_base copy), verbatim.
// variant 1+ : conv2d_3x3::tsr::run<b16, move4, sec, diag> (tsr.hpp), see VARIANTS in test_bench.py.
// variant 99 : empty (push / pop only): the fixed launch overhead.
//
// RAW LLK (bench): the shift has no helper (see conv2d_3x3_tap_shift.hpp).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "tsr.hpp"
#include "tsr2.hpp"

namespace {
constexpr uint32_t CB_IN = 0;
constexpr uint32_t CB_OUT = 16;
}  // namespace

void kernel_main() {
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    constexpr uint32_t reps = get_compile_time_arg_val(1);
    constexpr uint32_t ct = get_compile_time_arg_val(2);
    constexpr uint32_t rows = get_compile_time_arg_val(3);
    constexpr int32_t W = get_compile_time_arg_val(4);
    constexpr int32_t H = get_compile_time_arg_val(5);
    constexpr int32_t d = get_compile_time_arg_val(6);
    constexpr uint32_t group_h = get_compile_time_arg_val(7);
    constexpr uint32_t o_first = get_compile_time_arg_val(8);
    constexpr uint32_t in0_k = 6 * ct;
    constexpr uint32_t G_ = (rows + group_h - 1) / group_h;
    constexpr uint32_t o_end_raw = o_first + rows * 32;
    constexpr uint32_t o_end = o_end_raw < static_cast<uint32_t>(H * W) ? o_end_raw : static_cast<uint32_t>(H * W);

    compute_kernel_hw_startup(CB_IN, CB_OUT);
    DataflowBuffer in(CB_IN);
    DataflowBuffer out(CB_OUT);

    conv2d_3x3::padded_slab::Plan plan;
    conv2d_3x3::padded_slab::plan<W, W, d, d, H * W>(o_first, o_end, plan);
    uint8_t kh_lo[G_];
    uint8_t kh_hi[G_];
    for (uint32_t g = 0; g < G_; ++g) {
        int lo = 3, hi = 0;
        for (uint32_t t = g * group_h; t < (g + 1) * group_h && t < rows; ++t) {
            const int32_t ho = (o_first + t * 32) / W;
            for (int kh = 0; kh < 3; ++kh) {
                const int32_t h = ho + kh * d - d;
                if (h >= 0 && h < H) {
                    lo = kh < lo ? kh : lo;
                    hi = kh + 1 > hi ? kh + 1 : hi;
                }
            }
        }
        if (lo >= hi) {
            lo = 0;
            hi = 1;
        }
        kh_lo[g] = static_cast<uint8_t>(lo);
        kh_hi[g] = static_cast<uint8_t>(hi);
    }

#define TSR_RUN(B16, M4, SEC, DIAG)                                                                 \
    conv2d_3x3::tsr::run<B16, M4, SEC, DIAG, ct, in0_k, CB_IN, CB_OUT, W, H, W, d, d, group_h, G_>( \
        in, out, rows, o_first, o_end, plan, kh_lo, kh_hi)

    for (uint32_t rep = 0; rep < reps; ++rep) {
        if constexpr (variant == 0) {
            conv2d_3x3::tap_shift::run<ct, in0_k, CB_IN, CB_OUT, W, H, W, d, d, group_h, G_>(
                in, out, rows, o_first, o_end, plan, kh_lo, kh_hi);
        } else if constexpr (variant == 1) {
            TSR_RUN(false, false, 4, 0);
        } else if constexpr (variant == 2) {
            TSR_RUN(false, false, 4, 1);
        } else if constexpr (variant == 3) {
            TSR_RUN(false, false, 4, 2);
        } else if constexpr (variant == 4) {
            TSR_RUN(false, true, 4, 0);
        } else if constexpr (variant == 10) {
            TSR_RUN(true, false, 4, 0);
        } else if constexpr (variant == 11) {
            TSR_RUN(true, true, 4, 0);
        } else if constexpr (variant == 12) {
            TSR_RUN(true, true, 8, 0);
        } else if constexpr (variant == 13) {
            TSR_RUN(true, false, 8, 1);
        } else if constexpr (variant == 14) {
            TSR_RUN(true, false, 8, 2);
        } else if constexpr (variant == 15) {
            TSR_RUN(true, false, 4, 1);
        } else if constexpr (variant == 16) {
            TSR_RUN(true, false, 8, 0);
        } else if constexpr (variant >= 20 && variant < 32) {
            // rowbuf table: G, rdiag, b16, pmode
            constexpr uint32_t G = variant == 21 ? 1 : (variant == 22 ? 4 : 2);
            constexpr uint32_t rdiag = (variant >= 23 && variant <= 25) ? variant - 22 : (variant == 29 ? 1 : 0);
            constexpr bool b16 = !(variant >= 27 && variant <= 29);
            constexpr uint32_t pmode = (variant == 26 || variant == 28) ? 1 : 0;
            constexpr bool uoff = variant == 30 || variant == 31;
            conv2d_3x3::tsr::run_rowbuf<
                (variant == 31 ? 1 : G),
                rdiag,
                b16,
                pmode,
                uoff,
                ct,
                in0_k,
                CB_IN,
                CB_OUT,
                W,
                H,
                W,
                d,
                d,
                group_h,
                G_>(in, out, rows, o_first, o_end, plan, kh_lo, kh_hi);
        } else if constexpr (variant >= 40 && variant < 50) {
            constexpr uint32_t gmax = variant == 43 ? 1 : 2;
            constexpr uint32_t rdiag =
                variant == 41 ? 1 : (variant == 42 ? 3 : (variant == 44 ? 2 : (variant == 45 ? 4 : 0)));
            conv2d_3x3::rb2::run<gmax, rdiag, ct, in0_k, CB_IN, CB_OUT, W, H, W, d, d, group_h, G_>(
                in, out, rows, o_first, o_end, plan, kh_lo, kh_hi);
        } else {
            out.reserve_back(rows * in0_k);
            out.push_back(rows * in0_k);
        }
        cb_wait_front(CB_OUT, rows * in0_k);
        cb_pop_front(CB_OUT, rows * in0_k);
    }
}
