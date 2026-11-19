// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated bench (compute only, one core, Blackhole, fp32 DEST half-sync, LoFi): stride-2 taps from DEST
// (op_copy/kernels/conv2d_3x3_s2_taps.hpp, s2_taps::build; bench variants s2t_mask.hpp, s2t_planes.hpp). The input slab
// (CB_IN: the slab plan's input tile-rows of output pixels [o_first, o_first + 32 rows), ct tiles each) is resident (a
// reader pushes it once). Each rep builds every tap of `rows` output tile-rows x 3 kernel rows into the [rows, 9 ct]
// in0 block (CB_OUT; K order (kh, kw, c)), which the kernel then pops.
//
// variant = diag (0 full, 1 pack only, 2 loads + moves, no pack); 99: empty (push / pop only).
// RAW LLK (bench): see conv2d_3x3_s2_taps.hpp.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "s2t_planes.hpp"
#include "../op_copy/kernels/conv2d_3x3_s2_taps.hpp"

namespace conv2d_3x3::s2_taps_mask {
template <
    uint32_t ct,
    uint32_t cb_in,
    uint32_t cb_out,
    uint32_t out_w,
    int32_t in_h,
    int32_t in_w,
    int32_t num_in_px,
    uint32_t mode,
    uint32_t diag = 0,
    typename OutOff,
    typename Dead>
ALWI void build_planes(DataflowBuffer& in, uint32_t o_first, uint32_t o_end, const OutOff& out_off, const Dead& dead) {
    static_assert(in_w % 64 == 0 && out_w * 2 == static_cast<uint32_t>(in_w));
    constexpr uint32_t tpr = in_w / 32;
    SlabPlan plan;
    slab_plan<out_w, in_w, num_in_px>(o_first, o_end, plan);
    init<cb_in, cb_out>();
    [[maybe_unused]] uint32_t in_l1 = 0;
    [[maybe_unused]] uint32_t tile_bytes = 0;
    UNPACK((in_l1 = get_local_cb_interface(get_operand_id(cb_in)).fifo_rd_ptr << 4));
    UNPACK((tile_bytes = get_local_cb_interface(get_operand_id(cb_in)).fifo_page_size << 4));
    const int32_t ho_first = static_cast<int32_t>(o_first / out_w);
    const int32_t ho_last = static_cast<int32_t>((o_end - 1) / out_w);
    uint32_t waited = 0;  // slab rows waited for
    for (int32_t h = 2 * ho_first - 1; h <= 2 * ho_last + 1; ++h) {
        // Odd h: kernel row 2 of output row (h - 1) / 2, then kernel row 0 of (h + 1) / 2; even h: kernel row 1 of h
        // / 2.
        for (int32_t kh = (h & 1) ? 2 : 1; kh >= 0; kh -= 2) {
            const int32_t ho = (h + 1 - kh) / 2;
            if (ho < ho_first || ho > ho_last) {
                continue;
            }
            // The block's tiles of output row ho: t in [t_lo, t_hi), columns from c_lo.
            const uint32_t row_first = static_cast<uint32_t>(ho) * out_w;
            const uint32_t p_lo = row_first > o_first ? row_first : o_first;
            const uint32_t p_hi = row_first + out_w < o_end ? row_first + out_w : o_end;
            const uint32_t t_lo = (p_lo - o_first) / 32;
            const uint32_t n = (p_hi - p_lo) / 32;
            const uint32_t c_lo = (p_lo - row_first) / 32;
            const bool live = h >= 0 && h < in_h;
            uint32_t u = 0;
            while (u < n) {
                if (dead(t_lo + u, kh)) {
                    ++u;
                    continue;
                }
                uint32_t m = 1;  // the sub-run of units the walk reads
                while (u + m < n && !dead(t_lo + u + m, kh)) {
                    ++m;
                }
                const uint32_t c = c_lo + u;
                [[maybe_unused]] uint32_t k_first = 0;  // tile j of the row at slab row k_first + j
                if (live) {
                    const uint32_t row0 = static_cast<uint32_t>(h) * tpr;
                    const uint32_t need = slab_row(plan, row0 + 2 * (c + m) - 1) + 1;
                    if (need > waited) {
                        in.wait_front(need * ct);
                        waited = need;
                    }
                    const uint32_t j_first = c > 0 ? 2 * c - 1 : 0;
                    k_first = slab_row(plan, row0 + j_first) - j_first;
                }
                for (uint32_t ch = 0; ch < ct; ++ch) {
                    auto tile_l1 = [&](uint32_t j) { return in_l1 + ((k_first + j) * ct + ch) * tile_bytes; };
                    auto off = [&](uint32_t v) { return out_off(t_lo + u + v, kh, ch); };
                    if constexpr (mode == 0) {
                        run_row<cb_out, diag>(c, m, live, ct, tile_l1, off);
                    } else {
                        s2_planes::run_row<mode, cb_out, diag>(c, m, live, ct, tile_l1, off);
                    }
                }
                u += m;
            }
            if (kh == 0) {
                break;
            }
        }
    }
    uninit<cb_out>();
}

}  // namespace conv2d_3x3::s2_taps_mask

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
    constexpr uint32_t o_first = get_compile_time_arg_val(6);
    constexpr uint32_t Wo = W / 2;
    constexpr uint32_t num_out_px = ((H - 1) / 2 + 1) * Wo;
    constexpr uint32_t o_end = o_first + rows * 32 < num_out_px ? o_first + rows * 32 : num_out_px;
    constexpr uint32_t in0_k = 9 * ct;
    // variant = 10 * mode + diag (mode 0: the mask-only design (s2t_mask.hpp); 1: + O plane, PACRs issued directly;
    // 2: E / O planes; 3: the op's s2_taps (O plane, replayed PACRs)), 99 empty.
    constexpr uint32_t diag = variant == 99 ? 0 : variant % 10;
    constexpr uint32_t mode = variant == 99 ? 0 : variant / 10;

    compute_kernel_hw_startup(CB_IN, CB_OUT);
    DataflowBuffer in(CB_IN);
    DataflowBuffer out(CB_OUT);
    for (uint32_t rep = 0; rep < reps; ++rep) {
        out.reserve_back(rows * in0_k);
        if constexpr (variant != 99) {
            if constexpr (mode == 3) {
                conv2d_3x3::s2_taps::build<ct, CB_IN, CB_OUT, Wo, H, W, H * W, diag>(
                    in,
                    o_first,
                    o_end,
                    [](uint32_t t, int32_t kh, uint32_t ch) { return t * in0_k + kh * 3 * ct + ch; },
                    [](uint32_t, int32_t) { return false; });
            } else {
                conv2d_3x3::s2_taps_mask::build_planes<ct, CB_IN, CB_OUT, Wo, H, W, H * W, mode, diag>(
                    in,
                    o_first,
                    o_end,
                    [](uint32_t t, int32_t kh, uint32_t ch) { return t * in0_k + kh * 3 * ct + ch; },
                    [](uint32_t, int32_t) { return false; });
            }
        }
        out.push_back(rows * in0_k);
        cb_wait_front(CB_OUT, rows * in0_k);
        cb_pop_front(CB_OUT, rows * in0_k);
    }
}
