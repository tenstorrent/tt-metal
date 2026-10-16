// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — matmul-core compute (TRISCs). fp32 DEST + full sync (8-tile subblocks).
//
// Per entry (one wave of a block): matmul_entry — the wave's rows x cols x Kt, K-windows of kb, into ONE reserved fp32
// slot of cb_partial_hold. DEST-resident K (Perf 1): the entry's K-windows are grouped into super-blocks of S windows
// (CT arg; entry 0 uses 1: its windows arrive at DRAM pace); subblock-outer / K-inner, each output subblock accumulates
// a whole super-block in fp32 DEST and is packed ONCE per super-block (plain pack for the first, packer L1-acc after).
// The wave's tiles are dense at the front of the P-tile slot (row width = cols); the W window keeps its nominal row
// stride (win_n), so a ragged last wave reads and pops the nominal window.
//
// RAW LLK — measured helper bypass (do NOT "fix" back to compute_kernel_lib::matmul_block): the helper's loop is
// K-block-outer / subblock-inner and acquires + packs DEST per (K-block, subblock), so it cannot keep a subblock in
// DEST across K-blocks; it also cannot address a fronted window that the CB ring wrapped. In fp32 full-sync DEST math
// stalls for every pack, and the per-window fp32 L1-acc packs were ~13% of the matmul. Measured (focus 640x2048x7168,
// QB2, traced, compute floor with MRS_ABLATE=dram,mcast): helper 112.8 us, this path at S=1 111.6, at S=2 109.1; whole
// op 126.4 -> 123.5 us; GLM 200.3 -> 194.6; MiMo 219.2 -> 208.0 (perf_experiments/dest_resident_k, changelog Perf 1).
// The windows of a super-block stay fronted until its last subblock; a wrapped window is addressed by pointing the
// UNPACK thread's fifo_rd_ptr at it, and the last subblock pops each window right after consuming it, so the read
// pointer is exact again at the end.
// Then finish_block for the lagged entry (or this one): copy (chain start), hold + arrival (relay / fin with one
// upstream), or hold + fwd + bwd (fin with two upstreams). FIN_ADD_FPU: FPU adds (hold + arrival, then the second
// arrival through DEST reuse); else SFPU fp32 adds (FPU fwd+bwd into D1, SFPU D0 += D1).
// Every cross-device add runs in fp32 DEST; the result is packed to bf16 into cb_finished.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"  // permanent stage zones (free when the profiler is off)
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t ROLE_COPY = 0, ROLE_ADD_FWD = 1, ROLE_ADD_BWD = 2, ROLE_ADD_BOTH = 3;

template <uint32_t cb_hold, uint32_t cb_finished, uint32_t cb_arr_fwd, uint32_t cb_arr_bwd, uint32_t P, bool fpu_add>
ALWI void finish_block(uint32_t role) {
    MaybeDeviceZoneScope("FIN");
    using namespace ckl;
    constexpr auto shape = IterationShape::tiles(P);
    if (role == ROLE_COPY) {
        copy<input(cb_hold), output(cb_finished)>(shape);
    } else if constexpr (fpu_add) {
        // FPU adds into fp32 DEST. The hold reaches DEST through srcA here exactly as through CopyTile in the SFPU
        // variant (no UnpackToDestFp32), so the sum sees the same operand precision at a fraction of the cost.
        if (role == ROLE_ADD_FWD) {
            add<input(cb_hold), input(cb_arr_fwd), output(cb_finished)>(shape);
        } else if (role == ROLE_ADD_BWD) {
            add<input(cb_hold), input(cb_arr_bwd), output(cb_finished)>(shape);
        } else {
            eltwise_chain(
                shape,
                BinaryFpu<BinaryFpuOp::Add, input(cb_hold), input(cb_arr_fwd)>{},
                DestReuseBinary<BinaryFpuOp::Add, input(cb_arr_bwd), DestReuseType::DEST_TO_SRCA>{},
                PackTile<output(cb_finished)>{});
        }
    } else {
        if (role == ROLE_ADD_FWD) {
            binary_sfpu<AddBinary<>, input(cb_hold), input(cb_arr_fwd), output(cb_finished)>(shape);
        } else if (role == ROLE_ADD_BWD) {
            binary_sfpu<AddBinary<>, input(cb_hold), input(cb_arr_bwd), output(cb_finished)>(shape);
        } else {
            eltwise_chain(
                shape,
                CopyTile<input(cb_hold)>{},
                BinaryFpu<BinaryFpuOp::Add, input(cb_arr_fwd), input(cb_arr_bwd), Dst::D1>{},
                AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
                PackTile<output(cb_finished)>{});
        }
    }
}

void kernel_main() {
    constexpr uint32_t cb_act_k_window = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_k_window = get_compile_time_arg_val(1);
    constexpr uint32_t cb_partial_hold = get_compile_time_arg_val(2);
    constexpr uint32_t cb_finished = get_compile_time_arg_val(3);
    constexpr uint32_t cb_arrivals_fwd = get_compile_time_arg_val(4);
    constexpr uint32_t cb_arrivals_bwd = get_compile_time_arg_val(5);
    constexpr uint32_t win_n = get_compile_time_arg_val(6);  // W window row stride (tiles): one wave's nominal width
    constexpr uint32_t kb = get_compile_time_arg_val(7);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t P = get_compile_time_arg_val(9);              // slot tiles (the largest wave)
    constexpr uint32_t num_entries = get_compile_time_arg_val(10);   // G x num_waves
    constexpr bool fin_add_fpu = get_compile_time_arg_val(11) != 0;  // finish adds on the FPU (else SFPU fp32 adds)
    constexpr uint32_t num_waves = get_compile_time_arg_val(12);
    constexpr uint32_t super_block = get_compile_time_arg_val(13);  // K-windows per DEST-resident pack (divides nkb)
    static_assert(super_block >= 1 && num_k_blocks % super_block == 0, "super-block must divide the K-blocks");
    // ENTRY-0 RAMP (perf_experiments/entry0_ramp): entry 0's first ramp_windows windows arrive in ramp_pieces k-pieces
    // [ramp_end[p-1], ramp_end[p]); each piece is one DEST unit (all subblocks accumulate the piece's k range, then
    // pack with L1-acc), so the first matmul waits for piece 0 only. A window is popped after its last piece.
    constexpr uint32_t ramp_windows = get_compile_time_arg_val(14);
    constexpr uint32_t ramp_pieces = get_compile_time_arg_val(15);
    constexpr bool ramp_a_whole = (get_compile_time_arg_val(16) & 1) != 0;
    constexpr uint32_t ramp_end[8] = {
        get_compile_time_arg_val(17),
        get_compile_time_arg_val(18),
        get_compile_time_arg_val(19),
        get_compile_time_arg_val(20),
        get_compile_time_arg_val(21),
        get_compile_time_arg_val(22),
        get_compile_time_arg_val(23),
        get_compile_time_arg_val(24)};
    // ramp_dest: the pieces are NOT separate DEST units: each ramped window is one unit (packed once per subblock, as
    // unramped), and its first subblock waits for the pieces progressively inside its k loop (no extra packs)
    constexpr bool ramp_dest = (get_compile_time_arg_val(16) & 16) != 0;
    constexpr uint32_t ramp_units = ramp_dest ? 0 : ramp_windows * ramp_pieces;  // entry-0 piece units (0: none)
    // runtime args: per wave rows | cols << 8 | sb_h << 16 | sb_w << 24, then per entry role | lag << 8
    constexpr uint32_t roles_idx = num_waves;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_k_window, cb_weight_k_window, cb_partial_hold);

    CircularBuffer act_buf(cb_act_k_window), weight_buf(cb_weight_k_window), hold_buf(cb_partial_hold);

    auto matmul_entry = [&](uint32_t e) {
        MaybeDeviceZoneScope("MM");
        const uint32_t wave = get_arg_val<uint32_t>(e % num_waves);
        const uint32_t rows = wave & 0xFF, cols = (wave >> 8) & 0xFF;
        const uint32_t sb_h = (wave >> 16) & 0xFF, sb_w = wave >> 24;
        const uint32_t n_sb_r = rows / sb_h, n_sb_c = cols / sb_w;
        const uint32_t a_win = rows * kb, w_win = win_n * kb;  // window tiles, as the reader pushes them
        const uint32_t row_group = sb_h * cols;
        reconfig_data_format(cb_weight_k_window, cb_act_k_window);  // matmul: srcA = in1, srcB = in0
        PACK((pack_reconfig_data_format(cb_partial_hold)));
        ckernel::matmul_block_init(cb_act_k_window, cb_weight_k_window, false, sb_w, sb_h, kb);
        const uint32_t S = e == 0 ? 1 : super_block;
        // entry 0: the ramp's piece units, then one unit per remaining window
        const uint32_t n_super =
            e == 0 ? ramp_units + (num_k_blocks - (ramp_units ? ramp_windows : 0)) : num_k_blocks / S;
        hold_buf.reserve_back(P);  // one fp32 slot per entry
        for (uint32_t g = 0; g < n_super; ++g) {
            // unit g of entry 0 inside the ramp: k range [k_lo, k_hi) of its window; else the whole window(s)
            const bool piece = e == 0 && g < ramp_units;
            const uint32_t pc = piece ? g % ramp_pieces : 0;
            const uint32_t k_lo = piece && pc ? ramp_end[pc - 1] : 0, k_hi = piece ? ramp_end[pc] : kb;
            const bool window_end = !piece || pc + 1 == ramp_pieces;  // this unit finishes its window (pop)
            const uint32_t a_need = piece ? (ramp_a_whole ? a_win : (a_win / kb) * k_hi) : a_win;
            const uint32_t w_need = piece ? win_n * k_hi : w_win;
            uint32_t a_rd0 = 0, w_rd0 = 0, a_step = 0, w_step = 0;
            UNPACK(({
                a_rd0 = get_local_cb_interface(cb_act_k_window).fifo_rd_ptr;
                w_rd0 = get_local_cb_interface(cb_weight_k_window).fifo_rd_ptr;
                a_step = a_win * get_local_cb_interface(cb_act_k_window).fifo_page_size;
                w_step = w_win * get_local_cb_interface(cb_weight_k_window).fifo_page_size;
            }));
            for (uint32_t sr = 0; sr < n_sb_r; ++sr) {
                for (uint32_t sc = 0; sc < n_sb_c; ++sc) {
                    const bool first_sb = sr == 0 && sc == 0;
                    const bool last_sb = sr == n_sb_r - 1 && sc == n_sb_c - 1;
                    const bool prog = ramp_dest && e == 0 && g < ramp_windows && first_sb;  // per-piece waits
                    tile_regs_acquire();
                    for (uint32_t wnd = 0; wnd < S; ++wnd) {
                        if (first_sb && !prog) {  // progressive: the first subblock starts on the first window
                            const uint32_t ahead = (last_sb ? 0 : wnd) + 1;  // a lone subblock pops as it goes
                            act_buf.wait_front(piece ? a_need : ahead * a_win);
                            weight_buf.wait_front(piece ? w_need : ahead * w_win);
                        }
                        UNPACK(({
                            auto& ca = get_local_cb_interface(cb_act_k_window);
                            auto& cw = get_local_cb_interface(cb_weight_k_window);
                            uint32_t pa = a_rd0 + wnd * a_step;
                            if (pa >= ca.fifo_limit) {
                                pa -= ca.fifo_size;
                            }
                            uint32_t pw = w_rd0 + wnd * w_step;
                            if (pw >= cw.fifo_limit) {
                                pw -= cw.fifo_size;
                            }
                            ca.fifo_rd_ptr = pa;
                            cw.fifo_rd_ptr = pw;
                        }));
                        uint32_t in0_index = sr * sb_h * kb + k_lo;
                        uint32_t in1_index = sc * sb_w + k_lo * win_n;
                        uint32_t pnext = 0;
                        for (uint32_t kk = k_lo; kk < k_hi; ++kk) {
                            if (prog && kk == (pnext == 0 ? 0u : ramp_end[pnext - 1])) {  // piece pnext starts
                                act_buf.wait_front(ramp_a_whole ? a_win : (a_win / kb) * ramp_end[pnext]);
                                weight_buf.wait_front(win_n * ramp_end[pnext]);
                                ++pnext;
                            }
#ifndef SKIP_COMPUTE  // perf ablation (MRS_ABLATE=compute): stub the math, keep every CB wait / pop / pack
                            ckernel::matmul_block(
                                cb_act_k_window, cb_weight_k_window, in0_index, in1_index, 0, false, sb_w, sb_h, kb);
#endif
                            ++in0_index;
                            in1_index += win_n;
                        }
                        if (last_sb && window_end) {  // frees each window once consumed (fifo_rd_ptr is exact here)
                            act_buf.pop_front(a_win);
                            weight_buf.pop_front(w_win);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    PACK((llk_pack_reconfig_l1_acc(g == 0 ? 0 : 1)));
                    ckl::pack_subblock_row_strided(0, cb_partial_hold, sr * row_group + sc * sb_w, cols, sb_h, sb_w);
                    tile_regs_release();
                }
            }
        }
        hold_buf.push_back(P);
        PACK((llk_pack_reconfig_l1_acc(0)));  // the finish chains pack plainly
    };

    constexpr uint32_t NONE = 0xFFFFFFFF;
    uint32_t pending = NONE;
    for (uint32_t e = 0; e < num_entries; ++e) {
        matmul_entry(e);
        if (pending != NONE) {
            finish_block<cb_partial_hold, cb_finished, cb_arrivals_fwd, cb_arrivals_bwd, P, fin_add_fpu>(
                get_arg_val<uint32_t>(roles_idx + pending) & 0xFF);
            pending = NONE;
        }
        const uint32_t role = get_arg_val<uint32_t>(roles_idx + e) & 0xFF;
        const bool lag =
            (get_arg_val<uint32_t>(roles_idx + e) >> 8) & 1;  // host: arrival expected late -> finish later
        if (lag) {
            pending = e;  // the add waits for its arrival behind the next entry's matmul
        } else {
            finish_block<cb_partial_hold, cb_finished, cb_arrivals_fwd, cb_arrivals_bwd, P, fin_add_fpu>(role);
        }
    }
}
