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

// FUSED FINISH (perf experiment fused_finish): a non-lagged entry folds its finish into the matmul's LAST super-block.
// Each output subblock of that super-block brings the earlier super-blocks' fp32 hold and the arrival(s) into DEST
// and packs bf16 straight into cb_finished (pushed per subblock row group, so the streamed fin writer starts while
// the remaining subblocks still compute). The hold round trip of the last super-block and the separate FIN pass go.
// FF_MODE 1 (pre): before the subblock's matmul, DEST = hold [+ arr] [+ arr2] (copy / FPU add / DEST_TO_SRCA reuse
//   add: exactly today's finish ops on the same operands), then the matmul accumulates the last super-block on top in
//   fp32 DEST (the last super-block itself never rounds through srcA).
// FF_MODE 2 (post): after the subblock's matmul, DEST_TO_SRCA reuse adds of hold / arr / arr2 (rounds DEST via srcA).
// FF_MODE 0: the op's current path (separate finish).
// A lagged entry (arrival produced late) keeps the separate finish, which now runs before the NEXT entry's last
// super-block (it must precede that entry's fused output in cb_finished's FIFO order).
#ifndef FF_MODE
#define FF_MODE 1
#endif

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"  // permanent stage zones (free when the profiler is off)
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
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
    // runtime args: per wave rows | cols << 8 | sb_h << 16 | sb_w << 24, then per entry role | lag << 8
    constexpr uint32_t roles_idx = num_waves;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_k_window, cb_weight_k_window, cb_partial_hold);

    CircularBuffer act_buf(cb_act_k_window), weight_buf(cb_weight_k_window), hold_buf(cb_partial_hold);
    CircularBuffer fin_buf(cb_finished), fwd_buf(cb_arrivals_fwd), bwd_buf(cb_arrivals_bwd);

    constexpr uint32_t NONE = 0xFFFFFFFF;
    uint32_t pending = NONE;
    auto run_pending = [&]() {
        if (pending != NONE) {
            finish_block<cb_partial_hold, cb_finished, cb_arrivals_fwd, cb_arrivals_bwd, P, fin_add_fpu>(
                get_arg_val<uint32_t>(roles_idx + pending) & 0xFF);
            pending = NONE;
        }
    };

#if FF_MODE == 1 || FF_MODE == 2 || FF_MODE == 4
    // The finish ops of one output subblock, in DEST (pre: before its matmul, DEST is clear; post: after it).
    // hold / arrival tiles of the subblock sit row-strided at col_base + r * cols (dense wave layout of the slot).
    auto fuse_subblock = [&](uint32_t role, bool has_hold, uint32_t col_base, uint32_t cols, uint32_t h, uint32_t w) {
        const bool fwd = role == ROLE_ADD_FWD || role == ROLE_ADD_BOTH;
        const bool bwd = role == ROLE_ADD_BWD || role == ROLE_ADD_BOTH;
        const uint32_t arr1 = fwd ? cb_arrivals_fwd : cb_arrivals_bwd;  // first arrival operand
#if FF_MODE == 1 || FF_MODE == 4
        // srcA = hold (or the first arrival), srcB = the next operand; matmul state was srcA = W, srcB = A
        if (role == ROLE_COPY || (!has_hold && role != ROLE_ADD_BOTH)) {
            const uint32_t src = role == ROLE_COPY ? cb_partial_hold : arr1;  // role COPY implies has_hold here
            copy_tile_to_dst_init_short_with_dt(cb_weight_k_window, src);
            for (uint32_t r = 0; r < h; ++r) {
                copy_block(src, r * cols + col_base, r * w, w);
            }
            reconfig_data_format_srca(src, cb_weight_k_window);
        } else {
            const uint32_t a_cb = has_hold ? cb_partial_hold : cb_arrivals_fwd;
            const uint32_t b_cb = has_hold ? arr1 : cb_arrivals_bwd;
            reconfig_data_format(cb_weight_k_window, a_cb, cb_act_k_window, b_cb);
            add_init(a_cb, b_cb);
            for (uint32_t r = 0; r < h; ++r) {
                for (uint32_t c = 0; c < w; ++c) {
                    add_tiles(a_cb, b_cb, r * cols + col_base + c, r * cols + col_base + c, r * w + c);
                }
            }
            if (has_hold && role == ROLE_ADD_BOTH) {  // + bwd: DEST -> srcA reuse (today's FPU fin chain)
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_arrivals_bwd);
                for (uint32_t r = 0; r < h; ++r) {
                    for (uint32_t c = 0; c < w; ++c) {
                        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                            cb_arrivals_bwd, r * cols + col_base + c, r * w + c);
                    }
                }
            }
            reconfig_data_format(
                a_cb, cb_weight_k_window, has_hold && role == ROLE_ADD_BOTH ? cb_arrivals_bwd : b_cb, cb_act_k_window);
        }
#else  // FF_MODE == 2: post-matmul DEST_TO_SRCA reuse adds; srcA must carry DEST at fp32 width (hold's format)
        reconfig_data_format(cb_weight_k_window, cb_partial_hold, cb_act_k_window, cb_partial_hold);
        uint32_t cur_b = cb_partial_hold;
        auto reuse_add = [&](uint32_t cb) {
            reconfig_data_format_srcb(cur_b, cb);
            cur_b = cb;
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb);
            for (uint32_t r = 0; r < h; ++r) {
                for (uint32_t c = 0; c < w; ++c) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        cb, r * cols + col_base + c, r * w + c);
                }
            }
        };
        if (has_hold) {
            reuse_add(cb_partial_hold);
        }
        if (fwd) {
            reuse_add(cb_arrivals_fwd);
        }
        if (bwd) {
            reuse_add(cb_arrivals_bwd);
        }
        reconfig_data_format(cb_partial_hold, cb_weight_k_window, cur_b, cb_act_k_window);
#endif
        ckernel::matmul_block_init(cb_act_k_window, cb_weight_k_window, false, w, h, kb);
    };
#endif
#if FF_MODE == 3
    // pre: DEST = hold (copy through srcA, exactly today's finish operand path); no arrival needed yet
    auto pre_hold = [&](uint32_t col_base, uint32_t cols, uint32_t h, uint32_t w) {
        copy_tile_to_dst_init_short_with_dt(cb_weight_k_window, cb_partial_hold);
        for (uint32_t r = 0; r < h; ++r) {
            copy_block(cb_partial_hold, r * cols + col_base, r * w, w);
        }
        reconfig_data_format_srca(cb_partial_hold, cb_weight_k_window);
        ckernel::matmul_block_init(cb_act_k_window, cb_weight_k_window, false, w, h, kb);
    };
    // post: two arrivals -> one FPU add accumulated onto DEST (DEST += fwd + bwd, the DEST partial never leaves
    // DEST); one arrival -> DEST_TO_SRCA reuse add (DEST rounds through srcA at fp32 / TF32 width)
    auto post_arrivals = [&](uint32_t role, uint32_t col_base, uint32_t cols, uint32_t h, uint32_t w) {
        if (role == ROLE_ADD_BOTH) {
            reconfig_data_format(cb_weight_k_window, cb_arrivals_fwd, cb_act_k_window, cb_arrivals_bwd);
            add_init(cb_arrivals_fwd, cb_arrivals_bwd, true /* acc_to_dest */);
            for (uint32_t r = 0; r < h; ++r) {
                for (uint32_t c = 0; c < w; ++c) {
                    add_tiles(
                        cb_arrivals_fwd, cb_arrivals_bwd, r * cols + col_base + c, r * cols + col_base + c, r * w + c);
                }
            }
            reconfig_data_format(cb_arrivals_fwd, cb_weight_k_window, cb_arrivals_bwd, cb_act_k_window);
        } else {
            const uint32_t arr = role == ROLE_ADD_FWD ? cb_arrivals_fwd : cb_arrivals_bwd;
            reconfig_data_format(cb_weight_k_window, cb_partial_hold, cb_act_k_window, arr);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(arr);
            for (uint32_t r = 0; r < h; ++r) {
                for (uint32_t c = 0; c < w; ++c) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        arr, r * cols + col_base + c, r * w + c);
                }
            }
            reconfig_data_format(cb_partial_hold, cb_weight_k_window, arr, cb_act_k_window);
        }
        ckernel::matmul_block_init(cb_act_k_window, cb_weight_k_window, false, w, h, kb);
    };
#endif

    // fused: this entry's finish is folded into its last super-block (pack bf16 -> cb_finished)
    // returns whether the entry's finish was fused. eligible: may fuse; dyn: fuse only if the arrival(s) already
    // landed when the last super-block starts (decided by UNPACK, broadcast to MATH / PACK through the mailboxes)
    auto matmul_entry = [&](uint32_t e, bool eligible, bool dyn, uint32_t role) -> bool {
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
        const uint32_t n_super = num_k_blocks / S;
        const bool has_hold = !eligible || dyn || n_super > 1;
        bool fused = eligible;
        if (has_hold) {
            hold_buf.reserve_back(P);  // one fp32 slot per entry
        }
        for (uint32_t g = 0; g < n_super; ++g) {
            if (eligible && dyn && g == n_super - 1) {
                uint32_t ready = 0;
                UNPACK(({
                    auto avail = [&](uint32_t cb) {
                        const uint16_t rec = (uint16_t)reg_read((uint32_t)get_cb_tiles_received_ptr(cb));
                        return (uint16_t)(rec - (uint16_t)get_local_cb_interface(cb).tiles_acked) >= P;
                    };
                    ready = 1;
                    if (role == ROLE_ADD_FWD || role == ROLE_ADD_BOTH) {
                        ready &= avail(cb_arrivals_fwd) ? 1u : 0u;
                    }
                    if (role == ROLE_ADD_BWD || role == ROLE_ADD_BOTH) {
                        ready &= avail(cb_arrivals_bwd) ? 1u : 0u;
                    }
                    mailbox_write(ckernel::ThreadId::MathThreadId, ready);
                    mailbox_write(ckernel::ThreadId::PackThreadId, ready);
                }));
                MATH((ready = mailbox_read(ckernel::ThreadId::UnpackThreadId)));
                PACK((ready = mailbox_read(ckernel::ThreadId::UnpackThreadId)));
                fused = ready != 0;
            }
            const bool fuse_now = fused && g == n_super - 1;
#if FF_MODE != 0
            if (fuse_now) {
                // the lagged previous entry's finish goes first (FIFO order of cb_finished)
                if (pending != NONE) {
                    run_pending();
                    reconfig_data_format(cb_weight_k_window, cb_act_k_window);
                    ckernel::matmul_block_init(cb_act_k_window, cb_weight_k_window, false, sb_w, sb_h, kb);
                }
                if (has_hold) {
                    hold_buf.push_back(P);
                    hold_buf.wait_front(P);
                }
#ifndef FF_ABLATE_ARR_LATE  // perf-only ablation: wait the arrivals after the super-block (adds read stale data)
                if (role == ROLE_ADD_FWD || role == ROLE_ADD_BOTH) {
                    fwd_buf.wait_front(P);
                }
                if (role == ROLE_ADD_BWD || role == ROLE_ADD_BOTH) {
                    bwd_buf.wait_front(P);
                }
#endif
                PACK((pack_reconfig_data_format(cb_finished)));
                PACK((llk_pack_reconfig_l1_acc(0)));
                fin_buf.reserve_back(P);
            }
#endif
            auto super_block_body = [&]() {
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
                        tile_regs_acquire();
#if FF_MODE == 1 || FF_MODE == 4
                        if (fuse_now && (has_hold || role != ROLE_COPY)) {
                            fuse_subblock(role, has_hold, sr * row_group + sc * sb_w, cols, sb_h, sb_w);
                        }
#elif FF_MODE == 3
                        if (fuse_now && has_hold) {
                            pre_hold(sr * row_group + sc * sb_w, cols, sb_h, sb_w);
                        }
#endif
                        for (uint32_t wnd = 0; wnd < S; ++wnd) {
                            if (first_sb) {  // progressive: the first subblock starts on the first window
                                const uint32_t ahead = (last_sb ? 0 : wnd) + 1;  // a lone subblock pops as it goes
                                act_buf.wait_front(ahead * a_win);
                                weight_buf.wait_front(ahead * w_win);
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
                            uint32_t in0_index = sr * sb_h * kb;
                            uint32_t in1_index = sc * sb_w;
#ifndef SKIP_COMPUTE  // perf ablation (MRS_ABLATE=compute): stub the math, keep every CB wait / pop / pack
                            for (uint32_t kk = 0; kk < kb; ++kk) {
                                ckernel::matmul_block(
                                    cb_act_k_window,
                                    cb_weight_k_window,
                                    in0_index,
                                    in1_index,
                                    0,
                                    false,
                                    sb_w,
                                    sb_h,
                                    kb);
                                ++in0_index;
                                in1_index += win_n;
                            }
#endif
                            if (last_sb) {  // frees each window as soon as it is consumed (fifo_rd_ptr is exact here)
                                act_buf.pop_front(a_win);
                                weight_buf.pop_front(w_win);
                            }
                        }
#if FF_MODE == 2
                        if (fuse_now && (has_hold || role != ROLE_COPY)) {
                            fuse_subblock(role, has_hold, sr * row_group + sc * sb_w, cols, sb_h, sb_w);
                        }
#elif FF_MODE == 3
                        if (fuse_now && role != ROLE_COPY) {
                            post_arrivals(role, sr * row_group + sc * sb_w, cols, sb_h, sb_w);
                        }
#endif
                        tile_regs_commit();
                        tile_regs_wait();
                        if (fuse_now) {
                            // rows of earlier row groups were pushed: offsets are relative to this row group
                            ckl::pack_subblock_row_strided(0, cb_finished, sc * sb_w, cols, sb_h, sb_w);
                        } else {
                            PACK((llk_pack_reconfig_l1_acc(g == 0 ? 0 : 1)));
                            ckl::pack_subblock_row_strided(
                                0, cb_partial_hold, sr * row_group + sc * sb_w, cols, sb_h, sb_w);
                        }
                        tile_regs_release();
                    }
                    if (fuse_now) {
                        fin_buf.push_back(row_group);  // the streamed fin writer can store this row group now
                    }
                }
            };
            if (fuse_now) {
                MaybeDeviceZoneScope("MM_FUSED");  // the fused last super-block (matmul + finish)
                super_block_body();
            } else {
                super_block_body();
            }
        }
        if (fused) {
            if (rows * cols < P) {
                fin_buf.push_back(P - rows * cols);  // a ragged wave: the nominal slot stays whole
            }
            if (has_hold) {
                hold_buf.pop_front(P);
            }
#ifdef FF_ABLATE_ARR_LATE
            if (role == ROLE_ADD_FWD || role == ROLE_ADD_BOTH) {
                fwd_buf.wait_front(P);
            }
            if (role == ROLE_ADD_BWD || role == ROLE_ADD_BOTH) {
                bwd_buf.wait_front(P);
            }
#endif
            if (role == ROLE_ADD_FWD || role == ROLE_ADD_BOTH) {
                fwd_buf.pop_front(P);
            }
            if (role == ROLE_ADD_BWD || role == ROLE_ADD_BOTH) {
                bwd_buf.pop_front(P);
            }
        } else {
            hold_buf.push_back(P);
        }
        PACK((llk_pack_reconfig_l1_acc(0)));  // the finish chains pack plainly
        return fused;
    };

    for (uint32_t e = 0; e < num_entries; ++e) {
        const uint32_t role = get_arg_val<uint32_t>(roles_idx + e) & 0xFF;
        const bool lag =
            (get_arg_val<uint32_t>(roles_idx + e) >> 8) & 1;  // host: arrival expected late -> finish later
        const uint32_t n_super = num_k_blocks / (e == 0 ? 1 : super_block);
        // fused unless lagged; a single-super-block entry behind a lagged one stays unfused (the pending finish
        // would otherwise run before this whole matmul and lose its lag)
        const uint32_t word = get_arg_val<uint32_t>(roles_idx + e);
        const bool fuse_static = (word >> 9) & 1;                                 // host policy: always fuse
        const bool fuse_dyn = FF_MODE == 4 && ((word >> 10) & 1) && n_super > 1;  // host policy: fuse if landed
        const bool eligible = FF_MODE != 0 && (fuse_static || fuse_dyn) && !lag && !(pending != NONE && n_super == 1);
        const bool fused = matmul_entry(e, eligible, fuse_dyn && !fuse_static, role);
        run_pending();  // (no-op when the fused entry already ran it)
        if (lag) {
            pending = e;  // the add waits for its arrival behind the next entry's matmul
        } else if (!fused) {
            finish_block<cb_partial_hold, cb_finished, cb_arrivals_fwd, cb_arrivals_bwd, P, fin_add_fpu>(role);
        }
    }
}
