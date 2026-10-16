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

// FUSED FINISH (Perf 2 experiment fused_finish). An entry whose finish may fuse folds it into the matmul's LAST
// super-block: each output subblock first accumulates its last-super-block matmul in fp32 DEST, then the earlier
// super-blocks' fp32 hold and the arrival(s) are added IN DEST, and the subblock is packed bf16 straight into
// cb_finished, pushed per subblock row group (the streamed fin writer stores a row group while the next one computes).
// The last super-block's fp32 L1-acc pack, the hold re-read and the separate finish pass all disappear.
//   in-DEST adds, after the matmul (DEST holds the last super-block's partial):
//     hold + arrival            one FPU add accumulated onto DEST (ELWADD acc_to_dest: DEST += hold + arr); the DEST
//                               partial never leaves DEST, the hold reaches srcA exactly as in the separate finish
//     hold + fwd + bwd          the same acc add with fwd, then bwd by a DEST_TO_SRCA reuse add (as the FPU fin chain)
//     hold only (chain start)   DEST_TO_SRCA reuse add of the hold;  no hold (one super-block): arrivals only
//   Adding the hold AFTER the matmul (not preloading DEST with it) is load-bearing for precision: preloading DEST and
//   letting the matmul accumulate on top measured rel-RMS +3..7% on every cell (focus 0.00486 -> 0.00511).
// WHEN to fuse is decided at run time, per entry: the arrival is transport-bound (it lands about when this entry's
// matmul ends), so waiting for it inside the matmul stalls the FPU (focus +4 us when every entry fuses). After the
// first subblock of the last super-block has accumulated, UNPACK checks whether the entry's arrival(s) already landed
// (non-blocking CB credit read) and broadcasts the answer to MATH and PACK through the TRISC mailboxes (the
// get_tile_address pattern), so all three threads take the same branch. Landed (or no arrival: chain start) ->
// fused. Not landed -> the old path for this entry: pack the last super-block into the hold, separate finish after.
// The host marks an entry fusable unless it lags (its arrival is produced at the same entry index upstream; the
// finish then runs behind the next entry's matmul as before), and an entry right behind a lagged one never fuses
// (cb_finished's FIFO order: the pending finish must be pushed first).

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
    // runtime args: per wave rows | cols << 8 | sb_h << 16 | sb_w << 24, then per entry
    // role | lag << 8 | fusable << 9
    constexpr uint32_t roles_idx = num_waves;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_k_window, cb_weight_k_window, cb_partial_hold);

    CircularBuffer act_buf(cb_act_k_window), weight_buf(cb_weight_k_window), hold_buf(cb_partial_hold);
    CircularBuffer fin_buf(cb_finished), fwd_buf(cb_arrivals_fwd), bwd_buf(cb_arrivals_bwd);

    // UNPACK: have the entry's arrival(s) landed (non-blocking)? The answer goes to MATH and PACK by mailbox, so all
    // three threads branch alike. A chain start has no arrival and always fuses.
    auto arrivals_landed = [&](uint32_t role) -> bool {
        uint32_t ready = 1;
        UNPACK(({
            auto landed = [&](uint32_t cb) {
                const uint16_t rec = (uint16_t)reg_read((uint32_t)get_cb_tiles_received_ptr(cb));
                return (uint16_t)(rec - (uint16_t)get_local_cb_interface(cb).tiles_acked) >= P;
            };
            if (role == ROLE_ADD_FWD || role == ROLE_ADD_BOTH) {
                ready &= landed(cb_arrivals_fwd) ? 1u : 0u;
            }
            if (role == ROLE_ADD_BWD || role == ROLE_ADD_BOTH) {
                ready &= landed(cb_arrivals_bwd) ? 1u : 0u;
            }
            mailbox_write(ckernel::ThreadId::MathThreadId, ready);
            mailbox_write(ckernel::ThreadId::PackThreadId, ready);
        }));
        MATH((ready = mailbox_read(ckernel::ThreadId::UnpackThreadId)));
        PACK((ready = mailbox_read(ckernel::ThreadId::UnpackThreadId)));
        return ready != 0;
    };

    // The finish of one output subblock, in DEST, after its last-super-block matmul. Its hold / arrival tiles sit
    // row-strided at col_base + r * cols (the slot's dense wave layout); DEST tile r * w + c. Matmul state on entry and
    // on exit: srcA = W, srcB = A, matmul MOP.
    auto finish_in_dest = [&](uint32_t role, bool has_hold, uint32_t col_base, uint32_t cols, uint32_t h, uint32_t w) {
        const bool fwd = role == ROLE_ADD_FWD || role == ROLE_ADD_BOTH;
        const bool bwd = role == ROLE_ADD_BWD || role == ROLE_ADD_BOTH;
        // DEST_TO_SRCA reuse add of cb (srcA carries DEST at fp32 / TF32 width: hold's format; srcB = cb)
        auto reuse_add = [&](uint32_t cb, uint32_t srca_old, uint32_t srcb_old) {
            reconfig_data_format(srca_old, cb_partial_hold, srcb_old, cb);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb);
            for (uint32_t r = 0; r < h; ++r) {
                for (uint32_t c = 0; c < w; ++c) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        cb, r * cols + col_base + c, r * w + c);
                }
            }
        };
        // DEST += a + b (one FPU add accumulated onto DEST; srcA = a, srcB = b)
        auto acc_add = [&](uint32_t a, uint32_t b) {
            reconfig_data_format(cb_weight_k_window, a, cb_act_k_window, b);
            add_init(a, b, true /* acc_to_dest */);
            for (uint32_t r = 0; r < h; ++r) {
                for (uint32_t c = 0; c < w; ++c) {
                    const uint32_t t = r * cols + col_base + c;
                    add_tiles(a, b, t, t, r * w + c);
                }
            }
        };
        uint32_t srca = cb_weight_k_window, srcb = cb_act_k_window;  // what srcA / srcB are configured for now
        if (has_hold) {
#ifdef FF_REUSE_ONLY  // precision option: hold and every arrival by DEST_TO_SRCA reuse adds (one op per operand)
            reuse_add(cb_partial_hold, srca, srcb);
            srca = srcb = cb_partial_hold;
            if (fwd) {
                reuse_add(cb_arrivals_fwd, srca, srcb);
                srcb = cb_arrivals_fwd;
            }
            if (bwd) {
                reuse_add(cb_arrivals_bwd, srca, srcb);
                srcb = cb_arrivals_bwd;
            }
#else
            if (fwd || bwd) {
                const uint32_t arr = fwd ? cb_arrivals_fwd : cb_arrivals_bwd;
                acc_add(cb_partial_hold, arr);
                srca = cb_partial_hold;
                srcb = arr;
                if (fwd && bwd) {
                    reuse_add(cb_arrivals_bwd, srca, srcb);
                    srcb = cb_arrivals_bwd;
                }
            } else {
                reuse_add(cb_partial_hold, srca, srcb);
                srca = srcb = cb_partial_hold;
            }
#endif
        } else if (fwd && bwd) {
            acc_add(cb_arrivals_fwd, cb_arrivals_bwd);
            srca = cb_arrivals_fwd;
            srcb = cb_arrivals_bwd;
        } else if (fwd || bwd) {
            const uint32_t arr = fwd ? cb_arrivals_fwd : cb_arrivals_bwd;
            reuse_add(arr, srca, srcb);
            srca = cb_partial_hold;
            srcb = arr;
        } else {
            return;  // chain start with a single super-block: DEST already is the result
        }
        reconfig_data_format(srca, cb_weight_k_window, srcb, cb_act_k_window);
        ckernel::matmul_block_init(cb_act_k_window, cb_weight_k_window, false, w, h, kb);
    };

    // One entry's matmul. fusable: the finish may fold into the last super-block (taken iff its arrivals landed by
    // the end of that super-block's first subblock). Returns whether it did.
    auto matmul_entry = [&](uint32_t e, bool fusable, uint32_t role) -> bool {
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
        // a single-super-block entry without an arrival needs no hold at all; any other entry keeps its slot (an
        // arrival entry may still take the separate finish)
        const bool has_hold = !(fusable && n_super == 1 && role == ROLE_COPY);
        if (has_hold) {
            hold_buf.reserve_back(P);  // one fp32 slot per entry
        }
        bool fused = false;
        for (uint32_t g = 0; g < n_super; ++g) {
            const bool last_super = g == n_super - 1;
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
                                cb_act_k_window, cb_weight_k_window, in0_index, in1_index, 0, false, sb_w, sb_h, kb);
                            ++in0_index;
                            in1_index += win_n;
                        }
#endif
                        if (last_sb) {  // frees each window as soon as it is consumed (fifo_rd_ptr is exact here)
                            act_buf.pop_front(a_win);
                            weight_buf.pop_front(w_win);
                        }
                    }
                    if (fusable && last_super && first_sb && arrivals_landed(role)) {
                        // switch to the fused finish for every subblock of this super-block
                        fused = true;
                        if (has_hold) {
                            hold_buf.push_back(P);  // PACK: every earlier super-block is packed
                            hold_buf.wait_front(P);
                        }
                        if (role == ROLE_ADD_FWD || role == ROLE_ADD_BOTH) {
                            fwd_buf.wait_front(P);  // landed: returns at once
                        }
                        if (role == ROLE_ADD_BWD || role == ROLE_ADD_BOTH) {
                            bwd_buf.wait_front(P);
                        }
                        PACK((pack_reconfig_data_format(cb_finished)));
                        PACK((llk_pack_reconfig_l1_acc(0)));
                        fin_buf.reserve_back(P);
                    }
                    if (fused) {
                        finish_in_dest(role, has_hold && n_super > 1, sr * row_group + sc * sb_w, cols, sb_h, sb_w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    if (fused) {
                        // earlier row groups are pushed: offsets are relative to this row group
                        ckl::pack_subblock_row_strided(0, cb_finished, sc * sb_w, cols, sb_h, sb_w);
                    } else {
                        PACK((llk_pack_reconfig_l1_acc(g == 0 ? 0 : 1)));
                        ckl::pack_subblock_row_strided(
                            0, cb_partial_hold, sr * row_group + sc * sb_w, cols, sb_h, sb_w);
                    }
                    tile_regs_release();
                }
                if (fused) {
                    fin_buf.push_back(row_group);  // the streamed fin writer can store this row group now
                }
            }
        }
        if (fused) {
            if (rows * cols < P) {
                fin_buf.push_back(P - rows * cols);  // a ragged wave: the nominal slot stays whole
            }
            if (has_hold) {
                hold_buf.pop_front(P);
            }
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

    constexpr uint32_t NONE = 0xFFFFFFFF;
    uint32_t pending = NONE;
    for (uint32_t e = 0; e < num_entries; ++e) {
        const uint32_t word = get_arg_val<uint32_t>(roles_idx + e);
        const uint32_t role = word & 0xFF;
        const bool lag = (word >> 8) & 1;  // host: arrival expected late -> finish behind the next matmul
        const bool fusable = ((word >> 9) & 1) && !lag && pending == NONE;
        const bool fused = matmul_entry(e, fusable, role);
        if (pending != NONE) {
            finish_block<cb_partial_hold, cb_finished, cb_arrivals_fwd, cb_arrivals_bwd, P, fin_add_fpu>(
                get_arg_val<uint32_t>(roles_idx + pending) & 0xFF);
            pending = NONE;
        }
        if (lag) {
            pending = e;  // the add waits for its arrival behind the next entry's matmul
        } else if (!fused) {
            finish_block<cb_partial_hold, cb_finished, cb_arrivals_fwd, cb_arrivals_bwd, P, fin_add_fpu>(role);
        }
    }
}
