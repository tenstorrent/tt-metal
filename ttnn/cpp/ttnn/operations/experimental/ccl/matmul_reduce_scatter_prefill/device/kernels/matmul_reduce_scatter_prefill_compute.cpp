// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter_prefill — matmul-core compute (TRISCs). fp32 DEST + full sync (8-tile subblocks).
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
// QB2, traced, compute floor with the DRAM reads and multicast stubbed): helper 112.8 us, this path at S=1 111.6, at
// S=2 109.1; whole op 126.4 -> 123.5 us; GLM 200.3 -> 194.6; MiMo 219.2 -> 208.0. The windows of a super-block stay
// fronted until its last subblock; a wrapped window is addressed by pointing the UNPACK thread's fifo_rd_ptr at it, and
// the last subblock pops each window right after consuming it, so the read pointer is exact again at the end. Then
// finish_block for the lagged entry (or this one): copy (chain start), hold + arrival (relay / fin with one upstream),
// or hold + fwd + bwd (fin with two upstreams): FPU adds (hold + arrival, then the second arrival through DEST reuse).
// (The SFPU-add finish variant was deleted in Perf 2: the deferred finish must repeat the fused ops.) Every
// cross-device add runs in fp32 DEST; the result is packed to bf16 into cb_finished.

// FUSED FINISH. The finish of an entry folds into the
// matmul's LAST super-block: each output subblock accumulates its last-super-block matmul in fp32 DEST, the earlier
// super-blocks' fp32 hold is added IN DEST (DEST += hold + 0: FPU add accumulated onto DEST against a zero tile) and
// the arrival(s) by DEST_TO_SRCA reuse adds (DEST -> srcA, arrival -> srcB, FPU add into fp32 DEST), and the subblock
// is packed bf16 straight into cb_finished, pushed per subblock row group (the streamed fin writer stores a row group
// while the next one computes). The last super-block's fp32 L1-acc pack, the hold re-read and the separate finish pass
// disappear. WHEN: the arrival is transport-bound (it lands about when this entry's matmul ends), so waiting for it
// inside the matmul stalls the FPU (focus +4 us when every entry waits). After the first subblock of the last
// super-block has accumulated, UNPACK checks (non-blocking CB credit read) whether the arrival(s) landed and broadcasts
// the answer to MATH and PACK through the TRISC mailboxes (the get_tile_address pattern), so all three threads branch
// alike.
//   landed (or no arrival: chain start)  -> every subblock: DEST += hold + 0, += arrival(s) (reuse), pack bf16
//   not landed (late)                    -> every subblock: DEST += hold + 0, pack fp32 IN PLACE over its hold
//                                           tiles; PACK then drains and signals UNPACK (the slot was pushed
//                                           before these packs), and the separate finish adds the arrival(s)
// DETERMINISM: both branches run the identical op sequence on identical values. The late branch stores the
// (DEST + hold) sum exactly (fp32 pack) and the finish reads it back through srcA, which is the same fp32 -> srcA
// conversion the fused branch's DEST_TO_SRCA move applies, then the same FPU adds with the same operands in the
// same registers. So the bits never depend on the run-time arrival timing (test_regression::test_deterministic).
// A lagged entry (arrival produced at the same entry index upstream) and the entry right behind it keep the old
// path (finish behind the next matmul; cb_finished's FIFO order).
//
// RAW LLK — measured helper bypass (fused finish; do NOT "fix" back to the finish_block helper chains for the fused
// entries): compute_kernel_lib::add / copy / eltwise_chain own the DEST lifecycle (acquire, op, pack, release per tile
// batch) and read their operands from the CB front. The fused finish must (1) run its adds on a DEST the caller
// already acquired and filled with a matmul subblock, without acquiring or packing, (2) accumulate onto that DEST
// (ELWADD acc_to_dest, DEST += hold + 0) and chain DEST_TO_SRCA reuse adds of tiles addressed row-strided inside a
// fronted slot, (3) pack row-strided into a caller-reserved cb_finished region pushed per row group, and (4) let the
// late branch pack in place into an already-pushed slot (packer write pointer rewound). No helper mode expresses an
// op on caller-owned, pre-filled DEST. The run-time decision is a non-blocking CB credit read on UNPACK broadcast by
// the TRISC mailboxes (no helper exists for either). Measured (QB2 1x4 ring, traced, max over chips): focus
// 119.0-119.1 -> 116.6-117.2 us, GLM 185.7 -> 182.1, MiMo 205.3 -> 204.4-205.0.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/kernels/matmul_reduce_scatter_prefill_args.hpp"

namespace ckl = compute_kernel_lib;
namespace mrs = matmul_reduce_scatter_prefill;

// Pack an (h x w) DEST sub-block to absolute row-major positions in a CB: DEST[dst_start_idx .. +h*w-1] row-first,
// row r at r * row_stride + col_base. The caller reserved >= h * row_stride tiles and owns the matching push.
ALWI void pack_subblock_row_strided(
    uint32_t dst_start_idx, uint32_t pack_target_id, uint32_t col_base, uint32_t row_stride, uint32_t h, uint32_t w) {
    for (uint32_t r = 0; r < h; r++) {
        const uint32_t row_base = r * row_stride + col_base;
        for (uint32_t c = 0; c < w; c++) {
            pack_tile<true>(dst_start_idx + r * w + c, pack_target_id, row_base + c);
        }
    }
}

// The finish's buffers, bound once by kernel_main and passed to every helper call.
struct FinishBuffers {
    DataflowBuffer& hold;
    DataflowBuffer& finished;
    DataflowBuffer& arr_fwd;
    DataflowBuffer& arr_bwd;
};

template <uint32_t cb_hold, uint32_t cb_finished, uint32_t cb_arr_fwd, uint32_t cb_arr_bwd, uint32_t P>
ALWI void finish_block(uint32_t role, const FinishBuffers& buf) {
    using namespace ckl;
    constexpr auto shape = IterationShape::tiles(P);
    // FPU adds into fp32 DEST; the hold reaches DEST through srcA (no UnpackToDestFp32). This separate finish runs only
    // for lagged / post-lag / late entries and must repeat the fused finish's op sequence exactly (determinism).
    if (role == mrs::role::COPY) {
        copy<input(cb_hold), output(cb_finished)>(shape, buf.hold, buf.finished);
    } else if (role == mrs::role::ADD_FWD) {
        add<input(cb_hold), input(cb_arr_fwd), output(cb_finished)>(shape, buf.hold, buf.arr_fwd, buf.finished);
    } else if (role == mrs::role::ADD_BWD) {
        add<input(cb_hold), input(cb_arr_bwd), output(cb_finished)>(shape, buf.hold, buf.arr_bwd, buf.finished);
    } else {
        eltwise_chain(
            shape,
            BinaryFpu<BinaryFpuOp::Add, input(cb_hold), input(cb_arr_fwd)>{buf.hold, buf.arr_fwd},
            DestReuseBinary<BinaryFpuOp::Add, input(cb_arr_bwd), DestReuseType::DEST_TO_SRCA>{buf.arr_bwd},
            PackTile<output(cb_finished)>{buf.finished});
    }
}

void kernel_main() {
    using namespace mrs::compute_ct;
    constexpr uint32_t cb_act_k_window = get_compile_time_arg_val(CB_ACT_K_WINDOW);
    constexpr uint32_t cb_weight_k_window = get_compile_time_arg_val(CB_WEIGHT_K_WINDOW);
    constexpr uint32_t cb_partial_hold = get_compile_time_arg_val(CB_PARTIAL_HOLD);
    constexpr uint32_t cb_finished = get_compile_time_arg_val(CB_FINISHED);
    constexpr uint32_t cb_arrivals_fwd = get_compile_time_arg_val(CB_ARRIVALS_FWD);
    constexpr uint32_t cb_arrivals_bwd = get_compile_time_arg_val(CB_ARRIVALS_BWD);
    constexpr uint32_t win_n =
        get_compile_time_arg_val(WIN_N);  // W window row stride (tiles): one wave's nominal width
    constexpr uint32_t kb = get_compile_time_arg_val(KB);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(NUM_K_BLOCKS);
    constexpr uint32_t P = get_compile_time_arg_val(mrs::compute_ct::P);     // slot tiles (the largest wave)
    constexpr uint32_t num_entries = get_compile_time_arg_val(NUM_ENTRIES);  // G x num_waves
    constexpr uint32_t num_waves = get_compile_time_arg_val(NUM_WAVES);
    constexpr uint32_t super_block = get_compile_time_arg_val(SUPER_BLOCK);  // K-windows per DEST-resident pack
    constexpr uint32_t cb_zero = get_compile_time_arg_val(CB_ZERO);          // one bf16 zero tile (the writer fills it)
    constexpr uint32_t num_chunks = get_compile_time_arg_val(NUM_CHUNKS);  // the entry walk runs once per output chunk
    constexpr uint32_t n_inner = get_compile_time_arg_val(N_INNER);        // chunks per outer (resident-axis) index
    constexpr bool resident = get_compile_time_arg_val(RESIDENT) != 0;
    static_assert(super_block >= 1 && num_k_blocks % super_block == 0, "super-block must divide the K-blocks");
    // runtime args: per wave rows | cols << 8 | sb_h << 16 | sb_w << 24, then per entry role | lag << 8
    constexpr uint32_t roles_idx = num_waves;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_k_window, cb_weight_k_window, cb_partial_hold);

    CircularBuffer act_buf(cb_act_k_window), weight_buf(cb_weight_k_window), hold_buf(cb_partial_hold);
    CircularBuffer fin_buf(cb_finished), fwd_buf(cb_arrivals_fwd), bwd_buf(cb_arrivals_bwd);
    DataflowBuffer hold_dfb(cb_partial_hold), fin_dfb(cb_finished), fwd_dfb(cb_arrivals_fwd), bwd_dfb(cb_arrivals_bwd);
    const FinishBuffers finish_bufs{hold_dfb, fin_dfb, fwd_dfb, bwd_dfb};
    cb_wait_front(cb_zero, 1);  // the writer's zero tile; never popped

    // UNPACK: have the entry's arrival(s) landed (non-blocking)? Broadcast to MATH and PACK by mailbox.
    auto arrivals_landed = [&](uint32_t role) -> bool {
        uint32_t ready = 1;
        UNPACK(({
            auto landed = [&](uint32_t cb) {
                const uint16_t rec = (uint16_t)reg_read((uint32_t)get_cb_tiles_received_ptr(cb));
                return (uint16_t)(rec - (uint16_t)get_local_cb_interface(cb).tiles_acked) >= P;
            };
            if (role == mrs::role::ADD_FWD || role == mrs::role::ADD_BOTH) {
                ready &= landed(cb_arrivals_fwd) ? 1u : 0u;
            }
            if (role == mrs::role::ADD_BWD || role == mrs::role::ADD_BOTH) {
                ready &= landed(cb_arrivals_bwd) ? 1u : 0u;
            }
            mailbox_write(ckernel::ThreadId::MathThreadId, ready);
            mailbox_write(ckernel::ThreadId::PackThreadId, ready);
        }));
        MATH((ready = mailbox_read(ckernel::ThreadId::UnpackThreadId)));
        PACK((ready = mailbox_read(ckernel::ThreadId::UnpackThreadId)));
        return ready != 0;
    };

    // In-DEST finish ops of one output subblock after its last-super-block matmul: DEST_TO_SRCA reuse adds of
    // `hold` (when set) then of the arrivals (when with_arrivals). Tiles sit row-strided at col_base + r * cols (the
    // slot's dense wave layout); DEST tile r * w + c. Matmul state on entry and exit: srcA = W, srcB = A, matmul MOP.
    auto finish_in_dest =
        [&](uint32_t role, bool hold, bool with_arrivals, uint32_t col_base, uint32_t cols, uint32_t h, uint32_t w) {
            uint32_t srcb = cb_act_k_window;
            bool any = false;
            auto reuse_add = [&](uint32_t cb) {
                // srcA carries DEST at the hold's (fp32 -> srcA) width, exactly as the hold reaches srcA when unpacked
                reconfig_data_format(any ? cb_partial_hold : cb_weight_k_window, cb_partial_hold, srcb, cb);
                srcb = cb;
                any = true;
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb);
                for (uint32_t r = 0; r < h; ++r) {
                    for (uint32_t c = 0; c < w; ++c) {
                        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                            cb, r * cols + col_base + c, r * w + c);
                    }
                }
            };
            if (hold) {
                // DEST += hold + 0: one FPU add accumulated onto DEST (srcA = hold, srcB = the zero tile), no DEST move
                reconfig_data_format(cb_weight_k_window, cb_partial_hold, cb_act_k_window, cb_zero);
                add_init(cb_partial_hold, cb_zero, true /* acc_to_dest */);
                for (uint32_t r = 0; r < h; ++r) {
                    for (uint32_t c = 0; c < w; ++c) {
                        add_tiles(cb_partial_hold, cb_zero, r * cols + col_base + c, 0, r * w + c);
                    }
                }
                srcb = cb_zero;
                any = true;
            }
            if (with_arrivals) {
                if (role == mrs::role::ADD_FWD || role == mrs::role::ADD_BOTH) {
                    reuse_add(cb_arrivals_fwd);
                }
                if (role == mrs::role::ADD_BWD || role == mrs::role::ADD_BOTH) {
                    reuse_add(cb_arrivals_bwd);
                }
            }
            if (any) {
                reconfig_data_format(cb_partial_hold, cb_weight_k_window, srcb, cb_act_k_window);
                ckernel::matmul_block_init(cb_act_k_window, cb_weight_k_window, false, w, h, kb);
            }
        };

    // One entry's matmul. fusable: its finish folds into the last super-block (an arrival entry: only if the
    // arrival landed by the end of that super-block's first subblock). Returns whether the finish is complete.
    auto matmul_entry = [&](uint32_t e, bool fill, bool fusable, uint32_t role) -> bool {
        const uint32_t wave = get_arg_val<uint32_t>(e % num_waves);
        const uint32_t rows = wave & 0xFF, cols = (wave >> 8) & 0xFF;
        const uint32_t sb_h = (wave >> 16) & 0xFF, sb_w = wave >> 24;
        const uint32_t n_sb_r = rows / sb_h, n_sb_c = cols / sb_w;
        const uint32_t a_win = rows * kb, w_win = win_n * kb;  // window tiles, as the reader pushes them
        const uint32_t row_group = sb_h * cols;
        reconfig_data_format(cb_weight_k_window, cb_act_k_window);  // matmul: srcA = in1, srcB = in0
        PACK((pack_reconfig_data_format(cb_partial_hold)));
        ckernel::matmul_block_init(cb_act_k_window, cb_weight_k_window, false, sb_w, sb_h, kb);
        const uint32_t S = fill ? 1 : super_block;  // an operand fill arrives at DRAM pace: pack per window
        const uint32_t n_super = num_k_blocks / S;
        const bool has_arrival = role != mrs::role::COPY;
        const bool hold_prev = n_super > 1;  // earlier super-blocks to add in the last one
        // a fused chain start with one super-block never touches the hold
        const bool use_hold = !(fusable && !has_arrival && !hold_prev);
        if (use_hold) {
            hold_buf.reserve_back(P);  // one fp32 slot per entry
        }
        bool fused = fusable && !has_arrival;  // an arrival entry decides at its last super-block
        bool late = false;                     // fusable arrival entry whose arrival had not landed
        uint32_t slot_wr = 0;                  // PACK: the hold slot's write pointer (the late branch packs in place)
        for (uint32_t g = 0; g < n_super; ++g) {
            const bool last_super = g == n_super - 1;
            if (fusable && last_super && hold_prev) {
                PACK((slot_wr = get_local_cb_interface(cb_partial_hold).fifo_wr_ptr));
                hold_buf.push_back(P);  // every earlier super-block is packed (PACK program order)
                hold_buf.wait_front(P);
            }
            if (fused && last_super) {  // chain start: known up front
                PACK((pack_reconfig_data_format(cb_finished)));
                PACK((llk_pack_reconfig_l1_acc(0)));
                fin_buf.reserve_back(P);
            }
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
                        for (uint32_t kk = 0; kk < kb; ++kk) {
                            ckernel::matmul_block(
                                cb_act_k_window, cb_weight_k_window, in0_index, in1_index, 0, false, sb_w, sb_h, kb);
                            ++in0_index;
                            in1_index += win_n;
                        }
                        if (last_sb) {  // frees each window as soon as it is consumed (fifo_rd_ptr is exact here)
                            act_buf.pop_front(a_win);
                            weight_buf.pop_front(w_win);
                        }
                    }
                    if (fusable && has_arrival && last_super && first_sb) {
                        if (arrivals_landed(role)) {
                            fused = true;
                            if (role == mrs::role::ADD_FWD || role == mrs::role::ADD_BOTH) {
                                fwd_buf.wait_front(P);  // landed: returns at once
                            }
                            if (role == mrs::role::ADD_BWD || role == mrs::role::ADD_BOTH) {
                                bwd_buf.wait_front(P);
                            }
                            PACK((pack_reconfig_data_format(cb_finished)));
                            PACK((llk_pack_reconfig_l1_acc(0)));
                            fin_buf.reserve_back(P);
                        } else {
                            late = true;
                            PACK((llk_pack_reconfig_l1_acc(0)));  // the (DEST + hold) sum overwrites the hold
                        }
                    }
                    const uint32_t col_base = sr * row_group + sc * sb_w;
                    if (last_super && (fused || late)) {
                        finish_in_dest(role, hold_prev, fused, col_base, cols, sb_h, sb_w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    if (last_super && fused) {
                        // earlier row groups are pushed: offsets are relative to this row group
                        pack_subblock_row_strided(0, cb_finished, sc * sb_w, cols, sb_h, sb_w);
                    } else if (last_super && late && hold_prev) {
                        // in place over this subblock's hold tiles (the slot is already pushed)
                        PACK(({
                            auto& ch = get_local_cb_interface(cb_partial_hold);
                            const uint32_t wr = ch.fifo_wr_ptr;
                            ch.fifo_wr_ptr = slot_wr;
                            pack_subblock_row_strided(0, cb_partial_hold, col_base, cols, sb_h, sb_w);
                            ch.fifo_wr_ptr = wr;
                        }));
                    } else {
                        PACK((llk_pack_reconfig_l1_acc(g == 0 || late ? 0 : 1)));
                        pack_subblock_row_strided(0, cb_partial_hold, col_base, cols, sb_h, sb_w);
                    }
                    tile_regs_release();
                }
                if (last_super && fused) {
                    fin_buf.push_back(row_group);  // the streamed fin writer can store this row group now
                }
            }
        }
        if (late && hold_prev) {
            // The slot was pushed before the in-place packs above, so no CB push orders them before the finish's hold
            // reads, and UNPACK (gated by srcA/srcB, not DEST) could read a tile the last pack is still writing. PACK
            // drains the packer (the LLK's STALLWAIT + tensix_sync idiom) and hands UNPACK a token it waits on.
            PACK(({
                TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);
                ckernel::tensix_sync();
                mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
            }));
            UNPACK(((void)mailbox_read(ckernel::ThreadId::PackThreadId)));
        }
        if (fused) {
            if (rows * cols < P) {
                fin_buf.push_back(P - rows * cols);  // a ragged wave: the nominal slot stays whole
            }
            if (fusable && hold_prev) {
                hold_buf.pop_front(P);
            } else if (use_hold) {  // a single-super-block arrival entry reserved a slot it did not need
                hold_buf.push_back(P);
                hold_buf.wait_front(P);
                hold_buf.pop_front(P);
            }
            if (role == mrs::role::ADD_FWD || role == mrs::role::ADD_BOTH) {
                fwd_buf.pop_front(P);
            }
            if (role == mrs::role::ADD_BWD || role == mrs::role::ADD_BOTH) {
                bwd_buf.pop_front(P);
            }
        } else if (!(fusable && hold_prev)) {  // (a fusable entry with earlier super-blocks pushed its slot already)
            hold_buf.push_back(P);
        }
        PACK((llk_pack_reconfig_l1_acc(0)));  // the finish chains pack plainly
        return fused;
    };

    constexpr uint32_t NONE = 0xFFFFFFFF;
    uint32_t pending = NONE;
    for (uint32_t c = 0; c < num_chunks; ++c) {
        // the resident operand reloads in the first entry of a chunk whose outer index changed (always chunk 0)
        const bool chunk_fill = c == 0 || (resident && c % n_inner == 0);
        for (uint32_t e = 0; e < num_entries; ++e) {
            const uint32_t word = get_arg_val<uint32_t>(roles_idx + e);
            const uint32_t role = word & 0xFF;
            const bool lag =
                (word >> mrs::role::LAG_SHIFT) & 1;  // host: arrival expected late -> finish behind the next matmul
            const bool fusable = !lag && pending == NONE;
            const bool fused = matmul_entry(e, chunk_fill && e == 0, fusable, role);
            if (pending != NONE) {
                finish_block<cb_partial_hold, cb_finished, cb_arrivals_fwd, cb_arrivals_bwd, P>(
                    get_arg_val<uint32_t>(roles_idx + pending) & 0xFF, finish_bufs);
                pending = NONE;
            }
            if (lag) {
                pending = e;  // the add waits for its arrival behind the next entry's matmul
            } else if (!fused) {
                finish_block<cb_partial_hold, cb_finished, cb_arrivals_fwd, cb_arrivals_bwd, P>(role, finish_bufs);
            }
        }
    }
}
