// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter: matmul-core compute (TRISCs). fp32 DEST + full sync (8-tile subblocks).
//
// Per entry (one block): matmul_entry, its rows x cols x Kt in K-windows of kb, into ONE reserved fp32 slot of
// cb_partial_hold. DEST-resident K: the entry's K-windows are grouped into super-blocks of S windows (CT arg; an
// operand fill uses 1, since its windows arrive at DRAM pace); subblock-outer / K-inner, each output subblock
// accumulates a whole super-block in fp32 DEST and is packed once per super-block (plain pack for the first, packer
// L1-acc after). The entry's tiles are dense at the front of the P-tile slot (row width = cols); the W window keeps
// its nominal row stride (win_n). The windows of a super-block stay fronted until its last subblock; a wrapped window
// is addressed by pointing the UNPACK thread's fifo_rd_ptr at it, and the last subblock pops each window right after
// consuming it, so the read pointer is exact again at the end.
//
// Fused finish: an entry's finish folds into the matmul's last super-block. Each output subblock accumulates its
// last-super-block matmul in fp32 DEST; the earlier super-blocks' fp32 hold is added in DEST (DEST += hold + 0: an FPU
// add accumulated onto DEST against a zero tile), then the arrival(s) by DEST_TO_SRCA reuse adds (DEST -> srcA,
// arrival -> srcB, FPU add into fp32 DEST), and the subblock is packed bf16 straight into cb_finished, pushed per
// subblock row group (the streamed fin writer stores a row group while the next one computes). The arrival is
// transport-bound (it lands about when this entry's matmul ends), so waiting for it inside the matmul would stall the
// FPU: after the first subblock of the last super-block has accumulated, UNPACK checks (non-blocking CB credit read)
// whether the arrival(s) landed and broadcasts the answer to MATH and PACK through the TRISC mailboxes, so all three
// threads branch alike.
//   landed (or no arrival: chain start)  -> every subblock: DEST += hold + 0, += arrival(s) (reuse), pack bf16
//   not landed (late)                    -> every subblock: DEST += hold + 0, pack fp32 in place over its hold
//                                           tiles; finish_block adds the arrival(s) after the matmul
// A lagged entry (its arrival is produced at the same entry index upstream) and the entry right behind it also take
// finish_block, behind the next entry's matmul (cb_finished's FIFO order).
// DETERMINISM: both branches run the identical op sequence on identical values. The late branch stores the
// (DEST + hold) sum exactly (fp32 pack) and finish_block reads it back through srcA, the same fp32 -> srcA conversion
// the fused branch's DEST_TO_SRCA move applies, then the same FPU adds with the same operands in the same registers.
// So the bits never depend on the run-time arrival timing.
//
// The matmul and the fused finish use raw LLK calls: the kernel_lib matmul and eltwise helpers own the DEST
// lifecycle (acquire, op, pack, release per batch), and neither can keep a subblock in DEST across K-windows or run an
// add on a DEST the caller already filled.

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t ROLE_COPY = 0, ROLE_ADD_FWD = 1, ROLE_ADD_BWD = 2, ROLE_ADD_BOTH = 3;

// Pack an h x w DEST subblock (DEST tile r * w + c) to tile r * row_stride + col_base + c of the reserved region.
ALWI void pack_subblock_row_strided(
    uint32_t dst0, uint32_t cb, uint32_t col_base, uint32_t row_stride, uint32_t h, uint32_t w) {
    for (uint32_t r = 0; r < h; r++) {
        for (uint32_t c = 0; c < w; c++) {
            pack_tile<true>(dst0 + r * w + c, cb, r * row_stride + col_base + c);
        }
    }
}

// The separate finish (lagged, post-lag and late entries): copy, hold + arrival, or hold + fwd + bwd, FPU adds into
// fp32 DEST; the hold reaches DEST through srcA (no UnpackToDestFp32), as in the fused finish (see DETERMINISM).
template <uint32_t cb_hold, uint32_t cb_finished, uint32_t cb_arr_fwd, uint32_t cb_arr_bwd, uint32_t P>
ALWI void finish_block(
    uint32_t role, DataflowBuffer& hold, DataflowBuffer& fwd, DataflowBuffer& bwd, DataflowBuffer& out) {
    DeviceZoneScopedN("FIN");
    using namespace ckl;
    constexpr auto shape = IterationShape::tiles(P);
    if (role == ROLE_COPY) {
        copy<input(cb_hold), output(cb_finished)>(shape, hold, out);
    } else if (role == ROLE_ADD_FWD) {
        add<input(cb_hold), input(cb_arr_fwd), output(cb_finished)>(shape, hold, fwd, out);
    } else if (role == ROLE_ADD_BWD) {
        add<input(cb_hold), input(cb_arr_bwd), output(cb_finished)>(shape, hold, bwd, out);
    } else {
        eltwise_chain(
            shape,
            BinaryFpu<BinaryFpuOp::Add, input(cb_hold), input(cb_arr_fwd)>{hold, fwd},
            DestReuseBinary<BinaryFpuOp::Add, input(cb_arr_bwd), DestReuseType::DEST_TO_SRCA>{bwd},
            PackTile<output(cb_finished)>{out});
    }
}

void kernel_main() {
    constexpr uint32_t cb_act_k_window = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_k_window = get_compile_time_arg_val(1);
    constexpr uint32_t cb_partial_hold = get_compile_time_arg_val(2);
    constexpr uint32_t cb_finished = get_compile_time_arg_val(3);
    constexpr uint32_t cb_arrivals_fwd = get_compile_time_arg_val(4);
    constexpr uint32_t cb_arrivals_bwd = get_compile_time_arg_val(5);
    constexpr uint32_t win_n = get_compile_time_arg_val(6);  // W window row stride (tiles): an entry's nominal width
    constexpr uint32_t kb = get_compile_time_arg_val(7);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t P = get_compile_time_arg_val(9);             // slot tiles (the largest entry)
    constexpr uint32_t num_entries = get_compile_time_arg_val(10);  // G
    constexpr uint32_t super_block = get_compile_time_arg_val(11);  // K-windows per DEST-resident pack (divides nkb)
    constexpr uint32_t cb_zero = get_compile_time_arg_val(12);      // one bf16 zero tile (the writer fills it)
    constexpr uint32_t num_chunks = get_compile_time_arg_val(13);   // the entry walk runs once per output chunk
    constexpr uint32_t n_inner = get_compile_time_arg_val(14);      // chunks per outer (resident-axis) index
    constexpr bool resident = get_compile_time_arg_val(15) != 0;
    static_assert(super_block >= 1 && num_k_blocks % super_block == 0, "super-block must divide the K-blocks");
    // runtime args: per entry rows | cols << 8 | sb_h << 16 | sb_w << 24, then per entry role | lag << 8
    constexpr uint32_t roles_idx = num_entries;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_k_window, cb_weight_k_window, cb_partial_hold);

    CircularBuffer act_buf(cb_act_k_window), weight_buf(cb_weight_k_window), hold_buf(cb_partial_hold);
    CircularBuffer fin_buf(cb_finished), fwd_buf(cb_arrivals_fwd), bwd_buf(cb_arrivals_bwd);
    // the same buffers, bound to finish_block's eltwise chains
    DataflowBuffer hold_dfb(cb_partial_hold), fin_dfb(cb_finished), fwd_dfb(cb_arrivals_fwd), bwd_dfb(cb_arrivals_bwd);
    cb_wait_front(cb_zero, 1);  // the writer's zero tile; never popped

    // UNPACK: have the entry's arrival(s) landed (non-blocking)? Broadcast to MATH and PACK by mailbox.
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

    // In-DEST finish ops of one output subblock after its last-super-block matmul: DEST_TO_SRCA reuse adds of
    // `hold` (when set) then of the arrivals (when with_arrivals). Tiles sit row-strided at col_base + r * cols (the
    // slot's dense entry layout); DEST tile r * w + c. Matmul state on entry and exit: srcA = W, srcB = A, matmul MOP.
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
                if (role == ROLE_ADD_FWD || role == ROLE_ADD_BOTH) {
                    reuse_add(cb_arrivals_fwd);
                }
                if (role == ROLE_ADD_BWD || role == ROLE_ADD_BOTH) {
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
        DeviceZoneScopedN("MM");
        const uint32_t geom = get_arg_val<uint32_t>(e);
        const uint32_t rows = geom & 0xFF, cols = (geom >> 8) & 0xFF;
        const uint32_t sb_h = (geom >> 16) & 0xFF, sb_w = geom >> 24;
        const uint32_t n_sb_r = rows / sb_h, n_sb_c = cols / sb_w;
        const uint32_t a_win = rows * kb, w_win = win_n * kb;  // window tiles, as the reader pushes them
        const uint32_t row_group = sb_h * cols;
        reconfig_data_format(cb_weight_k_window, cb_act_k_window);  // matmul: srcA = in1, srcB = in0
        PACK((pack_reconfig_data_format(cb_partial_hold)));
        ckernel::matmul_block_init(cb_act_k_window, cb_weight_k_window, false, sb_w, sb_h, kb);
        const uint32_t S = fill ? 1 : super_block;  // an operand fill arrives at DRAM pace: pack per window
        const uint32_t n_super = num_k_blocks / S;
        const bool has_arrival = role != ROLE_COPY;
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
                        // The window's kb inner-dim steps: in0 rows are kb tiles wide, in1 rows win_n.
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
                            if (role == ROLE_ADD_FWD || role == ROLE_ADD_BOTH) {
                                fwd_buf.wait_front(P);  // landed: returns at once
                            }
                            if (role == ROLE_ADD_BWD || role == ROLE_ADD_BOTH) {
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
        if (fused) {
            if (rows * cols < P) {
                fin_buf.push_back(P - rows * cols);  // a narrower entry: the nominal slot stays whole
            }
            if (fusable && hold_prev) {
                hold_buf.pop_front(P);
            } else if (use_hold) {  // a single-super-block arrival entry reserved a slot it did not need
                hold_buf.push_back(P);
                hold_buf.wait_front(P);
                hold_buf.pop_front(P);
            }
            if (role == ROLE_ADD_FWD || role == ROLE_ADD_BOTH) {
                fwd_buf.pop_front(P);
            }
            if (role == ROLE_ADD_BWD || role == ROLE_ADD_BOTH) {
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
            const bool lag = (word >> 8) & 1;  // host: arrival expected late -> finish behind the next matmul
            const bool fusable = !lag && pending == NONE;
            const bool fused = matmul_entry(e, chunk_fill && e == 0, fusable, role);
            if (pending != NONE) {
                finish_block<cb_partial_hold, cb_finished, cb_arrivals_fwd, cb_arrivals_bwd, P>(
                    get_arg_val<uint32_t>(roles_idx + pending) & 0xFF, hold_dfb, fwd_dfb, bwd_dfb, fin_dfb);
                pending = NONE;
            }
            if (lag) {
                pending = e;  // the add waits for its arrival behind the next entry's matmul
            } else if (!fused) {
                finish_block<cb_partial_hold, cb_finished, cb_arrivals_fwd, cb_arrivals_bwd, P>(
                    role, hold_dfb, fwd_dfb, bwd_dfb, fin_dfb);
            }
        }
    }
}
