// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Bench-only raw-LLK variants of the transport add (same CT/RT args as the op kernel, plus CT arg 8 = cb_zero).
// MODE (define):
//   0  replica of the helper's per-block sequence: [add_init, add x n, (reuse_init, reuse-add x n)], pack.
//      2-input: init hoisted once (as the chain does).
//   1  3-input "acc pair": DEST is zero at acquire (packer clears the half; boot ZEROACC for the first use),
//      dest += p + a; dest += b + 0 (resident zero tile). ONE hoisted add_init(acc_to_dest) — no per-block
//      reinit, no DEST->srcA move.
//   2  3-input "copy + acc add": dest = copy(p); dest += a + b. Two inits per block (copy / add-acc).
//   3  as 1 but the pair order per block is p+a for all n tiles, then b+0 for all n tiles.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"

#ifndef MODE
#define MODE 0
#endif

#ifdef TRISC_UNPACK
// Unpack one bf16 tile of `cb` into srcA and publish a ZEROED srcB face alongside each srcA face
// (UNPACR_NOP ZEROSRC + SET_DVALID — the same srcB filler the stock unpack_A MOP issues), WITHOUT touching the
// programmed unpack-AB MOP. The FPU then runs the same ELWADD(acc_to_dest) MOP: dest += tile + 0.
// Mirrors _llk_unpack_AB_ (context acquire, base-address write, semaphore handshake) with the MOP replaced by
// its four-face expansion.
inline void unpack_a_zero_b(uint32_t cb, uint32_t tile) {
    const uint32_t id = get_operand_id(cb);
    const uint32_t addr = get_local_cb_interface(id).fifo_rd_ptr - 1 + get_local_cb_interface(id).fifo_page_size * tile;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    if (0 == unp_cfg_context) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = addr;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = addr;
    }
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
#pragma GCC unroll 4
    for (uint32_t f = 0; f < 4; ++f) {
        TTI_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_UNPACK
// Block unpack: n CONTIGUOUS tiles from the CB fronts under ONE unpacker config context (one base-address write,
// one semaphore handshake), the programmed per-tile MOP run n times back to back; the UNPACR Z counters keep
// incrementing face by face, so tile i+1 is read from the next 2 KiB page. zero_b: srcB is the ZEROSRC filler
// (no srcB L1 read) instead of the B CB.
template <bool zero_b>
inline void unpack_block(uint32_t cba, uint32_t cbb, uint32_t n) {
    const uint32_t addr_a = get_local_cb_interface(get_operand_id(cba)).fifo_rd_ptr - 1;
    const uint32_t addr_b = get_local_cb_interface(get_operand_id(cbb)).fifo_rd_ptr - 1;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(addr_a, addr_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    for (uint32_t i = 0; i < n; ++i) {
        if constexpr (zero_b) {
#pragma GCC unroll 4
            for (uint32_t f = 0; f < 4; ++f) {
                TTI_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
            }
        } else {
            ckernel::ckernel_template::run();
        }
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_MATH
// Block math: the programmed ELWADD MOP n times from DEST slot `first`; the MOP's end op resets only the A/B
// counters, so the DEST write counter walks on into the next 64-row tile slot.
inline void math_block(uint32_t first, uint32_t n) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(first);
    for (uint32_t i = 0; i < n; ++i) {
        ckernel::ckernel_template::run();
    }
    math::clear_dst_reg_addr();
}
#endif

// dest[idst] += tile(cb, itile) + 0, under the hoisted ELWADD(acc_to_dest) MOP.
ALWI void add_tile_acc_zero_b(uint32_t cb, uint32_t itile, uint32_t idst) {
    UNPACK((unpack_a_zero_b(cb, itile)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(cb, cb, idst, true)));
}

void kernel_main() {
    constexpr uint32_t cb_p = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_b = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(3);
    constexpr uint32_t has_a = get_compile_time_arg_val(4);
    constexpr uint32_t has_b = get_compile_time_arg_val(5);
    constexpr uint32_t add_block = get_compile_time_arg_val(6);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(8);
    const uint32_t num_segs = get_arg_val<uint32_t>(1);
    static_assert(has_a, "bench: relay/final with arrival A");
    constexpr bool three = has_b;

    compute_kernel_hw_startup(cb_p, cb_a, cb_sum);
    if constexpr (!three || MODE == 1 || MODE == 3 || MODE == 4 || MODE == 5) {
        add_init(cb_p, cb_a, /*acc_to_dest=*/three);
    }
    if constexpr (three && (MODE == 4 || MODE == 5)) {
        MATH((TTI_ZEROACC(p_zeroacc::CLR_ALL, 1, 0, ADDR_MOD_1, 0)));
    }
    if constexpr (three && (MODE == 1 || MODE == 3)) {
        // Both DEST halves to zero once (afterwards the packer clears each half it releases), then pack one
        // zero tile into cb_zero (the + 0 operand of the second pair).
        MATH((TTI_ZEROACC(p_zeroacc::CLR_ALL, 1, 0, ADDR_MOD_1, 0)));
        cb_reserve_back(cb_zero, 1);
        tile_regs_acquire();
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_zero);
        tile_regs_release();
        cb_push_back(cb_zero, 1);
        cb_wait_front(cb_zero, 1);
    }

    for (uint32_t s = 0; s < num_segs; ++s) {
        for (uint32_t off = 0; off < seg_tiles; off += add_block) {
            const uint32_t n = (seg_tiles - off) < add_block ? (seg_tiles - off) : add_block;
            cb_wait_front(cb_p, n);
            cb_wait_front(cb_a, n);
            if constexpr (three) {
                cb_wait_front(cb_b, n);
            }
            tile_regs_acquire();
            if constexpr (MODE == 5) {
                UNPACK((unpack_block<false>(cb_p, cb_a, n)));
                MATH((math_block(0, n)));
                if constexpr (three) {
                    UNPACK((unpack_block<true>(cb_b, cb_b, n)));
                    MATH((math_block(0, n)));
                }
            } else if constexpr (!three) {
                for (uint32_t j = 0; j < n; ++j) {
                    add_tiles(cb_p, cb_a, j, j, j);
                }
            } else if constexpr (MODE == 0) {
                add_init(cb_p, cb_a);
                for (uint32_t j = 0; j < n; ++j) {
                    add_tiles(cb_p, cb_a, j, j, j);
                }
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_b);
                for (uint32_t j = 0; j < n; ++j) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_b, j, j);
                }
            } else if constexpr (MODE == 1) {
                for (uint32_t j = 0; j < n; ++j) {
                    add_tiles(cb_p, cb_a, j, j, j);
                    add_tiles(cb_b, cb_zero, j, 0, j);
                }
            } else if constexpr (MODE == 4) {
                for (uint32_t j = 0; j < n; ++j) {
                    add_tiles(cb_p, cb_a, j, j, j);
                    add_tile_acc_zero_b(cb_b, j, j);
                }
            } else if constexpr (MODE == 3) {
                for (uint32_t j = 0; j < n; ++j) {
                    add_tiles(cb_p, cb_a, j, j, j);
                }
                for (uint32_t j = 0; j < n; ++j) {
                    add_tiles(cb_b, cb_zero, j, 0, j);
                }
            } else if constexpr (MODE == 2) {
                copy_init(cb_p);
                for (uint32_t j = 0; j < n; ++j) {
                    copy_tile(cb_p, j, j);
                }
                add_init(cb_a, cb_b, true);
                for (uint32_t j = 0; j < n; ++j) {
                    add_tiles(cb_a, cb_b, j, j, j);
                }
            }
            tile_regs_commit();
            cb_pop_front(cb_p, n);
            cb_pop_front(cb_a, n);
            if constexpr (three) {
                cb_pop_front(cb_b, n);
            }
            cb_reserve_back(cb_sum, n);
            tile_regs_wait();
#ifndef DIAG_NOPACK
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_sum);
            }
#endif
            tile_regs_release();
            cb_push_back(cb_sum, n);
        }
    }
}
