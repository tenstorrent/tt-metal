// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — transport-core TRISC: relay_add_block / final_add_block (fast path).
//
// cb_xport_sum = cb_xport_partial + arrival A [+ arrival B], accumulated in fp32 DEST (fp32_dest_acc_en=True)
// and packed bf16. Same CT/RT args and CB contract as the eltwise_chain version; segments are still completed
// one at a time (no cross-segment lookahead) and never straddle the CB wrap.
//
// RAW LLK (bypasses compute_kernel_lib::eltwise_chain BinaryFpu / DestReuseBinary / PackTile for the add walk;
// the copy-through walk still uses the helper). Mechanism:
//  * one ELWADD(acc_to_dest) init for the whole walk — DEST is zero at every acquire (boot ZEROACC, then the
//    packer clears each half it releases), so  dest += p + a ; dest += b + 0  is the 3-input sum with ONE
//    math MOP and ONE unpack-AB MOP. The baseline re-ran add_init + add_reuse_dest_init every DEST block and
//    moved DEST -> srcA (tf32) face by face; this keeps the running sum in fp32 DEST (more precise).
//  * "+ 0": srcB is published as the unpacker's ZEROSRC filler (UNPACR_NOP SET_DVALID UNP_ZEROSRC — what the
//    stock unpack_A MOP issues for srcB), so the third input costs one srcA unpack, no zero tile, no extra CB.
//  * block unpack: the DEST block's contiguous tiles are unpacked under ONE unpacker config context (one
//    base-address write + one semaphore handshake), the per-tile MOP run back to back while the UNPACR Z
//    counters walk on into the next page; block math: the ELWADD MOP run back to back while the DEST write
//    counter walks on into the next 64-row slot. The per-tile add_tiles() software overhead on the UNPACK
//    TRISC (address/context/semaphore per tile) was the throughput limiter.
// What the helper lacks (capability): no 3-operand fp32-DEST accumulate form without a DEST->src move, no
// zero-srcB operand, and no multi-tile (one context) unpack/math issue — add_block() is a per-tile loop.
// SEGWAIT (define, default 1): CB handshakes per segment (wait/pop inputs, reserve/push sum once per segment)
// instead of per DEST block.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef SEGWAIT
#define SEGWAIT 1
#endif

#ifdef TRISC_UNPACK
// n contiguous tiles starting at tile `first` of the CB fronts, one unpacker config context.
// ZERO == 0: srcA <- cba, srcB <- cbb (the programmed unpack-AB MOP).
// ZERO == 1: srcA <- cba, srcB <- ZEROSRC filler (no srcB L1 read).
// ZERO == 2: srcA <- ZEROSRC filler, srcB <- cba (the third input rides the B unpacker).
template <uint32_t ZERO>
inline void xadd_unpack_block(uint32_t cba, uint32_t cbb, uint32_t first, uint32_t n) {
    const uint32_t ida = get_operand_id(cba);
    const uint32_t addr_a =
        get_local_cb_interface(ida).fifo_rd_ptr - 1 + get_local_cb_interface(ida).fifo_page_size * first;
    uint32_t addr_b = addr_a;
    if constexpr (ZERO == 0) {
        const uint32_t idb = get_operand_id(cbb);
        addr_b = get_local_cb_interface(idb).fifo_rd_ptr - 1 + get_local_cb_interface(idb).fifo_page_size * first;
    }
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(addr_a, addr_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    for (uint32_t i = 0; i < n; ++i) {
        if constexpr (ZERO == 1) {
#pragma GCC unroll 4
            for (uint32_t f = 0; f < 4; ++f) {
                TTI_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
                TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
            }
        } else if constexpr (ZERO == 2) {
#pragma GCC unroll 4
            for (uint32_t f = 0; f < 4; ++f) {
                TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
                TTI_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            }
        } else {
            ckernel::ckernel_template::run();  // unpack-AB MOP: 4 x (UNPACR A, UNPACR B)
        }
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_MATH
// Re-program only the ELWADD MOP (acc bit) — addr mods / counters untouched.
inline void xadd_math_mop(uint32_t acc) {
    eltwise_binary_configure_mop_standard<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MathFidelity::LoFi>(
        acc, ckernel::DEFAULT_TENSOR_SHAPE);
}
// ELWADD(acc_to_dest) MOP n times from DEST slot 0; the MOP end op resets only the src counters.
inline void xadd_math_block(uint32_t n) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    for (uint32_t i = 0; i < n; ++i) {
        ckernel::ckernel_template::run();
    }
    math::clear_dst_reg_addr();
}
#endif

#ifdef TRISC_PACK
// n DEST tiles (slots 0..n-1) -> n contiguous output pages starting at page `first` of the reserved window, in
// ONE pack MOP run (the LLK's num_tiles pack MOP: 4*n faces, tile closed once at the end). The MOP is
// re-programmed only when n changes (a 7-tile segment alternates 4 / 3).
inline void xadd_pack_block(uint32_t cb, uint32_t first, uint32_t n, uint32_t& mop_tiles) {
    if (n != mop_tiles) {
        _llk_pack_mop_config_<PackMode::Default, false>(FACE_R_DIM, TILE_C_DIM, 4, n);
        mop_tiles = n;
    }
    const uint32_t id = get_output_id(cb);
    const uint32_t addr =
        get_local_cb_interface(id).fifo_wr_ptr + get_local_cb_interface(id).fifo_page_size * first - 1;
    _llk_pack_<DST_SYNC_MODE, DST_ACCUM_MODE, PackMode::Default>(0, addr);
}
#endif

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_xport_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_arrival_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_arrival_b = get_compile_time_arg_val(2);
    constexpr uint32_t cb_xport_sum = get_compile_time_arg_val(3);
    constexpr uint32_t has_a = get_compile_time_arg_val(4);
    constexpr uint32_t has_b = get_compile_time_arg_val(5);
    constexpr uint32_t add_block = get_compile_time_arg_val(6);  // tiles per DEST batch (<= seg_tiles, <= 4)
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(7);
    const uint32_t num_copy_segs = get_arg_val<uint32_t>(0);  // upstream-less entries: copied through
    const uint32_t num_segs = get_arg_val<uint32_t>(1);       // relay entries (or the finals' block): added

    constexpr uint32_t cb_second = has_a ? cb_arrival_a : cb_arrival_b;
    constexpr bool three = has_a && has_b;
    constexpr auto in_cfg = [](uint32_t cb) {
        return input(cb, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block);
    };
    compute_kernel_hw_startup(cb_xport_partial, cb_second, cb_xport_sum);
    if (num_copy_segs > 0) {
        MaybeDeviceZoneScope("xadd_copy");
        eltwise_chain(
            IterationShape::grid(num_copy_segs, seg_tiles).block_size(add_block),
            CopyTile<in_cfg(cb_xport_partial)>{},
            PackTile<output(cb_xport_sum, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{});
    }
    if (num_segs == 0) {
        return;
    }
    MaybeDeviceZoneScope("xadd_add");
    // One init for the whole walk (after the copy chain, which left the unpack-A / datacopy MOPs programmed).
    add_init(cb_xport_partial, cb_second, /*acc_to_dest=*/three);
    // acc_to_dest needs DEST == 0 at every acquire. The packer clears each half it releases
    // (_llk_pack_dest_section_done_), so only the first acquire of each half needs an explicit clear (DEST's boot
    // contents are not guaranteed; a half the copy-through walk used is already clear, re-clearing is harmless).
    uint32_t halves_to_clear = three ? 2 : 0;
    [[maybe_unused]] uint32_t pack_mop_tiles = 0;  // 0 = not programmed for a block yet

    for (uint32_t s = 0; s < num_segs; ++s) {
#if SEGWAIT
        cb_wait_front(cb_xport_partial, seg_tiles);
        cb_wait_front(cb_second, seg_tiles);
        if constexpr (three) {
            cb_wait_front(cb_arrival_b, seg_tiles);
        }
        cb_reserve_back(cb_xport_sum, seg_tiles);
#endif
        for (uint32_t off = 0; off < seg_tiles; off += add_block) {
            const uint32_t n = (seg_tiles - off) < add_block ? (seg_tiles - off) : add_block;
#if SEGWAIT
            const uint32_t first = off;
#else
            const uint32_t first = 0;
            cb_wait_front(cb_xport_partial, n);
            cb_wait_front(cb_second, n);
            if constexpr (three) {
                cb_wait_front(cb_arrival_b, n);
            }
#endif
            tile_regs_acquire();
            if constexpr (three) {
                if (halves_to_clear > 0) {
                    MATH((TT_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, dest_offset_id % 2)));
                    --halves_to_clear;
                }
            }
            UNPACK((xadd_unpack_block<0>(cb_xport_partial, cb_second, first, n)));
#ifdef MATHSPLIT
            if constexpr (three) {
                MATH((xadd_math_mop(0)));
            }
#endif
            MATH((xadd_math_block(n)));
            if constexpr (three) {
#ifndef ZSIDE
#define ZSIDE 1
#endif
#if ZSIDE == 3
                if (off & add_block) {  // alternate the third input's unpacker per DEST block
                    UNPACK((xadd_unpack_block<2>(cb_arrival_b, cb_arrival_b, first, n)));
                } else {
                    UNPACK((xadd_unpack_block<1>(cb_arrival_b, cb_arrival_b, first, n)));
                }
#else
                UNPACK((xadd_unpack_block<ZSIDE>(cb_arrival_b, cb_arrival_b, first, n)));
#endif
#ifdef MATHSPLIT
                MATH((xadd_math_mop(1)));
#endif
                MATH((xadd_math_block(n)));
            }
            tile_regs_commit();
#if !SEGWAIT
            cb_pop_front(cb_xport_partial, n);
            cb_pop_front(cb_second, n);
            if constexpr (three) {
                cb_pop_front(cb_arrival_b, n);
            }
            cb_reserve_back(cb_xport_sum, n);
#endif
            tile_regs_wait();
#if defined(PACKBLOCK) && SEGWAIT
            PACK((xadd_pack_block(cb_xport_sum, off, n, pack_mop_tiles)));
#elif !defined(DIAG_NOPACK)
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_xport_sum);  // sequential: the output write pointer walks the reserved window
            }
#endif
            tile_regs_release();
#if !SEGWAIT
            cb_push_back(cb_xport_sum, n);
#endif
        }
#if SEGWAIT
        cb_pop_front(cb_xport_partial, seg_tiles);
        cb_pop_front(cb_second, seg_tiles);
        if constexpr (three) {
            cb_pop_front(cb_arrival_b, seg_tiles);
        }
        cb_push_back(cb_xport_sum, seg_tiles);
#endif
    }
}
