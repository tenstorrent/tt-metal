// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — transport-core TRISC: relay_add_block / final_add_block.
//
// cb_xport_sum = cb_xport_partial + arrival A [+ arrival B], accumulated in fp32 DEST (the transport kernels always
// run with fp32_dest_acc_en=True) and packed bf16. A relay port has (partial, A); a final core has (partial, A, B),
// or one of the two arrivals at a line end.
// A ring port's list starts with entries that have no upstream (the chip's own partial starts the block's chain):
// their tiles are copied through partial -> sum first (cb_xport_sum keeps exactly one producer), then the relay
// entries are added.
// The walk is segment by segment: every segment (the sender's unit) is waited for, added, pushed and popped on its
// own -- never held back for tiles of the next segment (a cross-segment lookahead chains into the upstream chip
// and can close a wait cycle on short block lists). Within a segment the DEST blocks are add_block (<= 4, the fp32
// half-sync capacity) tiles, the last one ragged. Segments never straddle the CB wrap (capacity is a multiple of
// seg_tiles), so a segment's pages are contiguous in every CB.
//
// RAW LLK for the add walk -- bypasses compute_kernel_lib::eltwise_chain (BinaryFpu + DestReuseBinary + PackTile);
// the copy-through walk still uses the helper. Mechanism (same fp32-DEST precision contract, measured on BH p150,
// 280 tiles, seg 7: 2-input 9.83 -> 6.14 us, 3-input 27.96 -> 10.69 us):
//  * 3 inputs as two DEST-accumulating adds under ONE hoisted ELWADD(acc_to_dest) init:
//        dest += partial + A ;  dest += B + 0
//    DEST is zero at every acquire (first acquire of each half cleared here; afterwards the packer clears each
//    half it releases). The helper's 3-input form re-ran add_init + add_reuse_dest_init every DEST block and
//    moved DEST -> srcA face by face (which also truncates the running sum to srcA precision); here the sum
//    never leaves fp32 DEST, so the result is at least as accurate (fewer mismatches vs the RNE fp32 sum).
//  * "+ 0": srcB is the unpacker's ZEROSRC filler (UNPACR_NOP SET_DVALID UNP_ZEROSRC, the srcB filler the stock
//    unpack_A MOP issues) -- no zero tile, no extra CB, no second unpack MOP.
//  * block issue: per DEST block, ONE unpacker config context (one base-address write + one semaphore
//    handshake) with the unpack MOP run back to back while the UNPACR Z counters walk into the next page; the
//    ELWADD MOP run back to back while the DEST write counter walks into the next 64-row slot; ONE pack MOP
//    over the block's n tiles (the LLK num_tiles pack MOP, re-programmed only when n changes). The per-tile
//    add_tiles()/pack_tile() software overhead on the UNPACK / PACK TRISCs was the throughput limiter.
// What the helper lacks (capability, not ergonomics): a 3-operand fp32-DEST accumulate without a DEST->src move,
// a zero-srcB operand, and multi-tile (one-context / one-MOP) unpack / math / pack issue -- add_block() and
// pack_block() are per-tile loops.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Stage zones (permanent; opt-in via KERNEL_PERF_ZONES): xadd_copy / xadd_add span the whole copy-through / add walk
// (the walk waits per segment on the reader and reserves on the sender / writer, so occupancy, not payload).

#ifdef TRISC_UNPACK
// n contiguous tiles starting at page `first` of the CB fronts, under one unpacker config context.
// ZERO_B == false: srcA <- cba, srcB <- cbb (the programmed unpack-AB MOP, 4 x (UNPACR A, UNPACR B)).
// ZERO_B == true : srcA <- cba, srcB <- ZEROSRC filler (no srcB L1 read; cbb unused).
// Mirrors _llk_unpack_AB_ (context acquire, base-address write, semaphore handshake) with n MOP runs.
template <bool ZERO_B>
inline void xadd_unpack_block(uint32_t cba, uint32_t cbb, uint32_t first, uint32_t n) {
    const uint32_t ida = get_operand_id(cba);
    const uint32_t addr_a =
        get_local_cb_interface(ida).fifo_rd_ptr - 1 + get_local_cb_interface(ida).fifo_page_size * first;
    uint32_t addr_b = addr_a;
    if constexpr (!ZERO_B) {
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
        if constexpr (ZERO_B) {
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
// The programmed ELWADD MOP n times from DEST slot 0; its end op resets only the src counters, so the DEST write
// counter walks on into the next tile slot.
inline void xadd_math_block(uint32_t n) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    for (uint32_t i = 0; i < n; ++i) {
        ckernel::ckernel_template::run();
    }
    math::clear_dst_reg_addr();
}
#endif

#ifdef TRISC_PACK
// DEST slots 0..n-1 -> n contiguous output pages from page `first` of the reserved window, ONE pack MOP run
// (LLK num_tiles pack MOP: 4*n faces, tile closed once; n <= 4 for an fp32 DEST). Re-programmed when n changes.
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
    static_assert(add_block >= 1 && add_block <= 4, "fp32 half-sync DEST holds 4 tiles; pack MOP limit");

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
    // One init for the whole add walk (re-programs unpack-AB + ELWADD over whatever the copy walk left).
    add_init(cb_xport_partial, cb_second, /*acc_to_dest=*/three);
    // acc_to_dest needs DEST == 0 at every acquire: clear each half on its first acquire (boot contents are not
    // guaranteed); afterwards the packer clears every half it releases (_llk_pack_dest_section_done_).
    [[maybe_unused]] uint32_t halves_to_clear = three ? 2 : 0;
    [[maybe_unused]] uint32_t pack_mop_tiles = 0;  // 0: pack MOP not yet programmed for a block

    for (uint32_t s = 0; s < num_segs; ++s) {
        cb_wait_front(cb_xport_partial, seg_tiles);
        cb_wait_front(cb_second, seg_tiles);
        if constexpr (three) {
            cb_wait_front(cb_arrival_b, seg_tiles);
        }
        cb_reserve_back(cb_xport_sum, seg_tiles);
        for (uint32_t off = 0; off < seg_tiles; off += add_block) {
            const uint32_t n = (seg_tiles - off) < add_block ? (seg_tiles - off) : add_block;
            tile_regs_acquire();
            if constexpr (three) {
                if (halves_to_clear > 0) {
                    MATH((TT_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, dest_offset_id % 2)));
                    --halves_to_clear;
                }
            }
            UNPACK((xadd_unpack_block<false>(cb_xport_partial, cb_second, off, n)));
            MATH((xadd_math_block(n)));
            if constexpr (three) {
                UNPACK((xadd_unpack_block<true>(cb_arrival_b, cb_arrival_b, off, n)));
                MATH((xadd_math_block(n)));
            }
            tile_regs_commit();
            tile_regs_wait();
            PACK((xadd_pack_block(cb_xport_sum, off, n, pack_mop_tiles)));
            tile_regs_release();
        }
        cb_pop_front(cb_xport_partial, seg_tiles);
        cb_pop_front(cb_second, seg_tiles);
        if constexpr (three) {
            cb_pop_front(cb_arrival_b, seg_tiles);
        }
        cb_push_back(cb_xport_sum, seg_tiles);
    }
}
