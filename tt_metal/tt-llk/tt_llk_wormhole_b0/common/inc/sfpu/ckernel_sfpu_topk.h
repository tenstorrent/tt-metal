// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_load_config.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

static std::int32_t topk_replay_init = 0;

// Tie-break polarity for the stable compare-exchange (true = descending / largest first).
// This is a property of the GLOBAL sort order, not of any one call's sort direction, so the
// kernel must set it once (after topk init) when STABLE_SORT is used. Default false = ascending.
// Deliberately not reset by _init_topk.
static bool topk_stable_descending_mode = false;

TT_ALWAYS_INLINE void set_topk_stable_descending_mode(bool descending)
{
    topk_stable_descending_mode = descending;
}

inline void set_dst_write_addr(std::uint32_t addr)
{
    LLK_ASSERT(addr < DEST_REGISTER_HALF_SIZE, "Address overflow in set_dst_write_addr");
    std::uint32_t dst_index = addr + get_dest_buffer_base();
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);
}

// UInt16 values in 32-bit DEST (fp32_dest_acc_en): datum lives in the low 16 bits with garbage in the
// high half (#50215 / bit-11 removal). Sort/topk must INT32-load values, clear the high bits before
// compare-swap, and SFPSTORE mode-9 before pack so the packer reads the high half. Gated by
// TOPK_UINT16_FP32_DEST from the sort program factory when values are UInt16 and indices force 32-bit DEST.
#if defined(TOPK_UINT16_FP32_DEST) && TOPK_UINT16_FP32_DEST
constexpr bool TOPK_UINT16_IN_FP32_DEST = true;
#else
constexpr bool TOPK_UINT16_IN_FP32_DEST = false;
#endif

// SFPSTORE mode 9 (SFPSTORE_MOD0_FMT_LO16): low→high 16-bit so packer sees UInt16 in 32-bit DEST.
constexpr std::uint32_t TOPK_SFPSTORE_MODE_PACK_UINT16 = 9;

// Fused-key mode (FUSED template parameter on the drivers below): the network sorts opaque
// [bf16|u16] packed words that live only in the value region — index loads/stores disappear
// (half the DEST traffic) and every value access must be raw INT32, because a float-mode store
// denormal-flushes 0x0000xxxx keys (value +0.0), silently erasing the index bits.

// 32 SFPU vectors cover one 32-bit DEST tile at addresses 0,2,...,62 (same footprint as
// typecast VectorMode::RC with ITERATIONS=8). Explicit offsets + ADDR_MOD_3 (incr=0) avoid
// touching ADDR_MOD_6, which topk uses for alt-stores (incr=32).
// Wormhole: with addr_mod_base=1, insn ADDR_MOD_3 → phys ADDR_MOD_7 (SFPU invariant, incr=0).
// tile_index / store_mode are template parameters so the leaf uses TTI_SFPLOAD/TTI_SFPSTORE
// (ISA-immediate encoding); no RISC-V setup for the operand registers per vector.
#define TOPK_UINT16_STRIP_VEC(base, off, store_mode)                                  \
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, (base) + (off)); \
    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);                                  \
    TTI_SFPSTORE(p_sfpu::LREG0, store_mode, ADDR_MOD_3, (base) + (off))

template <std::uint32_t tile_index, std::uint32_t store_mode>
inline void topk_uint16_strip_tile()
{
    constexpr std::uint32_t base = tile_index * 64;
    TOPK_UINT16_STRIP_VEC(base, 0, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 2, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 4, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 6, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 8, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 10, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 12, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 14, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 16, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 18, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 20, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 22, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 24, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 26, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 28, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 30, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 32, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 34, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 36, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 38, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 40, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 42, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 44, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 46, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 48, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 50, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 52, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 54, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 56, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 58, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 60, store_mode);
    TOPK_UINT16_STRIP_VEC(base, 62, store_mode);
}

#undef TOPK_UINT16_STRIP_VEC

// Called from inside topk (addr_mod_base already 1).
inline void topk_uint16_clear_value_tiles_high_bits()
{
    if constexpr (TOPK_UINT16_IN_FP32_DEST)
    {
        sfpi::vConstIntPrgm0 = 0x0000FFFF;
        set_dst_write_addr(0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        topk_uint16_strip_tile<0, static_cast<std::uint32_t>(InstrModLoadStore::INT32)>();
        topk_uint16_strip_tile<1, static_cast<std::uint32_t>(InstrModLoadStore::INT32)>();
        set_dst_write_addr(0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
}

// Called from sort after topk/transpose (outside sfpu_start). Enable addr_mod_base for ADDR_MOD_3.
inline void topk_uint16_prepare_value_tile_for_pack(std::uint32_t dst_tile_index)
{
    if constexpr (TOPK_UINT16_IN_FP32_DEST)
    {
        sfpi::vConstIntPrgm0 = 0x0000FFFF;
        TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1);
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
        set_dst_write_addr(0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        if (dst_tile_index == 0)
        {
            topk_uint16_strip_tile<0, TOPK_SFPSTORE_MODE_PACK_UINT16>();
        }
        else
        {
            LLK_ASSERT(dst_tile_index == 1, "prepare_value_tile_for_pack expects dst tile 0 or 1");
            topk_uint16_strip_tile<1, TOPK_SFPSTORE_MODE_PACK_UINT16>();
        }
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
        TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 0);
        set_dst_write_addr(0);
    }
    else
    {
        (void)dst_tile_index;
    }
}

// Ungated variant for the fused-key final extraction: a u16 datum transposed into 32-bit DEST
// lands in the low half with stale garbage above it, while the packer reads the high half.
// Strip the garbage and move the datum up (SFPSTORE mode 9). Runs on MATH while DEST is acquired,
// after the transpose has drained. Called outside sfpu_start: enable addr_mod_base for ADDR_MOD_3.
inline void _topk_uint16_move_dest_tile_to_pack_half_(std::uint32_t dst_tile_index)
{
    sfpi::vConstIntPrgm0 = 0x0000FFFF;
    TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    switch (dst_tile_index)
    {
        case 0:
            topk_uint16_strip_tile<0, TOPK_SFPSTORE_MODE_PACK_UINT16>();
            break;
        case 1:
            topk_uint16_strip_tile<1, TOPK_SFPSTORE_MODE_PACK_UINT16>();
            break;
        case 2:
            topk_uint16_strip_tile<2, TOPK_SFPSTORE_MODE_PACK_UINT16>();
            break;
        default:
            LLK_ASSERT(dst_tile_index == 3, "move_dest_tile_to_pack_half expects dst tile 0..3");
            topk_uint16_strip_tile<3, TOPK_SFPSTORE_MODE_PACK_UINT16>();
            break;
    }
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 0);
    set_dst_write_addr(0);
}

// Fused-key stable topk: pack each datum into one 32-bit word [bf16 value | u16 index'] where
// index' = index XOR (0xFFFF iff (value_sign == 0) XNOR largest). Sign-magnitude SFPSWAP order on
// the packed word is then a strict total order whose value-tie order is the requested torch-stable
// order in BOTH global directions, and stays correct under the network's internal direction
// alternation (mirror runs) — so the plain UNSTABLE swap network sorts it. Requires 32-bit DEST
// (values exact-widened to [bf16|0x0000] by the fp32-dest datacopy) and raw INT32 load/store for
// every touch of a packed word (a float-mode store would denormal-flush 0x0000xxxx keys).
//
// _topk_fuse_tile_ consumes the u16 index tiles at DEST offset 128 (garbage high bits per #50215,
// masked here) and packs into the value tiles at offset 0; the index region is dead afterwards.
// _topk_defuse_tile_ restores the pre-fuse layout: [bf16|0x0000] value words in place, u16 indices
// back at offset 128 (index_store_mode selects raw INT32 or the mode-9 low->high store the packer
// needs to read u16 from 32-bit DEST). Both are one-shot O(tile) sweeps called explicitly by the
// kernel (fuse once per fresh slab with the GLOBAL direction — never per network call; defuse once
// on the final output), and both record over replay slots 0..9, so the fuse poisons
// topk_replay_init to force the network to re-record its load/store/phase windows.
// Called from inside the sfpu wrapper (addr_mod_base already 1 — ADDR_MOD_3 = physical incr-0 mod).
template <bool largest>
inline void _topk_fuse_tile_()
{
    // Lanes-on FIRST: the constant programming below goes through the SFPCONFIG path, which is
    // lane-PREDICATED (and clobbers LREG0 transiently) — programmed under a partially-enabled
    // ambient CC state, disabled lanes would keep stale LREG12 bits and the mask/complement would
    // silently misfire in exactly those lanes.
    TTI_SFPENCC(3, 0, 0, 10);
    sfpi::vConstIntPrgm0 = 0x0000FFFF;

    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // 9-slot body: one 32-lane vector per replay. The XOR complement runs only in lanes selected
    // by the value-sign test; the AND mask and the OR must be unconditional, hence the bracketing.
    lltt::record<lltt::Exec>(0, 9);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);   // value [bf16|0x0000]
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, 128); // index [garbage|u16]
    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);                       // L1 &= 0x0000FFFF (#50215)
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, largest ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG1, 0); // complement enabled lanes
    TTI_SFPENCC(3, 0, 0, 10);                        // all lanes back on
    TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);   // L0 |= L1 -> packed key
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
    TTI_INCRWC(0, 2, 0, 0); // next 32-lane vector (Matrix-unit issue, free vs the SFPU port)
    for (int i = 1; i < 64; i++)
    {
        lltt::replay(0, 9);
    }

    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Slots 0..8 now hold the fuse body; make the network re-record its cached windows.
    topk_replay_init = 0;
}

template <bool largest, std::uint32_t index_store_mode = static_cast<std::uint32_t>(InstrModLoadStore::INT32)>
inline void _topk_defuse_tile_(const int num_tiles)
{
    // Lanes-on FIRST — the constant write is lane-predicated (see _topk_fuse_tile_).
    TTI_SFPENCC(3, 0, 0, 10);
    sfpi::vConstIntPrgm0 = 0x0000FFFF;

    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // 10-slot body. The sign test runs on the packed word BEFORE its halves are cleared: bit 31
    // is the fused value's sign (the network moves whole words raw), so the same predicate as the
    // fuse selects the same lanes — the complement is self-inverse.
    lltt::record<lltt::Exec>(0, 10);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0); // packed [bf16|idx']
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, largest ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG1, 0); // un-complement lo16
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, 0); // L1 = [0x0000|u16 idx]
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0); // L0 = [bf16|0x0000] (exact bf16 pack)
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, index_store_mode, ADDR_MOD_3, 128); // index region restored
    TTI_INCRWC(0, 2, 0, 0);
    {
        const int n = 32 * num_tiles;
        for (int i = 1; i < n; i++)
        {
            lltt::replay(0, 10);
        }
    }

    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    topk_replay_init = 0;
}

// =============================================================================
//  Rank-stamped stable topk (RANK_STAMPED)
// =============================================================================
//
// Stable topk beyond the u16 index limit: the value words' free low 16 bits
// carry a sign-conditioned LOCAL RANK tag while the true (u32) index tiles
// keep riding the index-tracking swaps at DEST offset 128. The plain UNSTABLE
// SFPSWAP network then sorts distinct packed keys:
//
//     value word = [bf16 value | rank XOR (0xFFFF iff value_pos XNOR largest)]
//
// Under the raw sign-magnitude compare, key order within an equal-value class
// equals ascending-rank order along the GLOBAL sort direction (and the mirror
// order in the flipped direction) for both value signs -- the same
// conditioning the fused-key engine applies to its u16 index tag, with a
// local rank standing in for the index that no longer fits 16 bits.
//
// Ranks are LOCAL and are re-derived at every stamping point:
//   * once per freshly transposed 2-tile slab, before the local sort:
//     rank = the datum's 64-sequence position (`_topk_stamp_local_positions_`).
//     Both live callers satisfy the precondition "storage position order ==
//     ascending global index among equal values": a fresh slab trivially
//     (position IS the width offset), and the single-core insertion
//     accumulator inductively (it is re-sorted stably each round and covers
//     strictly lower width chunks than the incoming tile).
//   * before every compare inside the merge: rank = position within the
//     k-run; the LEFT run (always sorted in the global direction and always
//     the lower global-index range in the classic merge tree) takes
//     [0, min(k, 32)) -- every in-tree merge is a single call per tile pair,
//     so the left-run base is always 0 (a K=64 split-call variant would need
//     a per-call rank base plumbed back in) -- the RIGHT (mirror-direction)
//     run is complemented through 2K-1 into the disjoint upper range. Folded
//     into `_bitonic_topk_merge<..., RANK_STAMPED>`.
//   * rebuild needs nothing: it inherits distinct, correctly tie-ordered
//     tags from the preceding merge (value tiles travel as raw 32-bit words
//     through Float32-format CBs, exactly like fused packed keys).
// The stale tags are stripped (`_topk_strip_rank_tags_`) after the final
// transpose, before the value pack, which would otherwise RNE-round on them.
//
// Requires 32-bit DEST: every value load/store switches to raw INT32 (a
// float-mode store would denormal-flush 0x0000xxxx tagged words -- the fused
// rule), and the index tiles use the INT32 arm the u32 index path already has.
//
// TEN-2932 discipline (index-tracking mode is ON): the merge-time stamp runs
// AFTER the true indices are loaded into LREG4/5, so it must not issue any
// SFPLOAD/SFPLOADI to LREG0..3 (loads capture into LREG4..7); its low-16
// clear is an SFPAND against LREG11 = 0xFFFF0000 and every op is an ALU
// write to LREG0..3 or a programmable-constant read. The standalone sweeps
// below run while LREG4..7 are dead, so their load captures are harmless.

// Stamp the 2-tile slab's value words with their sign-conditioned sequence
// position (rank 0..63 per 64-datum column), clearing any stale low bits, and
// fold -0.0 into the +0.0 tie class on the way (torch treats +-0 as ONE tie
// class broken by index; the raw sign-magnitude compare would otherwise order
// all -0.0 strictly below all +0.0). Every datum enters the sort through
// exactly one local-sort call, so this single sweep canonicalizes the whole
// sort. Clobbers LREG0..2 and the lane enables (left fully enabled).
template <bool largest>
inline void _topk_stamp_local_positions_()
{
    // Lanes-on FIRST -- the constant programming below goes through the
    // lane-PREDICATED SFPCONFIG path (see _topk_fuse_tile_).
    TTI_SFPENCC(3, 0, 0, 10);
    sfpi::vConstIntPrgm0 = 0x0000FFFF; // LREG12: tag complement operand

    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Per-lane rank base: an SFPLOAD vector covers 4 consecutive Dst rows
    // (lane j reads row rwc + j/8), and consecutive rows within a 16-row face
    // are consecutive sequence positions. LTILEID = 2*j, so j>>3 = LTILEID>>4.
    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG2, 0);
    TTI_SFPSHFT((-4) & 0xFFF, 0, p_sfpu::LREG2, 1);

    for (int g = 0; g < 32; g++) // 4-row groups across the 2-tile slab
    {
        for (int parity = 0; parity < 2; parity++) // even / odd Dst columns
        {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0); // clear stale lo16 -> [bf16|0]
            // -0.0 -> +0.0: (w << 1) == 0 exactly for the two zero encodings.
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
            TTI_SFPSHFT(1, 0, p_sfpu::LREG1, 1);
            TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0); // zero lanes -> +0.0
            TTI_SFPENCC(3, 0, 0, 10);
            // Tag with the sequence position, sign-conditioned.
            TTI_SFPOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            TTI_SFPSETCC(0, p_sfpu::LREG0, 0, largest ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
            TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
            TTI_INCRWC(0, 2, 0, 0);
        }
        // Advance the rank base to the next 4-row group: +4 within a face;
        // when the group crosses into the paired face of the SAME sequence
        // positions (rows 16..31 hold positions 0..15 of the other column
        // half), rewind by 12 instead.
        if ((g & 7) == 3)
        {
            TTI_SFPIADD((-12) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        }
        else
        {
            TTI_SFPIADD(4, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        }
    }

    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

// Clear the low 16 bits (stale rank tags) of one value tile, leaving exact
// [bf16|0x0000] words so the following Float32->bf16 pack cannot RNE-round on
// tag bits. Runs on MATH while DEST is acquired, after the final transpose
// back to row layout has drained (same calling convention as
// _topk_uint16_move_dest_tile_to_pack_half_, including the addr_mod_base
// toggle, and reusing its sweep with the
// complementary mask in LREG12). LREG4..7 are dead here, so nothing tracked
// is at risk. Leaves LREG12 holding the strip mask -- every stamp/merge/fuse
// entry reprograms LREG12 defensively.
inline void _topk_strip_rank_tags_(std::uint32_t dst_tile_index)
{
    TTI_SFPENCC(3, 0, 0, 10);
    sfpi::vConstIntPrgm0 = 0xFFFF0000; // keep the bf16 half, clear the rank tag
    TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    if (dst_tile_index == 0)
    {
        topk_uint16_strip_tile<0, static_cast<std::uint32_t>(InstrModLoadStore::INT32)>();
    }
    else
    {
        LLK_ASSERT(dst_tile_index == 1, "strip_rank_tags expects dst tile 0 or 1");
        topk_uint16_strip_tile<1, static_cast<std::uint32_t>(InstrModLoadStore::INT32)>();
    }
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 0);
    set_dst_write_addr(0);
}

template <bool is_fp32_dest_acc_en, bool FUSED = false, bool RANK_STAMPED = false>
inline void bitonic_topk_load8(std::uint32_t offset, std::uint32_t dist)
{
    constexpr std::uint32_t dst_indices_offset  = 128; // 2 tile x 64 rows per tile
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    // UInt16-in-32b-DEST: full-width INT32 value access (high bits pre-cleared). Else DEFAULT (0).
    constexpr InstrModLoadStore instr_mod_value = (TOPK_UINT16_IN_FP32_DEST || FUSED || RANK_STAMPED) ? InstrModLoadStore::INT32 : InstrModLoadStore::DEFAULT;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset   = (offset & 0xF) + face_offset * 32;

    // Load 16 consecutive numbers
    TT_SFPLOAD(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_3, ld_offset);
    TT_SFPLOAD(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_3, ld_offset + dist);

    if constexpr (!FUSED)
    {
        // Load 16 consecutive indices
        TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, dst_indices_offset + ld_offset);
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, dst_indices_offset + ld_offset + dist);
    }
}

template <bool is_fp32_dest_acc_en, bool FUSED = false, bool RANK_STAMPED = false>
inline void bitonic_topk_store8(std::uint32_t offset, std::uint32_t dist)
{
    constexpr std::uint32_t dst_indices_offset  = 128; // 2 tile x 64 rows per tile
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    constexpr InstrModLoadStore instr_mod_value = (TOPK_UINT16_IN_FP32_DEST || FUSED || RANK_STAMPED) ? InstrModLoadStore::INT32 : InstrModLoadStore::DEFAULT;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset   = (offset & 0xF) + face_offset * 32;

    // Load 16 consecutive numbers
    TT_SFPSTORE(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_3, ld_offset);
    TT_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_3, ld_offset + dist);

    if constexpr (!FUSED)
    {
        // Load 16 consecutive indices
        TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, dst_indices_offset + ld_offset + 0);
        TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, dst_indices_offset + ld_offset + dist);
    }
}

template <bool is_fp32_dest_acc_en, bool FUSED = false, bool RANK_STAMPED = false>
inline void bitonic_topk_load16(std::uint32_t dist0, std::uint32_t dist1)
{
    constexpr std::uint32_t dst_indices_offset  = 128; // 2 tile x 64 rows per tile
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    constexpr InstrModLoadStore instr_mod_value = (TOPK_UINT16_IN_FP32_DEST || FUSED || RANK_STAMPED) ? InstrModLoadStore::INT32 : InstrModLoadStore::DEFAULT;

    // Load 16 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_3, 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPLOAD(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_3, 4);
        TTI_SFPLOAD(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_3, 8);
        TTI_SFPLOAD(p_sfpu::LREG3, instr_mod_value, ADDR_MOD_3, 12);
    }
    else
    {
        TT_SFPLOAD(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_3, 0 + dist0);
        TT_SFPLOAD(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_3, dist1);
        TT_SFPLOAD(p_sfpu::LREG3, instr_mod_value, ADDR_MOD_3, dist1 + dist0);
    }

    if constexpr (!FUSED)
    {
        // Load 16 consecutive indices
        TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 0);
        if ((dist0 == 4) && (dist1 == 8))
        {
            TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 4);
            TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 8);
            TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 12);
        }
        else
        {
            TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 0 + dist0);
            TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, dst_indices_offset + dist1);
            TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_3, dst_indices_offset + dist1 + dist0);
        }
    }
}

template <bool is_fp32_dest_acc_en, bool alt_addr_mod = false, bool FUSED = false, bool RANK_STAMPED = false>
inline void bitonic_topk_store16(std::uint32_t dist0, std::uint32_t dist1)
{
    constexpr std::uint32_t dst_indices_offset  = 128; // 2 tile x 64 rows per tile
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    constexpr InstrModLoadStore instr_mod_value = (TOPK_UINT16_IN_FP32_DEST || FUSED || RANK_STAMPED) ? InstrModLoadStore::INT32 : InstrModLoadStore::DEFAULT;

    // Load 16 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_3, 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_3, 4);
        TTI_SFPSTORE(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_3, 8);
        TTI_SFPSTORE(p_sfpu::LREG3, instr_mod_value, (FUSED && alt_addr_mod) ? ADDR_MOD_2 : ADDR_MOD_3, 12);
    }
    else
    {
        TT_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_3, 0 + dist0);
        TT_SFPSTORE(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_3, dist1);
        TT_SFPSTORE(p_sfpu::LREG3, instr_mod_value, (FUSED && alt_addr_mod) ? ADDR_MOD_2 : ADDR_MOD_3, dist1 + dist0);
    }

    if constexpr (!FUSED)
    {
        // Load 16 consecutive indices
        TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 0);
        if ((dist0 == 4) && (dist1 == 8))
        {
            TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 4);
            TTI_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 8);
            TTI_SFPSTORE(p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_2 : ADDR_MOD_3, dst_indices_offset + 12);
        }
        else
        {
            TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_3, dst_indices_offset + 0 + dist0);
            TT_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_3, dst_indices_offset + dist1);
            TT_SFPSTORE(p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_2 : ADDR_MOD_3, dst_indices_offset + dist1 + dist0);
        }
    }
}

// Stable compare-exchange for one register pair. Values are the primary key; on exact value
// ties the paired index registers (LREG4+n tracks LREGn) are compare-exchanged so ties resolve
// by index. VD ^= VC leaves 0 only in tied lanes, providing the tie predicate; the second XOR
// restores VD. INDEX_MIN_TO_VD selects the index-swap operand order to match the sort direction.
template <std::uint32_t VC, std::uint32_t VD, std::uint32_t MODE, bool INDEX_MIN_TO_VD>
TT_ALWAYS_INLINE void topk_cmp_swap_stable_directional()
{
    constexpr std::uint32_t IDX_VC = p_sfpu::LREG4 + (VC & 0x3);
    constexpr std::uint32_t IDX_VD = p_sfpu::LREG4 + (VD & 0x3);

    // Primary key: value compare-exchange.
    TTI_SFPSWAP(0, VC, VD, MODE);

    // Predicate lanes where compared values are exactly equal. Lanes-on/flags-true CC state
    // is an entry invariant established once per LLK entry point (see the STABLE_SORT branch
    // of _bitonic_topk_{phases_steps,merge,rebuild}) and re-established by the trailing
    // SFPENCC of every comparator body.
    TTI_SFPXOR(0, VC, VD, 0);
    TTI_SFPSETCC(0, VD, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);

    // Secondary key: index compare-exchange under the tie mask.
    if constexpr (INDEX_MIN_TO_VD)
    {
        TTI_SFPSWAP(0, IDX_VC, IDX_VD, MODE);
    }
    else
    {
        TTI_SFPSWAP(0, IDX_VD, IDX_VC, MODE);
    }
    TTI_SFPENCC(3, 0, 0, 10);

    // Restore values after the XOR scratch operation.
    TTI_SFPXOR(0, VC, VD, 0);
}

// Runtime-polarity wrapper for stable compare sites shared by ascending and descending sorts.
template <std::uint32_t VC, std::uint32_t VD, std::uint32_t MODE>
TT_ALWAYS_INLINE void topk_cmp_swap_stable_min_to_vd()
{
    if (topk_stable_descending_mode)
    {
        topk_cmp_swap_stable_directional<VC, VD, MODE, false>();
    }
    else
    {
        topk_cmp_swap_stable_directional<VC, VD, MODE, true>();
    }
}

template <bool STABLE_SORT, bool FUSED = false>
inline void bitonic_topk_ph3_st4_to_1(bool dir, bool &init_replay, int replay_start)
{
    if (dir == static_cast<bool>(SortDir::ArgMin))
    {
        // Full-register immediate write: 0x104 = swap reversal (bit 8) + index tracking (bit 2).
        // Fused mode has tracking OFF and must keep it off: write 0x100 / restore 0x000.
        TTI_SFPCONFIG(FUSED ? 0x100 : 0x104, 0xF, 1); // Reverse the max/min behaviour of SWAP
        TTI_SFPNOP;
        TTI_SFPNOP;
    }

    if constexpr (STABLE_SORT)
    {
        // The stable sequence exceeds the replay window, so issue inline; two passes to match
        // the unstable path's record + trailing replay. Direction is handled by the SFPCONFIG
        // reversal above, so one body serves both directions.
        topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX>();
        topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX>();
        topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX>();
        topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX>();
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Second pass.
        topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX>();
        topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX>();
        topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX>();
        topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX>();
        TTI_SFPTRANSP(0, 0, 0, 0);

        init_replay = false;
    }
    else
    {
        constexpr int replay_count = 5;
        if (init_replay)
        {
            lltt::record<lltt::Exec>(replay_start, replay_count);

            // Step 4
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

            // Step 3
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

            TTI_SFPTRANSP(0, 0, 0, 0);

            init_replay = false;
        }
        else
        {
            lltt::replay(replay_start, replay_count);
        }
        lltt::replay(replay_start, replay_count);
    }

    if (dir == static_cast<bool>(SortDir::ArgMin))
    {
        TTI_SFPCONFIG(FUSED ? 0x000 : 0x004, 0xF, 1); // Restore the max/min behaviour of SWAP
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
}

template <bool STABLE_SORT>
inline void bitonic_topk_ph2_st3_to_1();

template <>
inline void bitonic_topk_ph2_st3_to_1<true>()
{
    // Step 3
    topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX>();
    topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX>();

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 2
    topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX>();
    topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX>();

    // Step 1
    topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX>();
    topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX>();

    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <>
inline void bitonic_topk_ph2_st3_to_1<false>()
{
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <bool STABLE_SORT>
inline void bitonic_topk_ph1_st2_to_1();

template <>
inline void bitonic_topk_ph1_st2_to_1<true>()
{
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 2
    topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX>();
    topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX>();

    // Step 1
    topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX>();
    topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX>();

    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <>
inline void bitonic_topk_ph1_st2_to_1<false>()
{
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <bool STABLE_SORT>
inline void bitonic_topk_ph0_st1_to_1();

template <>
inline void bitonic_topk_ph0_st1_to_1<true>()
{
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 1
    topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX>();
    topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX>();

    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <>
inline void bitonic_topk_ph0_st1_to_1<false>()
{
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);
}

template <bool STABLE_SORT>
inline void bitonic_topk_step_N(bool dir);

template <>
inline void bitonic_topk_step_N<true>(bool dir)
{
    // Step N
    if (dir == static_cast<bool>(SortDir::ArgMax))
    {
        topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX>();
        topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX>();
    }
    else
    {
        topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX>();
        topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX>();
    }
}

template <>
inline void bitonic_topk_step_N<false>(bool dir)
{
    // Step N
    if (dir == static_cast<bool>(SortDir::ArgMax))
    {
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    }
    else
    {
        // Min
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    }
}

inline void bitonic_topk_inc_x8_dest(std::uint32_t inc, bool cr)
{
    std::uint32_t inc_grp8 = inc >> 3;
    if (cr)
    {
        for (std::uint32_t i = 0; i < inc_grp8; i++)
        {
            TTI_INCRWC(0b100, 8, 0, 0);
        }
    }
    else
    {
        for (std::uint32_t i = 0; i < inc_grp8; i++)
        {
            TTI_INCRWC(0, 8, 0, 0);
        }
    }
}

inline void bitonic_topk_inc_x4_dest(std::uint32_t inc, bool cr)
{
    std::uint32_t inc_grp4 = inc >> 2;
    if (cr)
    {
        for (std::uint32_t i = 0; i < inc_grp4; i++)
        {
            TTI_INCRWC(0b100, 4, 0, 0);
        }
    }
    else
    {
        for (std::uint32_t i = 0; i < inc_grp4; i++)
        {
            TTI_INCRWC(0, 4, 0, 0);
        }
    }
}

// -0.0 canonicalization for the comparator-stable network in 32-bit DEST. The SFPU
// compare-exchange orders values in sign-magnitude space, where -0.0 (0x80000000) sorts
// strictly below +0.0 -- but the stable contract follows torch, which treats them as ONE
// tie class broken by index. Rewrite -0.0 -> +0.0 in the two freshly loaded value tiles
// before the network runs: every datum enters the sort through _bitonic_topk_phases_steps
// exactly once (the multi-core local sort and the single-core insertion loop both
// local-sort each fresh 2-tile slab), and the fp32 pack/unpack transport is a bit
// identity, so this single entry sweep canonicalizes the whole sort. The 16-bit-DEST
// engines need no sweep (their bf16 SrcA datacopy already canonicalizes +-0, silicon-
// probed), uint16-in-fp32-dest values carry no live sign bit after their own strip, and
// fused packed keys never take the comparator-stable path.
// Predicate: (x & 0x7FFFFFFF) == 0 (zero magnitude); action: x &= 0x7FFFFFFF (-> +0.0).
inline void topk_stable_canonicalize_negzero_value_tiles()
{
    sfpi::vConstIntPrgm0 = 0x7FFFFFFF;
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (std::uint32_t off = 0; off < 128; off += 2)
    {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, off);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, off);
    }
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false, bool FUSED = false, bool RANK_STAMPED = false>
inline void _bitonic_topk_phases_steps(const int idir, const int i_end_phase, const int i_start_phase, const int i_end_step, const int i_start_step)
{
    // NOTE (stable sort): the tie-break polarity (topk_stable_descending_mode) is a property of the
    // GLOBAL sort order (largest vs smallest), not of this call's idir. Callers may intentionally run
    // this network with a flipped idir (e.g. per-core direction alternation in the multi-core topk) to
    // build bitonic sequences; the tie polarity must NOT flip with it. The kernel sets the mode once
    // via set_topk_stable_descending_mode().
    // If more than 1 phase is requested, do all the steps from all phases
    // If 1 phase is requested, use i_start_step/i_end_step parameters

    // UInt16-in-32b-DEST: clear garbage high bits before compare-swap (#50215).
    topk_uint16_clear_value_tiles_high_bits();

    static_assert(!(FUSED && STABLE_SORT), "fused and comparator-stable modes are mutually exclusive");
    static_assert(!FUSED || is_fp32_dest_acc_en, "fused packed keys require 32-bit DEST");
    static_assert(!(FUSED && TOPK_UINT16_IN_FP32_DEST), "fused keys and uint16-in-fp32-dest are mutually exclusive");
    static_assert(!(RANK_STAMPED && STABLE_SORT), "rank-stamped and comparator-stable modes are mutually exclusive");
    static_assert(!(RANK_STAMPED && FUSED), "rank-stamped and fused-key modes are mutually exclusive");
    static_assert(!RANK_STAMPED || is_fp32_dest_acc_en, "rank-stamped tagged keys require 32-bit DEST");
    static_assert(!(RANK_STAMPED && TOPK_UINT16_IN_FP32_DEST), "rank-stamped keys and uint16-in-fp32-dest are mutually exclusive");
    // Fused packed keys halve the load/store footprint; replay window bases stay put
    // (slots 4-7 / 12-15 simply go unused in fused mode).
    constexpr int ldst_count = FUSED ? 4 : 8;

    if constexpr (STABLE_SORT)
    {
        // Establish the lanes-on/flags-true CC entry invariant once; every stable comparator
        // body re-establishes it via its trailing SFPENCC, and the intervening loads/stores/
        // transposes/SFPCONFIG writes preserve CC state.
        TTI_SFPENCC(3, 0, 0, 10);
        if constexpr (is_fp32_dest_acc_en && !TOPK_UINT16_IN_FP32_DEST)
        {
            // fp32-family values in 32-bit DEST can carry -0.0: fold it into the +0.0 tie
            // class before the sign-magnitude network runs (see the sweep's doc comment).
            topk_stable_canonicalize_negzero_value_tiles();
        }
    }

    // init the replay buffer for local sort if uninitialized
    bool init_load  = (topk_replay_init >= 0) ? true : false;
    bool init_store = (topk_replay_init >= 0) ? true : false;
    bool init_phase;

    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            bool dir = idir;
            for (int ph = i_start_phase; ph < (i_end_phase + 1); ph++)
            {
                init_phase = true; // init each new phase of local sort in replay buffer

                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                switch (ph)
                {
                    case 0:
                        for (int d = 0; d < 4; d++)
                        {
                            // Groups of 16 datums being sorted at the same time
                            if (init_load)
                            {
                                lltt::record<lltt::Exec>(0, ldst_count);
                                bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 8);
                                init_load = false;
                            }
                            else
                            {
                                lltt::replay(0, ldst_count);
                            }
                            if constexpr (STABLE_SORT)
                            {
                                // Stable sequence exceeds the replay window; issue inline.
                                bitonic_topk_ph0_st1_to_1<STABLE_SORT>();
                                init_phase = false;
                            }
                            else
                            {
                                constexpr int replay_count = 4;
                                if (init_phase)
                                {
                                    lltt::record<lltt::Exec>(16, replay_count);
                                    bitonic_topk_ph0_st1_to_1<STABLE_SORT>();
                                    init_phase = false;
                                }
                                else
                                {
                                    lltt::replay(16, replay_count);
                                }
                            }
                            if (init_store)
                            {
                                lltt::record<lltt::Exec>(8, ldst_count);
                                bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, 8);
                                init_store = false;
                            }
                            else
                            {
                                lltt::replay(8, ldst_count);
                            }
                        }
                        break;
                    case 1:
                        for (int d = 0; d < 4; d++)
                        {
                            // Groups of 16 datums being sorted at the same time
                            lltt::replay(0, ldst_count);
                            if constexpr (STABLE_SORT)
                            {
                                // Stable sequence exceeds the replay window; issue inline.
                                bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                init_phase = false;
                            }
                            else
                            {
                                constexpr int replay_count = 6;
                                if (init_phase)
                                {
                                    lltt::record<lltt::Exec>(16, replay_count);
                                    bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                    init_phase = false;
                                }
                                else
                                {
                                    lltt::replay(16, replay_count);
                                }
                            }
                            lltt::replay(8, ldst_count);
                        }
                        break;
                    case 2:
                        for (int d = 0; d < 4; d++)
                        {
                            lltt::replay(0, ldst_count);
                            if constexpr (STABLE_SORT)
                            {
                                // Stable sequence exceeds the replay window; issue inline.
                                bitonic_topk_ph2_st3_to_1<STABLE_SORT>();
                                init_phase = false;
                            }
                            else
                            {
                                constexpr int replay_count = 9;
                                if (init_phase)
                                {
                                    lltt::record<lltt::Exec>(16, replay_count);
                                    bitonic_topk_ph2_st3_to_1<STABLE_SORT>();
                                    init_phase = false;
                                }
                                else
                                {
                                    lltt::replay(16, replay_count);
                                }
                            }
                            lltt::replay(8, ldst_count);
                        }
                        break;
                    case 3:
                        for (int d = 0; d < 4; d++)
                        {
                            lltt::replay(0, ldst_count);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, FUSED>(dir, init_phase, 16);
                            lltt::replay(8, ldst_count);
                            dir = !dir;
                        }
                        break;
                    default:
                        std::uint32_t num_steps               = ph + 1;
                        std::uint32_t start_step              = (i_start_phase == i_end_phase) ? i_start_step : num_steps;
                        std::uint32_t end_step                = (i_start_phase == i_end_phase) ? i_end_step : 4;
                        std::uint32_t sorted_seq_length       = 1 << num_steps;
                        std::uint32_t datums_compared         = 0;
                        std::uint32_t total_datums_to_compare = 64;
                        for (std::uint32_t ss = start_step; ss > end_step; ss--)
                        {
                            // Steps N to 5
                            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                            dir                      = idir;
                            std::uint32_t dist       = (ss == 5) ? 16 : 32;
                            std::uint32_t inner_d    = dist >> 3; // How many loops to sort the sequence of length (2^ss / 16). Each loop sorts 16
                            datums_compared          = 0;
                            std::uint32_t dst_offset = 0;
                            while (datums_compared < total_datums_to_compare)
                            {
                                for (std::uint32_t ii = 0; ii < inner_d; ii++)
                                {
                                    bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(
                                        4, 2 * dist); // load/store with offset of face 1 (in row major face layout)
                                    bitonic_topk_step_N<STABLE_SORT>(dir);
                                    bitonic_topk_store16<is_fp32_dest_acc_en, false, FUSED, RANK_STAMPED>(
                                        4, 2 * dist); // load/store with offset of face 1 (in row major face layout)
                                    std::uint32_t dst_inc = 8;
                                    dst_offset += dst_inc;
                                    bool dst_cr = false;
                                    if (ii == (inner_d - 1))
                                    {
                                        dst_cr     = true;
                                        dst_inc    = 4 * dist;
                                        dst_offset = 2 * dist;
                                    }
                                    else if (dst_offset == 16)
                                    {
                                        dst_cr  = true;
                                        dst_inc = 32;
                                    }
                                    bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                    datums_compared += 16;
                                }
                                dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                            }
                        }
                        // steps 4 to 1
                        dir = idir;
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                        datums_compared = 0;
                        while (datums_compared < total_datums_to_compare)
                        {
                            lltt::replay(0, ldst_count);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, FUSED>(dir, init_phase, 16);
                            lltt::replay(8, ldst_count);
                            datums_compared += 16;
                            dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                        }
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
    topk_replay_init = -1;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min, bool STABLE_SORT = false, bool FUSED = false, bool RANK_STAMPED = false>
inline void _bitonic_topk_merge(const int m_iter, const int k)
{
    // UInt16-in-32b-DEST: clear garbage high bits before compare-swap (#50215).
    topk_uint16_clear_value_tiles_high_bits();

    static_assert(!(FUSED && STABLE_SORT), "fused and comparator-stable modes are mutually exclusive");
    static_assert(!FUSED || is_fp32_dest_acc_en, "fused packed keys require 32-bit DEST");
    static_assert(!(FUSED && TOPK_UINT16_IN_FP32_DEST), "fused keys and uint16-in-fp32-dest are mutually exclusive");
    static_assert(!(RANK_STAMPED && STABLE_SORT), "rank-stamped and comparator-stable modes are mutually exclusive");
    static_assert(!(RANK_STAMPED && FUSED), "rank-stamped and fused-key modes are mutually exclusive");
    static_assert(!RANK_STAMPED || is_fp32_dest_acc_en, "rank-stamped tagged keys require 32-bit DEST");
    static_assert(!(RANK_STAMPED && TOPK_UINT16_IN_FP32_DEST), "rank-stamped keys and uint16-in-fp32-dest are mutually exclusive");

    if constexpr (STABLE_SORT)
    {
        // Establish the lanes-on/flags-true CC entry invariant once, before the quadrant
        // loops, so it dominates every per-iteration comparator execution; each comparator
        // body re-establishes it via its trailing SFPENCC, and the intervening loads/stores
        // preserve CC state.
        TTI_SFPENCC(3, 0, 0, 10);
    }

    if constexpr (RANK_STAMPED)
    {
        // Lanes-on FIRST -- the constant programming below goes through the lane-PREDICATED
        // SFPCONFIG path and transiently clobbers LREG0 (see _topk_fuse_tile_), so it must
        // run before any load and under fully enabled lanes.
        TTI_SFPENCC(3, 0, 0, 10);
        sfpi::vConstIntPrgm0 = 0x0000FFFF;                    // LREG12: tag complement operand
        _sfpu_load_config32_(p_sfpu::LREG11, 0xFFFF, 0x0000); // lo16 clear mask (SFPAND -- no loads mid-stamp)
        const std::uint32_t rank_span = 2 * static_cast<std::uint32_t>(k) - 1;
        _sfpu_load_config32_(p_sfpu::LREG13, rank_span >> 16, rank_span & 0xFFFF); // right-run complement (2K-1)
    }

    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
            int k_max             = k > 32 ? 32 : k;
            std::uint32_t inner_d = k_max >> 2; // inner loop comparisons to sort len=K sequence;
            std::uint32_t total_datums_to_compare =
                ((64 >> m_iter) < 2 * k_max) ? 2 * k_max
                                             : (64 >> m_iter); // max(2, max(64, 64/(2^m))) total datums to compare; there's always at least 2*K datums
            std::uint32_t dist            = (k_max << m_iter) > 32 ? 32 : (k_max << m_iter); // min(32, k*2^k)
            std::uint32_t ld_dist         = (dist < 16) ? dist : 2 * dist;                   // Accounts for face offsets within a tile
            std::uint32_t datums_compared = 0;
            std::uint32_t dst_offset      = 0;
            std::uint32_t dst_cr          = 0;

            while (datums_compared < total_datums_to_compare)
            {
                for (std::uint32_t ii = 0; ii < inner_d; ii++)
                {
                    bitonic_topk_load8<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(dst_offset, ld_dist);
                    if constexpr (RANK_STAMPED)
                    {
                        // Re-key both runs' value lo16 with fresh sign-conditioned local
                        // ranks so this merge AND the rebuild that follows it compare
                        // distinct keys whose tie order is the true index order. The
                        // true indices are live in LREG4/5 from the load above: from
                        // here to the swap, ALU-only writes to LREG0..3 (TEN-2932).
                        if (ii == 0)
                        {
                            // Fresh per-pair rank iota: rank = 4*ii + (j>>3) (each load
                            // covers 4 consecutive run positions per lane group;
                            // LTILEID = 2*j). Left-run base is 0 for every in-tree
                            // caller -- one merge call per tile pair.
                            TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG2, 0);
                            TTI_SFPSHFT((-4) & 0xFFF, 0, p_sfpu::LREG2, 1);
                        }
                        // largest = !top_min: complement the lanes whose sign matches the
                        // kept extreme, exactly as the fused-key conditioning does.
                        constexpr int stamp_cc = top_min ? sfpi::SFPSETCC_MOD1_LREG_LT0 : sfpi::SFPSETCC_MOD1_LREG_GTE0;
                        // Left run: global direction, lower rank range.
                        TTI_SFPAND(0, p_sfpu::LREG11, p_sfpu::LREG0, 0);
                        TTI_SFPOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
                        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, stamp_cc);
                        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
                        TTI_SFPENCC(3, 0, 0, 10);
                        // Right run: mirror direction, upper range -- rank' = (2K-1) - rank,
                        // a single XOR because 2K is a power of two and rank < 2K.
                        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);
                        TTI_SFPXOR(0, p_sfpu::LREG13, p_sfpu::LREG3, 0);
                        TTI_SFPAND(0, p_sfpu::LREG11, p_sfpu::LREG1, 0);
                        TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG1, 0);
                        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, stamp_cc);
                        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
                        TTI_SFPENCC(3, 0, 0, 10);
                        // Advance to the next 4 run positions.
                        TTI_SFPIADD(4, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
                    }
                    if constexpr (STABLE_SORT)
                    {
                        // top_min selects the value operand order (which run receives the
                        // minima), exactly as in the unstable arm below. The tie-break polarity
                        // comes from the runtime topk_stable_descending_mode -- the GLOBAL sort
                        // order phases_steps/rebuild already read. Anchoring the tie routing to
                        // the operand order (the index minimum rides with the value minimum or
                        // maximum per the global mode) makes every merge a comparator on ONE
                        // fixed total order [value, then index], so callers may issue merges
                        // with per-block alternating directions (ttnn.sort's bitonic merge
                        // network, which always binds top_min=false and routes the outputs
                        // afterwards) or in the global direction (the ttnn topk kernels, which
                        // bind top_min = !largest; the LLK/quasar test kernels, which bind
                        // TOPK_SORT_DIRECTION). For every global-direction caller this compiles
                        // to the pre-existing selection, since they program the runtime mode
                        // from the same flag they derive top_min from.
                        if constexpr (top_min)
                        {
                            topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG1, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX>();
                        }
                        else
                        {
                            topk_cmp_swap_stable_min_to_vd<p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX>();
                        }
                    }
                    else
                    {
                        TTI_SFPSWAP(0, top_min ? p_sfpu::LREG1 : p_sfpu::LREG0, top_min ? p_sfpu::LREG0 : p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                    }
                    bitonic_topk_store8<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(dst_offset, ld_dist);
                    datums_compared += 8;
                    if (ii == (inner_d - 1))
                    {
                        dst_cr += 2 * dist;
                        dst_offset = dst_cr;
                    }
                    else
                    {
                        dst_offset += 4;
                    }
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false, bool FUSED = false, bool RANK_STAMPED = false>
inline void _bitonic_topk_rebuild(const bool idir, const int m_iter, const int k, const int logk, const int skip_second)
{
    // NOTE (stable sort): tie-break polarity comes from the kernel-level
    // set_topk_stable_descending_mode(), NOT from idir. The multi-core topk deliberately
    // rebuilds with an alternating per-core idir so adjacent cores emit opposite-sorted sequences;
    // deriving the tie polarity from idir here would make those flipped cores emit ties in
    // (index-ascending) order instead of the mirror (index-descending) order the global bitonic
    // merge requires, misordering equal values on wide multi-core shapes.
    // UInt16-in-32b-DEST: clear garbage high bits before compare-swap (#50215).
    topk_uint16_clear_value_tiles_high_bits();

    static_assert(!(FUSED && STABLE_SORT), "fused and comparator-stable modes are mutually exclusive");
    static_assert(!FUSED || is_fp32_dest_acc_en, "fused packed keys require 32-bit DEST");
    static_assert(!(FUSED && TOPK_UINT16_IN_FP32_DEST), "fused keys and uint16-in-fp32-dest are mutually exclusive");
    static_assert(!(RANK_STAMPED && STABLE_SORT), "rank-stamped and comparator-stable modes are mutually exclusive");
    static_assert(!(RANK_STAMPED && FUSED), "rank-stamped and fused-key modes are mutually exclusive");
    static_assert(!RANK_STAMPED || is_fp32_dest_acc_en, "rank-stamped tagged keys require 32-bit DEST");
    static_assert(!(RANK_STAMPED && TOPK_UINT16_IN_FP32_DEST), "rank-stamped keys and uint16-in-fp32-dest are mutually exclusive");
    // Fused packed keys halve the load/store parts of the composite replay windows.
    constexpr int ldst_count       = FUSED ? 4 : 8;   // bare load16/store16 windows
    constexpr int rebuild_win_ld8  = FUSED ? 18 : 22; // load8 + ph1 body + store8 + 8x INCRWC
    constexpr int rebuild_win_ph1  = FUSED ? 18 : 26; // load16 + ph1 body + store16 + 4x INCRWC
    constexpr int rebuild_win_ph2  = FUSED ? 21 : 29; // load16 + ph2 body + store16 + 4x INCRWC
    constexpr int rebuild_win_st12 = FUSED ? 8 : 12;  // store16 + 4x INCRWC at base 13

    if constexpr (STABLE_SORT)
    {
        // Establish the lanes-on/flags-true CC entry invariant once; every stable comparator
        // body re-establishes it via its trailing SFPENCC, and the intervening loads/stores/
        // transposes/SFPCONFIG writes preserve CC state.
        TTI_SFPENCC(3, 0, 0, 10);
    }

    // init replay buffer for rebuild iteration 'm_iter' if uninitialized
    bool init_rebuild = (topk_replay_init != m_iter + 1) ? true : false;

    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            std::uint32_t total_datums_shift = (skip_second & 0x1);
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
            std::uint32_t rebuild_m = m_iter + 1;
            std::uint32_t total_datums_to_compare =
                ((64 >> rebuild_m) < 2 * k) ? 2 * k : (64 >> rebuild_m); // max(2*k, 64/(2^m)) total datums to compare; there's always at least 2*K datums
            total_datums_to_compare = total_datums_to_compare >> total_datums_shift; // Reduce by 2 if skipping last
            std::uint32_t dist      = (k << rebuild_m) > 32 ? 32 : (k << rebuild_m); // min(32, k*2^k)
            std::uint32_t ld_offset = (dist >> 4) * 32 + (dist & 0xF);
            std::uint32_t ld_dist;
            int ph                        = logk - 1;
            bool dir                      = idir;
            std::uint32_t datums_compared = 0;

            switch (ph)
            {
                case 0:

                    break;
                case 1:
                    if (m_iter >= 2)
                    {
                        while (datums_compared < total_datums_to_compare)
                        {
                            if constexpr (STABLE_SORT)
                            {
                                bitonic_topk_load8<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(0, ld_offset);
                                bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                bitonic_topk_store8<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(0, ld_offset);
                                bitonic_topk_inc_x8_dest(64, false);
                            }
                            else
                            {
                                // Groups of 8 datums being sorted at the same time
                                if (init_rebuild)
                                {
                                    lltt::record<lltt::Exec>(0, rebuild_win_ld8);
                                    bitonic_topk_load8<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(0, ld_offset);
                                    bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                    bitonic_topk_store8<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(0, ld_offset);
                                    bitonic_topk_inc_x8_dest(64, false);
                                    init_rebuild = false;
                                }
                                else
                                {
                                    lltt::replay(0, rebuild_win_ld8);
                                }
                            }
                            datums_compared += 16;
                        }
                        break;
                    }
                    else
                    {
                        ld_dist = (ld_offset < 16) ? 4 * ld_offset : 2 * ld_offset;
                        while (datums_compared < total_datums_to_compare)
                        {
                            if constexpr (STABLE_SORT)
                            {
                                bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(ld_offset, ld_dist);
                                bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(ld_offset, ld_dist);
                                TTI_INCRWC(0, 8, 0, 0);
                                TTI_INCRWC(0, 8, 0, 0);
                                TTI_INCRWC(0, 8, 0, 0);
                                TTI_INCRWC(0, 8, 0, 0);
                            }
                            else
                            {
                                // Groups of 16 datums being sorted at the same time
                                if (init_rebuild)
                                {
                                    lltt::record<lltt::Exec>(0, rebuild_win_ph1);
                                    bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(ld_offset, ld_dist);
                                    bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                    bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(ld_offset, ld_dist);
                                    TTI_INCRWC(0, 8, 0, 0);
                                    TTI_INCRWC(0, 8, 0, 0);
                                    TTI_INCRWC(0, 8, 0, 0);
                                    TTI_INCRWC(0, 8, 0, 0);
                                    init_rebuild = false;
                                }
                                else
                                {
                                    lltt::replay(0, rebuild_win_ph1);
                                }
                            }
                            datums_compared += 16;
                        }
                        break;
                    }
                case 2:
                    while (datums_compared < total_datums_to_compare)
                    {
                        if constexpr (STABLE_SORT)
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, ld_offset);
                            bitonic_topk_ph2_st3_to_1<STABLE_SORT>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, ld_offset);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                        }
                        else
                        {
                            // Groups of 16 datums being sorted at the same time
                            if (init_rebuild)
                            {
                                lltt::record<lltt::Exec>(0, rebuild_win_ph2);
                                bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, ld_offset);
                                bitonic_topk_ph2_st3_to_1<STABLE_SORT>();
                                bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, ld_offset);
                                TTI_INCRWC(0, 8, 0, 0);
                                TTI_INCRWC(0, 8, 0, 0);
                                TTI_INCRWC(0, 8, 0, 0);
                                TTI_INCRWC(0, 8, 0, 0);
                                init_rebuild = false;
                            }
                            else
                            {
                                lltt::replay(0, rebuild_win_ph2);
                            }
                        }
                        datums_compared += 16;
                    }
                    break;
                case 3:
                    while (datums_compared < total_datums_to_compare)
                    {
                        if constexpr (STABLE_SORT)
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 8);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, FUSED>(dir, init_rebuild, 8);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, 8);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                            TTI_INCRWC(0, 8, 0, 0);
                        }
                        else
                        {
                            // Groups of 16 datums being sorted at the same time
                            if (init_rebuild)
                            {
                                lltt::record<lltt::Exec>(0, ldst_count);
                                bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 8);
                                bitonic_topk_ph3_st4_to_1<STABLE_SORT, FUSED>(dir, init_rebuild, 8);
                                lltt::record<lltt::Exec>(13, rebuild_win_st12);
                                bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, 8);
                                TTI_INCRWC(0, 8, 0, 0);
                                TTI_INCRWC(0, 8, 0, 0);
                                TTI_INCRWC(0, 8, 0, 0);
                                TTI_INCRWC(0, 8, 0, 0);
                            }
                            else
                            {
                                lltt::replay(0, ldst_count);
                                bitonic_topk_ph3_st4_to_1<STABLE_SORT, FUSED>(dir, init_rebuild, 8);
                                lltt::replay(13, rebuild_win_st12);
                            }
                        }
                        datums_compared += 16;
                        dir = !dir;
                    }
                    break;
                default:
                    std::uint32_t num_steps               = ph + 1;
                    std::uint32_t start_step              = num_steps;
                    std::uint32_t end_step                = 4;
                    std::uint32_t sorted_seq_length       = 1 << num_steps;
                    std::uint32_t total_datums_to_compare = 64;
                    for (std::uint32_t ss = start_step; ss > end_step; ss--)
                    {
                        // Steps N to 5
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                        dir                      = idir;
                        datums_compared          = 0;
                        std::uint32_t dist       = (ss == 5) ? 16 : 32;
                        std::uint32_t inner_d    = dist >> 3; // How many loops to sort the sequence of length (2^ss / 16). Each loop sorts 16
                        std::uint32_t dst_offset = 0;
                        while (datums_compared < total_datums_to_compare)
                        {
                            for (std::uint32_t ii = 0; ii < inner_d; ii++)
                            {
                                bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(
                                    4, 2 * dist); // load/store with offset of face 1 (in row major face layout)
                                bitonic_topk_step_N<STABLE_SORT>(dir);
                                bitonic_topk_store16<is_fp32_dest_acc_en, false, FUSED, RANK_STAMPED>(
                                    4, 2 * dist); // load/store with offset of face 1 (in row major face layout)
                                std::uint32_t dst_inc = 8;
                                dst_offset += dst_inc;
                                bool dst_cr = false;
                                if (ii == (inner_d - 1))
                                {
                                    dst_cr     = true;
                                    dst_inc    = 4 * dist;
                                    dst_offset = 2 * dist;
                                }
                                else if (dst_offset == 16)
                                {
                                    dst_cr  = true;
                                    dst_inc = 32;
                                }
                                bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                datums_compared += 16;
                            }
                            dir = (datums_compared == sorted_seq_length) ? !dir : dir; // total_sorted = total_loops * 16; if total_sorted == sorted_seq_length
                        }
                    }
                    // steps 4 to 1
                    dir             = idir;
                    datums_compared = 0;
                    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                    while (datums_compared < total_datums_to_compare)
                    {
                        if (init_rebuild)
                        {
                            lltt::record<lltt::Exec>(0, ldst_count);
                            bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 8);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, FUSED>(dir, init_rebuild, 8);
                            lltt::record<lltt::Exec>(17, ldst_count);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, 8);
                        }
                        else
                        {
                            lltt::replay(0, ldst_count);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, FUSED>(dir, init_rebuild, 8);
                            lltt::replay(17, ldst_count);
                        }
                        datums_compared += 16;
                        dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                    }
            }

            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
    topk_replay_init = m_iter + 1;
}

inline void _init_topk()
{
    topk_replay_init = 0;
    _sfpu_load_config32_(0xF, 0x0, 0x4); // Set bit [2] of the SFPU_CONTROL_REG to enable index tracking mode
    if constexpr (TOPK_UINT16_IN_FP32_DEST)
    {
        // Mask used to clear garbage high bits when loading UInt16 from 32-bit DEST (LREG12 / vConstIntPrgm0).
        sfpi::vConstIntPrgm0 = 0x0000FFFF;
    }
}

// Rank-stamped init: index tracking ON (the true u32 indices ride the tracked
// swaps at DEST offset 128, as in the plain unfused modes) plus the tag
// complement constant. Written as an explicit set so a preceding fused-mode
// topk in the same kernel cannot leak tracking OFF. The merge programs the
// remaining stamp constants (LREG11/13/14) at each entry, since K and the
// rank base are runtime values there.
inline void _init_topk_rank_stamped_()
{
    topk_replay_init = 0;
    _sfpu_load_config32_(0xF, 0x0, 0x4); // SFPU_CONTROL_REG: ENABLE_DEST_INDEX (bit 2) = 1
    sfpi::vConstIntPrgm0 = 0x0000FFFF;   // LREG12: tag complement operand
}

// Fused-key init: index tracking stays OFF (the packed key carries the index; there is no
// L4-7 bank to mirror). Written as an explicit clear rather than a skip so a preceding
// tracked-mode topk in the same kernel cannot leak the bit in. Programs the fuse/defuse
// mask constant; the sweeps re-program it defensively at each entry as well.
inline void _init_topk_fused_()
{
    topk_replay_init = 0;
    _sfpu_load_config32_(0xF, 0x0, 0x0); // SFPU_CONTROL_REG: ENABLE_DEST_INDEX (bit 2) = 0
    sfpi::vConstIntPrgm0 = 0x0000FFFF;   // LREG12: #50215 mask + tie-complement operand
}

} // namespace sfpu
} // namespace ckernel
