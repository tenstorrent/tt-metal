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

// ---------------------------------------------------------------------------
// EXPERIMENTAL issue-rate knobs for _bitonic_topk_phases_steps (Blackhole).
//
// All three are OFF by default and every use site is #if-guarded, so the
// default build is byte-identical to the unflagged kernel. They exist to
// answer one question with measurements instead of assertion: is the local
// sort limited by the SFPU backend or by the rate at which the math RISC-V
// can issue into it?
//
//   TOPK_HOIST_INIT_GUARDS  Peel d == 0 out of the phase 0/1/2 replay loop so
//                           the `init_load`/`init_phase`/`init_store` branches
//                           leave the loop body. Behaviour-preserving: those
//                           flags can only be true on the first pass.
//
//   TOPK_MOP_INNER_LOOP     Drive the phase 0/1/2 d-loop from the math MOP
//                           expander (one TTI_MOP issue for d = 1..3) instead
//                           of 3 lltt::replay issues per d, using the
//                           ckernel_unpack_template 5-slot form the way
//                           ckernel_sfpu_topk_xl.h does. Implies
//                           TOPK_HOIST_INIT_GUARDS (the recording pass has to
//                           happen before the template can be programmed).
//                           NOTE: this claims the math thread's MOP expander;
//                           any other math-thread MOP user in the same kernel
//                           would have to reprogram after the sort.
//
//   TOPK_REPLAY_STEP_LOAD   Record the phase >= 4 compare/exchange loop's
//                           load16 into the replay buffer once per step and
//                           replay it, so that iteration costs the math RISC-V
//                           1 issue instead of 8. Same instruction stream to
//                           the SFPU; only the number of RISC-V issues changes.
//
//   TOPK_PROBE_RV_NOPS      DIAGNOSTIC ONLY, never ship. Injects exactly N
//                           RISC-V-only instructions (no Tensix issue, no
//                           register pressure) into each iteration of the
//                           phase >= 4 compare/exchange loop. If the kernel is
//                           RISC-V-issue-bound there, runtime grows ~1 cycle
//                           per injected instruction; if it is SFPU-backend
//                           bound, they are free.
// ---------------------------------------------------------------------------
#if defined(TOPK_MOP_INNER_LOOP) && !defined(TOPK_HOIST_INIT_GUARDS)
#define TOPK_HOIST_INIT_GUARDS 1
#endif

#if defined(TOPK_MOP_INNER_LOOP)
#include "ckernel_template.h"
#endif

// Default ON (Blackhole): replay-record the phase >= 4 step loop's load16 and
// store16 so each iteration costs the math RISC-V 2 issues instead of 16. The
// instruction stream reaching the SFPU is identical; only issue count changes.
// Measured op-level on BH silicon (canonical sweep, 2026-08-16): ttnn.topk
// single-core 1.154x-1.202x across N in [4096, 131072], k in [8, 512]; the
// multi-core topk path and every ttnn.sort factory are insensitive (~1.00x).
// Correctness: topk/sort/sampling/moe suites 347 passed with this on,
// stable=True spot-checked, nightly moe-gate consumers green.
// Opt out with -DTOPK_DISABLE_REPLAY_STEP (A/B bisect knob for reviewers).
#if !defined(TOPK_DISABLE_REPLAY_STEP) && !defined(TOPK_REPLAY_STEP_STORE)
#define TOPK_REPLAY_STEP_STORE 1
#endif

#if defined(TOPK_REPLAY_STEP_STORE) && !defined(TOPK_REPLAY_STEP_LOAD)
#define TOPK_REPLAY_STEP_LOAD 1
#endif

#if defined(TOPK_REPLAY_STEP_LOAD)
#if defined(TOPK_REPLAY_STEP_STORE)
// Load AND store recorded: 16 slots are needed, so the window starts at 16 and
// runs to the end of the 32-deep buffer. That overlaps the phase-3 lattice
// (slots 16-20, or 16-24 under STABLE_SORT), which is safe in one direction
// only and for one reason: every consumer of slots >= 16 re-records before it
// replays. The step loop re-records at the top of EVERY step (`init_step_load`
// is step-scoped), and the phase >= 4 "steps 4 to 1" tail that follows the step
// loop re-records the lattice itself, because `init_phase` is still true when
// it is reached. Slots 0-15 (load16(4, 8) / store16(4, 8)) are NOT touched:
// those are recorded once per kernel and replayed for the rest of it.
#define TOPK_STEP_LOAD_REPLAY_START  16
#define TOPK_STEP_STORE_REPLAY_START 24
#else
// Load only: 0-7 hold load16(4, 8), 8-15 store16(4, 8); the phase-3 compare
// lattice occupies 16-20 (or 16-24 under STABLE_SORT, replay_count = 9). The
// buffer is REPLAY_BUF_SIZE = 32 deep. Without STABLE_SORT, [21, 29) does not
// overlap anything. Under STABLE_SORT it shares slots 21-24 with the lattice,
// which is safe for the same reason as the STORE window above: every consumer
// of slots >= 16 re-records before it replays — the step loop re-records at
// the top of EVERY step (`init_step_load` is step-scoped), and the phase >= 4
// "steps 4 to 1" tail re-records the lattice itself, because `init_phase` is
// reset to true at the top of every phase per (face, col) pass.
#define TOPK_STEP_LOAD_REPLAY_START 21
#endif
#endif

#if defined(TOPK_PROBE_RV_NOPS) && (TOPK_PROBE_RV_NOPS > 0)
#define TOPK_PROBE_RV_NOPS_STR1(x) #x
#define TOPK_PROBE_RV_NOPS_STR(x)  TOPK_PROBE_RV_NOPS_STR1(x)
// `.rept` guarantees exactly N instructions survive to the ELF; `addi x0,x0,0`
// writes the zero register, so it needs no scratch and cannot alias anything
// the surrounding code holds live.
#define TOPK_PROBE_RV_NOP_BLOCK() asm volatile(".rept " TOPK_PROBE_RV_NOPS_STR(TOPK_PROBE_RV_NOPS) "\n\taddi x0, x0, 0\n\t.endr\n")
#else
#define TOPK_PROBE_RV_NOP_BLOCK() ((void)0)
#endif

namespace ckernel
{
namespace sfpu
{

static std::int32_t topk_replay_init = 0;

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

// 32 SFPU vectors cover one 32-bit DEST tile at addresses 0,2,...,62. Explicit offsets on
// ADDR_MOD_7 (topk's incr=0 bank) avoid mutating ADDR_MOD_6 used for alt-stores.
// tile_index / store_mode are template parameters so the leaf uses TTI_SFPLOAD/TTI_SFPSTORE
// (ISA-immediate encoding); no RISC-V setup for the operand registers per vector.
#define TOPK_UINT16_STRIP_VEC(base, off, store_mode)                                  \
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, (base) + (off)); \
    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);                                  \
    TTI_SFPSTORE(p_sfpu::LREG0, store_mode, ADDR_MOD_7, (base) + (off))

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

inline void topk_uint16_prepare_value_tile_for_pack(std::uint32_t dst_tile_index)
{
    if constexpr (TOPK_UINT16_IN_FP32_DEST)
    {
        sfpi::vConstIntPrgm0 = 0x0000FFFF;
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
        set_dst_write_addr(0);
    }
    else
    {
        (void)dst_tile_index;
    }
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load8(std::uint32_t offset, std::uint32_t dist)
{
    constexpr std::uint32_t dst_indices_offset = 128; // 2 tile x 64 rows per tile
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    constexpr InstrModLoadStore instr_mod_value = TOPK_UINT16_IN_FP32_DEST ? InstrModLoadStore::INT32 : InstrModLoadStore::DEFAULT;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset   = (offset & 0xF) + face_offset * 32;

    // Load 16 consecutive numbers
    TT_SFPLOAD(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, ld_offset);
    TT_SFPLOAD(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, ld_offset + dist);

    // Load 16 consecutive indices
    TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset);
    TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset + dist);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8(std::uint32_t offset, std::uint32_t dist)
{
    constexpr std::uint32_t dst_indices_offset = 128; // 2 tile x 64 rows per tile
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    constexpr InstrModLoadStore instr_mod_value = TOPK_UINT16_IN_FP32_DEST ? InstrModLoadStore::INT32 : InstrModLoadStore::DEFAULT;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset   = (offset & 0xF) + face_offset * 32;

    // Load 16 consecutive numbers
    TT_SFPSTORE(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, ld_offset);
    TT_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, ld_offset + dist);

    // Load 16 consecutive indices
    TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset + 0);
    TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + ld_offset + dist);
}

template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load16(std::uint32_t dist0, std::uint32_t dist1)
{
    constexpr std::uint32_t dst_indices_offset = 128; // 2 tile x 64 rows per tile
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    constexpr InstrModLoadStore instr_mod_value = TOPK_UINT16_IN_FP32_DEST ? InstrModLoadStore::INT32 : InstrModLoadStore::DEFAULT;

    // Load 16 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPLOAD(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 4);
        TTI_SFPLOAD(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_7, 8);
        TTI_SFPLOAD(p_sfpu::LREG3, instr_mod_value, ADDR_MOD_7, 12);
    }
    else
    {
        TT_SFPLOAD(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 0 + dist0);
        TT_SFPLOAD(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_7, dist1);
        TT_SFPLOAD(p_sfpu::LREG3, instr_mod_value, ADDR_MOD_7, dist1 + dist0);
    }

    // Load 16 consecutive indices
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 4);
        TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 8);
        TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 12);
    }
    else
    {
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 0 + dist0);
        TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, dst_indices_offset + dist1);
        TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, dst_indices_offset + dist1 + dist0);
    }
}

template <bool is_fp32_dest_acc_en, bool alt_addr_mod = false>
inline void bitonic_topk_store16(std::uint32_t dist0, std::uint32_t dist1)
{
    constexpr std::uint32_t dst_indices_offset = 128; // 2 tile x 64 rows per tile
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    constexpr InstrModLoadStore instr_mod_value = TOPK_UINT16_IN_FP32_DEST ? InstrModLoadStore::INT32 : InstrModLoadStore::DEFAULT;

    // Load 16 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, instr_mod_value, ADDR_MOD_7, 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 4);
        TTI_SFPSTORE(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_7, 8);
        TTI_SFPSTORE(p_sfpu::LREG3, instr_mod_value, ADDR_MOD_7, 12);
    }
    else
    {
        TT_SFPSTORE(p_sfpu::LREG1, instr_mod_value, ADDR_MOD_7, 0 + dist0);
        TT_SFPSTORE(p_sfpu::LREG2, instr_mod_value, ADDR_MOD_7, dist1);
        TT_SFPSTORE(p_sfpu::LREG3, instr_mod_value, ADDR_MOD_7, dist1 + dist0);
    }

    // Load 16 consecutive indices
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 4);
        TTI_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 8);
        TTI_SFPSTORE(p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7, dst_indices_offset + 12);
    }
    else
    {
        TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, dst_indices_offset + 0 + dist0);
        TT_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, dst_indices_offset + dist1);
        TT_SFPSTORE(p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7, dst_indices_offset + dist1 + dist0);
    }
}

template <bool STABLE_SORT>
inline void bitonic_topk_ph3_st4_to_1(bool dir, bool &init_replay, int replay_start)
{
    if (dir == static_cast<bool>(SortDir::ArgMin))
    {
        TTI_SFPCONFIG(0x104, 0xF, 1); // Reverse the max/min behaviour of SWAP
        TTI_SFPNOP;
        TTI_SFPNOP;
    }

    constexpr int replay_count = STABLE_SORT ? 9 : 5;

    if (init_replay)
    {
        if constexpr (STABLE_SORT)
        {
            load_replay_buf<Exec>(
                replay_start,
                replay_count,
                []
                {
                    // Step 4
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/2 NOP
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG1/3 NOP
                    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/2 NOP

                    // Step 3 (1-cycle stall: shares LREG3 with Step 4 above)
                    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG2/3 NOP
                    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG2/3 NOP

                    TTI_SFPTRANSP(0, 0, 0, 0);
                });
        }
        else
        {
            load_replay_buf<Exec>(
                replay_start,
                replay_count,
                []
                {
                    // Step 4
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

                    // Step 3
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

                    TTI_SFPTRANSP(0, 0, 0, 0);
                });
        }
        init_replay = false;
    }
    else
    {
        lltt::replay(replay_start, replay_count);
    }

    lltt::replay(replay_start, replay_count);

    if (dir == static_cast<bool>(SortDir::ArgMin))
    {
        TTI_SFPCONFIG(0x004, 0xF, 1); // Restore the max/min behaviour of SWAP
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
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG2/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX); // Hides LREG0/2 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX); // Hides LREG1/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX); // Hides LREG0/2 NOP

    // Step 1 (1-cycle stall: shares LREG1 with Step 2 above)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX); // Hides LREG0/1 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX); // Hides LREG2/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX); // Hides LREG0/1 NOP

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
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX); // Hides LREG0/2 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX); // Hides LREG1/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX); // Hides LREG0/2 NOP

    // Step 1 (1-cycle stall: shares LREG1 with Step 2 above)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX); // Hides LREG0/1 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX); // Hides LREG2/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX); // Hides LREG0/1 NOP

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
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG2/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP

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
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/2 NOP
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG1/3 NOP
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/2 NOP
    }
    else
    {
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/2 NOP
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX); // Hides LREG1/3 NOP
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/2 NOP
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

#if defined(TOPK_MOP_INNER_LOOP)
// Program the math MOP expander so ONE TTI_MOP issue drives the
// (load16 | phase-body | store16) triple that the phase 0/1/2 d-loop repeats.
// The four halo slots hold the whole body with nothing wasted:
//   A0 = REPLAY(0, 8)        load16(4, 8)
//   A1 = REPLAY(16, len)     the phase's compare lattice
//   A2 = REPLAY(8, 7)        the first 7 stores of store16
//   A3 = SFPSTORE LREG7 ...  store16's last store, which rides ADDR_MOD_6
// Splitting store16 across A2/A3 is what makes the body fit the four halo
// slots exactly; replay slots 8..15 still hold the complete store16 for the
// phase 3 / phase >= 4 paths that replay it directly.
template <bool is_fp32_dest_acc_en>
inline void topk_local_sort_mop_config(const std::uint32_t phase_replay_len)
{
    constexpr std::uint32_t dst_indices_offset  = 128;
    constexpr InstrModLoadStore instr_mod_index = is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    constexpr std::uint32_t store_last = TT_OP_SFPSTORE(p_sfpu::LREG7, static_cast<std::uint32_t>(instr_mod_index), ADDR_MOD_6, dst_indices_offset + 12);

    const ckernel_unpack_template tmpl(
        /*unpackB=*/false,
        /*unpackHalo=*/true,
        /*A0_instr=*/lltt::replay_insn(0, 8),
        /*A1_instr=*/lltt::replay_insn(16, phase_replay_len),
        /*A2_instr=*/lltt::replay_insn(8, 7),
        /*A3_instr=*/store_last,
        /*skipA_instr=*/TT_OP_NOP,
        /*B_instr=*/TT_OP_NOP,
        /*skipB_instr=*/TT_OP_NOP);
    tmpl.program();
}
#endif

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void _bitonic_topk_phases_steps(const int idir, const int i_end_phase, const int i_start_phase, const int i_end_step, const int i_start_step)
{
    // If more than 1 phase is requested, do all the steps from all phases
    // If 1 phase is requested, use i_start_step/i_end_step parameters

    // UInt16-in-32b-DEST: clear garbage high bits before compare-swap (#50215).
    topk_uint16_clear_value_tiles_high_bits();

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
                    {
                        constexpr int replay_count = STABLE_SORT ? 6 : 4;
#if defined(TOPK_HOIST_INIT_GUARDS)
                        // d == 0 peeled out: the three init flags can only be
                        // true on the first pass, so their branches do not
                        // belong in the loop body. Instruction stream unchanged.
                        if (init_load)
                        {
                            load_replay_buf<Exec>(0, 8, [] { bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8); });
                            init_load = false;
                        }
                        else
                        {
                            lltt::replay(0, 8);
                        }
                        if (init_phase)
                        {
                            load_replay_buf<Exec>(16, replay_count, [] { bitonic_topk_ph0_st1_to_1<STABLE_SORT>(); });
                            init_phase = false;
                        }
                        else
                        {
                            lltt::replay(16, replay_count);
                        }
                        if (init_store)
                        {
                            load_replay_buf<Exec>(8, 8, [] { bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8); });
                            init_store = false;
                        }
                        else
                        {
                            lltt::replay(8, 8);
                        }
#if defined(TOPK_MOP_INNER_LOOP)
                        topk_local_sort_mop_config<is_fp32_dest_acc_en>(replay_count);
                        ckernel_unpack_template::run(3);
#else
                        for (int d = 1; d < 4; d++)
                        {
                            lltt::replay(0, 8);
                            lltt::replay(16, replay_count);
                            lltt::replay(8, 8);
                        }
#endif
#else
                        for (int d = 0; d < 4; d++)
                        {
                            // Groups of 16 datums being sorted at the same time
                            if (init_load)
                            {
                                load_replay_buf<Exec>(0, 8, [] { bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8); });
                                init_load = false;
                            }
                            else
                            {
                                lltt::replay(0, 8);
                            }
                            if (init_phase)
                            {
                                load_replay_buf<Exec>(16, replay_count, [] { bitonic_topk_ph0_st1_to_1<STABLE_SORT>(); });
                                init_phase = false;
                            }
                            else
                            {
                                lltt::replay(16, replay_count);
                            }
                            if (init_store)
                            {
                                load_replay_buf<Exec>(8, 8, [] { bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8); });
                                init_store = false;
                            }
                            else
                            {
                                lltt::replay(8, 8);
                            }
                        }
#endif
                        break;
                    }
                    case 1:
                    {
                        constexpr int replay_count = STABLE_SORT ? 10 : 6;
#if defined(TOPK_HOIST_INIT_GUARDS)
                        lltt::replay(0, 8);
                        if (init_phase)
                        {
                            load_replay_buf<Exec>(16, replay_count, [] { bitonic_topk_ph1_st2_to_1<STABLE_SORT>(); });
                            init_phase = false;
                        }
                        else
                        {
                            lltt::replay(16, replay_count);
                        }
                        lltt::replay(8, 8);
#if defined(TOPK_MOP_INNER_LOOP)
                        topk_local_sort_mop_config<is_fp32_dest_acc_en>(replay_count);
                        ckernel_unpack_template::run(3);
#else
                        for (int d = 1; d < 4; d++)
                        {
                            lltt::replay(0, 8);
                            lltt::replay(16, replay_count);
                            lltt::replay(8, 8);
                        }
#endif
#else
                        // Groups of 16 datums being sorted at the same time
                        for (int d = 0; d < 4; d++)
                        {
                            lltt::replay(0, 8);
                            if (init_phase)
                            {
                                load_replay_buf<Exec>(16, replay_count, [] { bitonic_topk_ph1_st2_to_1<STABLE_SORT>(); });
                                init_phase = false;
                            }
                            else
                            {
                                lltt::replay(16, replay_count);
                            }
                            lltt::replay(8, 8);
                        }
#endif
                        break;
                    }
                    case 2:
                    {
                        constexpr int replay_count = STABLE_SORT ? 14 : 9;
#if defined(TOPK_HOIST_INIT_GUARDS)
                        lltt::replay(0, 8);
                        if (init_phase)
                        {
                            load_replay_buf<Exec>(16, replay_count, [] { bitonic_topk_ph2_st3_to_1<STABLE_SORT>(); });
                            init_phase = false;
                        }
                        else
                        {
                            lltt::replay(16, replay_count);
                        }
                        lltt::replay(8, 8);
#if defined(TOPK_MOP_INNER_LOOP)
                        topk_local_sort_mop_config<is_fp32_dest_acc_en>(replay_count);
                        ckernel_unpack_template::run(3);
#else
                        for (int d = 1; d < 4; d++)
                        {
                            lltt::replay(0, 8);
                            lltt::replay(16, replay_count);
                            lltt::replay(8, 8);
                        }
#endif
#else
                        for (int d = 0; d < 4; d++)
                        {
                            lltt::replay(0, 8);
                            if (init_phase)
                            {
                                load_replay_buf<Exec>(16, replay_count, [] { bitonic_topk_ph2_st3_to_1<STABLE_SORT>(); });
                                init_phase = false;
                            }
                            else
                            {
                                lltt::replay(16, replay_count);
                            }
                            lltt::replay(8, 8);
                        }
#endif
                        break;
                    }
                    case 3:
                        for (int d = 0; d < 4; d++)
                        {
                            lltt::replay(0, 8);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT>(dir, init_phase, 16);
                            lltt::replay(8, 8);
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
#if defined(TOPK_REPLAY_STEP_LOAD)
                            // `dist` is fixed for the whole step, so this step's
                            // load16 is one fixed 8-instruction sequence. Record
                            // it on the first iteration (which executes it) and
                            // replay it afterwards: 1 RISC-V issue per iteration
                            // instead of 8, with the SFPU seeing the identical
                            // stream. Slots [21, 29) are outside everything the
                            // rest of the kernel uses (0-7 load16(4,8), 8-15
                            // store16, 16-20 the phase-3 lattice), so nothing
                            // this step records has to be re-recorded later.
                            bool init_step_load = true;
#endif
#if defined(TOPK_REPLAY_STEP_STORE)
                            bool init_step_store = true;
#endif
                            while (datums_compared < total_datums_to_compare)
                            {
                                for (std::uint32_t ii = 0; ii < inner_d; ii++)
                                {
                                    // Diagnostic only; expands to nothing unless TOPK_PROBE_RV_NOPS is set.
                                    TOPK_PROBE_RV_NOP_BLOCK();
#if defined(TOPK_REPLAY_STEP_LOAD)
                                    if (init_step_load)
                                    {
                                        load_replay_buf<Exec>(
                                            TOPK_STEP_LOAD_REPLAY_START, 8, [dist] { bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist); });
                                        init_step_load = false;
                                    }
                                    else
                                    {
                                        lltt::replay(TOPK_STEP_LOAD_REPLAY_START, 8);
                                    }
#else
                                    bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist); // load/store with offset of face 1 (in row major face layout)
#endif
                                    bitonic_topk_step_N<STABLE_SORT>(dir);
#if defined(TOPK_REPLAY_STEP_STORE)
                                    if (init_step_store)
                                    {
                                        load_replay_buf<Exec>(
                                            TOPK_STEP_STORE_REPLAY_START, 8, [dist] { bitonic_topk_store16<is_fp32_dest_acc_en, false>(4, 2 * dist); });
                                        init_step_store = false;
                                    }
                                    else
                                    {
                                        lltt::replay(TOPK_STEP_STORE_REPLAY_START, 8);
                                    }
#else
                                    bitonic_topk_store16<is_fp32_dest_acc_en, false>(
                                        4, 2 * dist); // load/store with offset of face 1 (in row major face layout)
#endif
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
                            lltt::replay(0, 8);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT>(dir, init_phase, 16);
                            lltt::replay(8, 8);
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

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min, bool STABLE_SORT = false>
inline void _bitonic_topk_merge(const int m_iter, const int k)
{
    // UInt16-in-32b-DEST: clear garbage high bits before compare-swap (#50215).
    topk_uint16_clear_value_tiles_high_bits();

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
                    bitonic_topk_load8<is_fp32_dest_acc_en>(dst_offset, ld_dist);
                    TTI_SFPSWAP(0, top_min ? p_sfpu::LREG1 : p_sfpu::LREG0, top_min ? p_sfpu::LREG0 : p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                    if constexpr (STABLE_SORT)
                    {
                        // 1-cycle stall: second swap for index tracking on same LREGs
                        TTI_SFPSWAP(0, top_min ? p_sfpu::LREG1 : p_sfpu::LREG0, top_min ? p_sfpu::LREG0 : p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                    }
                    bitonic_topk_store8<is_fp32_dest_acc_en>(dst_offset, ld_dist);
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

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void _bitonic_topk_rebuild(const bool idir, const int m_iter, const int k, const int logk, const int skip_second)
{
    // UInt16-in-32b-DEST: clear garbage high bits before compare-swap (#50215).
    topk_uint16_clear_value_tiles_high_bits();

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
                            // Groups of 8 datums being sorted at the same time
                            if constexpr (STABLE_SORT)
                            {
                                bitonic_topk_load8<is_fp32_dest_acc_en>(0, ld_offset);
                                bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                bitonic_topk_store8<is_fp32_dest_acc_en>(0, ld_offset);
                                bitonic_topk_inc_x8_dest(64, false);
                            }
                            else
                            {
                                if (init_rebuild)
                                {
                                    load_replay_buf<Exec>(
                                        0,
                                        22,
                                        [ld_offset]
                                        {
                                            bitonic_topk_load8<is_fp32_dest_acc_en>(0, ld_offset);
                                            bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                            bitonic_topk_store8<is_fp32_dest_acc_en>(0, ld_offset);
                                            bitonic_topk_inc_x8_dest(64, false);
                                        });
                                    init_rebuild = false;
                                }
                                else
                                {
                                    lltt::replay(0, 22);
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
                                bitonic_topk_load16<is_fp32_dest_acc_en>(ld_offset, ld_dist);
                                bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                bitonic_topk_store16<is_fp32_dest_acc_en, true>(ld_offset, ld_dist);
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
                                    load_replay_buf<Exec>(
                                        0,
                                        26,
                                        [ld_offset, ld_dist]
                                        {
                                            bitonic_topk_load16<is_fp32_dest_acc_en>(ld_offset, ld_dist);
                                            bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(ld_offset, ld_dist);
                                            TTI_INCRWC(0, 8, 0, 0);
                                            TTI_INCRWC(0, 8, 0, 0);
                                            TTI_INCRWC(0, 8, 0, 0);
                                            TTI_INCRWC(0, 8, 0, 0);
                                        });
                                    init_rebuild = false;
                                }
                                else
                                {
                                    lltt::replay(0, 26);
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
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, ld_offset);
                            bitonic_topk_ph2_st3_to_1<STABLE_SORT>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, ld_offset);
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
                                load_replay_buf<Exec>(
                                    0,
                                    29,
                                    [ld_offset]
                                    {
                                        bitonic_topk_load16<is_fp32_dest_acc_en>(4, ld_offset);
                                        bitonic_topk_ph2_st3_to_1<STABLE_SORT>();
                                        bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, ld_offset);
                                        TTI_INCRWC(0, 8, 0, 0);
                                        TTI_INCRWC(0, 8, 0, 0);
                                        TTI_INCRWC(0, 8, 0, 0);
                                        TTI_INCRWC(0, 8, 0, 0);
                                    });
                                init_rebuild = false;
                            }
                            else
                            {
                                lltt::replay(0, 29);
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
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT>(dir, init_rebuild, 8);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
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
                                load_replay_buf<Exec>(0, 8, [] { bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8); });
                                bitonic_topk_ph3_st4_to_1<STABLE_SORT>(dir, init_rebuild, 8);
                                load_replay_buf<Exec>(
                                    13,
                                    12,
                                    []
                                    {
                                        bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                                        TTI_INCRWC(0, 8, 0, 0);
                                        TTI_INCRWC(0, 8, 0, 0);
                                        TTI_INCRWC(0, 8, 0, 0);
                                        TTI_INCRWC(0, 8, 0, 0);
                                    });
                            }
                            else
                            {
                                lltt::replay(0, 8);
                                bitonic_topk_ph3_st4_to_1<STABLE_SORT>(dir, init_rebuild, 8);
                                lltt::replay(13, 12);
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
                                bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist); // load/store with offset of face 1 (in row major face layout)
                                bitonic_topk_step_N<STABLE_SORT>(dir);
                                bitonic_topk_store16<is_fp32_dest_acc_en, false>(4, 2 * dist); // load/store with offset of face 1 (in row major face layout)
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
                            load_replay_buf<Exec>(0, 8, [] { bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8); });
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT>(dir, init_rebuild, 8);
                            load_replay_buf<Exec>(17, 8, [] { bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8); });
                        }
                        else
                        {
                            lltt::replay(0, 8);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT>(dir, init_rebuild, 8);
                            lltt::replay(17, 8);
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

} // namespace sfpu
} // namespace ckernel
