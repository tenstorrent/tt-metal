// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Wormhole TRISC helpers: TRISC i pushes Tensix thread i, from tt_llk_wormhole_b0 and the WormholeB0 ISA documents.
#pragma once

// cfg_defines.h and ckernel_ops.h come from tt-metal (chlkc_list.h -> ckernel.h)

#include <stdint.h>

#define WH_REG(addr) (*(volatile uint32_t*)(uintptr_t)(addr))

static constexpr uint32_t wh_tensix_gpr = 0xFFE00000u;   // the issuing thread's 64 GPRs
static constexpr uint32_t wh_tensix_insn = 0xFFE40000u;  // push to the issuing thread
static constexpr uint32_t wh_tensix_cfg = 0xFFEF0000u;   // backend configuration, state 0
static constexpr uint32_t wh_wall_clock_lo = 0xFFB121F0u;
static constexpr uint32_t wh_wall_clock_hi = 0xFFB121F8u;

// Data formats.
static constexpr uint32_t wh_fmt_fp32 = 0u, wh_fmt_tf32 = 4u, wh_fmt_bf16 = 5u;

// STALLWAIT and SEMWAIT block masks (STALLWAIT.md B0-B8).
static constexpr uint32_t wh_block_tdma = 0x1u, wh_block_sync = 0x2u, wh_block_pack = 0x4u, wh_block_unpack = 0x8u,
                          wh_block_thcon = 0x20u, wh_block_math = 0x40u, wh_block_cfg = 0x80u, wh_block_sfpu = 0x100u;
// STALLWAIT wait conditions (C0-C14).
static constexpr uint32_t wh_wait_thcon = 0x1u, wh_wait_unpack0 = 0x2u, wh_wait_unpack1 = 0x4u, wh_wait_pack = 0x78u,
                          wh_wait_math = 0x80u, wh_wait_srca_valid = 0x400u, wh_wait_srcb_valid = 0x800u,
                          wh_wait_trisc_cfg = 0x2000u, wh_wait_sfpu = 0x4000u;
static constexpr uint32_t wh_wait_idle =
    wh_wait_thcon | wh_wait_unpack0 | wh_wait_unpack1 | wh_wait_pack | wh_wait_math | wh_wait_sfpu | wh_wait_trisc_cfg;
// SEMWAIT conditions.
static constexpr uint32_t wh_sem_on_zero = 0x1u, wh_sem_on_max = 0x2u;

// SETADC counter sets and dimensions.
static constexpr uint32_t wh_adc_unpack0 = 1u, wh_adc_unpack1 = 2u, wh_adc_pack = 4u;
static constexpr uint32_t wh_adc_x = 0u, wh_adc_y = 1u, wh_adc_z = 2u, wh_adc_w = 3u;

// ZEROACC modes.
static constexpr uint32_t wh_zeroacc_16_rows_16b = 1u, wh_zeroacc_half_16b = 2u, wh_zeroacc_all_16b = 3u,
                          wh_zeroacc_half_32b = 6u, wh_zeroacc_all_32b = 7u;

// Pushes `instruction` to this TRISC's thread: a .ttinsn word when constant, else a store, which on Wormhole is the
// same push (PushTensixInstruction.md), so the two kinds stay in order.
[[gnu::always_inline]] static inline void wh_push(uint32_t instruction) {
    if (__builtin_constant_p(instruction)) {
        __asm__ volatile(".ttinsn %0" : : "n"(instruction));
    } else {
        WH_REG(wh_tensix_insn) = instruction;
    }
}

static inline void wh_cfg(uint32_t index, uint32_t value) { WH_REG(wh_tensix_cfg + index * 4u) = value; }

static inline volatile uint32_t& wh_gpr(uint32_t index) { return WH_REG(wh_tensix_gpr + index * 4u); }

// As wh_noc.h's: the high word latch is the tile's, shared with the other RISCs' low word reads.
static inline uint64_t wh_wall_clock() {
    for (;;) {
        const uint32_t lo = WH_REG(wh_wall_clock_lo);
        const uint32_t hi = WH_REG(wh_wall_clock_hi);
        if (WH_REG(wh_wall_clock_lo) > lo) {
            return (uint64_t)hi << 32 | lo;
        }
    }
}

// The L1 base field of the unpacker and packer address registers: 16-byte units, less the header unit.
static inline uint32_t wh_l1_base(uint32_t addr) { return addr / 16u - 1u; }

// Writes `value` to GPR pair `gpr` in the instruction stream, so a later WRCFG of it follows earlier instructions.
static inline void wh_set_gpr(uint32_t gpr, uint32_t value) {
    wh_push(TT_OP_SETDMAREG(0u, value & 0xFFFFu, 0u, gpr * 2u));
    wh_push(TT_OP_SETDMAREG(0u, value >> 16, 0u, gpr * 2u + 1u));
}

// A configuration write in the instruction stream; its readers need two instructions in between (WRCFG.md).
static inline void wh_write_cfg(uint32_t gpr, uint32_t index, uint32_t value) {
    wh_set_gpr(gpr, value);
    wh_push(TT_OP_WRCFG(gpr, 0u, index));
    wh_push(TT_OP_DMANOP);
    wh_push(TT_OP_DMANOP);
}

// Returns once the units in `wait` are idle for this thread and every earlier instruction has passed, through GPR 63.
static inline void wh_thread_idle(uint32_t wait) {
    volatile uint32_t& done = wh_gpr(63u);
    done = 0u;
    while (done) {
    }
    wh_push(TT_OP_STALLWAIT(wh_block_thcon | wh_block_sync, wait));
    wh_push(TT_OP_SETDMAREG(0u, 1u, 0u, 126u));
    while (!done) {
    }
}

// The issuing thread's matrix AddrMod `index` (0-3), mirrored into 4-7 against a stale ADDR_MOD_SET_Base.
static inline void wh_math_addr_mod(uint32_t index, uint32_t ab, uint32_t dst) {
    for (uint32_t bank = 0; bank < 8u; bank += 4u) {
        wh_push(TT_OP_SETC16(ADDR_MOD_AB_SEC0_SrcAIncr_ADDR32 + 2u * (index + bank), ab));
        wh_push(TT_OP_SETC16(ADDR_MOD_DST_SEC0_DestIncr_ADDR32 + index + bank, dst));
    }
}

// Thread state that survives a soft reset (dsv41-wh worklist), zeroed for the issuing thread: configuration state 0,
// Dst offset, AddrMod bank, Src bank overrides, dvalid clear disables, fidelity base, unpacker context offsets, FP16A,
// bias AddrMods, every ADC and RWC.
static inline void wh_thread_reset() {
    static constexpr uint32_t zero[] = {
        CFG_STATE_ID_StateID_ADDR32,
        DEST_TARGET_REG_CFG_MATH_Offset_ADDR32,
        ADDR_MOD_SET_Base_ADDR32,
        SRCA_SET_Base_ADDR32,
        SRCB_SET_Base_ADDR32,
        CLR_DVALID_SrcA_Disable_ADDR32,
        FIDELITY_BASE_Phase_ADDR32,
        UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32,
        FPU_BIAS_SEL_Pointer_ADDR32,
        FP16A_FORCE_Enable_ADDR32};
    for (const uint32_t index : zero) {
        wh_push(TT_OP_SETC16(index, 0u));
    }
    for (uint32_t i = 0; i < 8u; ++i) {
        wh_push(TT_OP_SETC16(ADDR_MOD_BIAS_SEC0_BiasIncr_ADDR32 + i, 0x10u));
        wh_push(TT_OP_SETC16(ADDR_MOD_AB_SEC0_BiasCntIncr_ADDR32 + 2u * i, 0u));
    }
    wh_push(TT_OP_SETADCXY(7u, 0u, 0u, 0u, 0u, 0xFu));
    wh_push(TT_OP_SETADCZW(7u, 0u, 0u, 0u, 0u, 0xFu));
    wh_push(TT_OP_SETRWC(0u, 0u, 0u, 0u, 0u, 0xFu));
}

static inline void wh_put_time(volatile uint32_t* at) {
    const uint64_t now = wh_wall_clock();
    at[0] = (uint32_t)now;
    at[1] = (uint32_t)(now >> 32);
}

// The WH benches' TRISC protocol over a bench block B (go, ready[3], done[3], start[3][2], end[3][2]) that the host
// writes before releasing the TRISCs: each TRISC sets its units up, meets the other two in ready[], then runs, waits
// for its thread to go idle and reports in done[], timing run on the wall clock.
template <class B>
[[noreturn]] void wh_bench_main(
    volatile B& b, uint32_t index, void (*setup)(const volatile B&), void (*run)(const volatile B&)) {
    const uint32_t go = b.go;
    setup(b);
    b.ready[index] = go;
    for (uint32_t i = 0; i < 3u; ++i) {
        while (b.ready[i] != go) {
        }
    }
    wh_put_time(b.start[index]);
    run(b);
    wh_thread_idle(wh_wait_idle);
    wh_put_time(b.end[index]);
    b.done[index] = go;
    for (;;) {
    }
}
