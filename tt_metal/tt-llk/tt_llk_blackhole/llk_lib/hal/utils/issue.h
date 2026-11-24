// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h" // TTI_INSN / TT_INSN, ckernel::instrn_buffer, __instrn_buffer
#include "llk_assert.h"

// Instruction issue and state-CFG MMIO stores the compiler can see.
//
// The HAL describes every instruction as an encoded word. Issued through TTI_INSN / TT_INSN (the generic
// __builtin_rvtt_ttinsn) the compiler cannot tell what a word does, so it keeps every config write around it.
// A compiler that models Tensix state (SFPI with the per-instruction builtins, e.g. __builtin_rvtt_bh_setc16)
// instead gets each word as the builtin of its instruction, and each state-CFG MMIO store as
// __builtin_rvtt_cfg_store, so it can delete the writes it proves redundant. Words it cannot decode at compile time
// stay opaque (the compiler then assumes they may read or write anything). Without the builtins, or with
// HAL_NO_COMPILER_VISIBLE_ISSUE or TT_LLK_NO_INTRINSIC_REDIRECT defined, every path below expands to exactly the
// TTI_INSN / TT_INSN / volatile store it replaces.
#if defined(__riscv_xtttensixbh) && __has_builtin(__builtin_rvtt_bh_setc16) && __has_builtin(__builtin_rvtt_cfg_store) && \
    !defined(TT_LLK_NO_INTRINSIC_REDIRECT) && !defined(HAL_NO_COMPILER_VISIBLE_ISSUE)
#define HAL_COMPILER_VISIBLE_ISSUE 1
#include "visible_issue_table.h"
#else
#define HAL_COMPILER_VISIBLE_ISSUE 0
#endif

#if HAL_COMPILER_VISIBLE_ISSUE
namespace hal::detail
{
/**
 * @brief An instruction word from its opcode and its packed fields (what TT_OP expands to in HAL code).
 *
 * The fields of every TT_OP_* encoding lie below bit 24. Saying so lets the compiler prove the opcode of a word whose
 * fields are run-time values: issue_buffered needs `word >> 24` to fold, and with a plain `(opcode << 24) + fields`
 * it folds only once every field is a constant, so the word stays opaque and the opcode switch stays in every caller
 * until then. No code is generated; a field that overflows its width is a caller bug either way.
 */
inline constexpr __attribute__((always_inline)) std::uint32_t op_word(const std::uint32_t opcode, const std::uint32_t fields)
{
    if ((fields >> 24) != 0u)
    {
        __builtin_unreachable();
    }
    return (opcode << 24) + fields;
}
} // namespace hal::detail
#undef TT_OP
#define TT_OP(opcode, params) (::hal::detail::op_word((opcode), (params)))
#endif

namespace hal::detail
{

#if defined(HAL_REPORT_OPAQUE_ISSUE)
/// Diagnostic (HAL_REPORT_OPAQUE_ISSUE): each compile-time word issued opaquely names itself in a -Wdeprecated warning.
template <std::uint32_t Word>
[[deprecated("opaque instruction word")]] inline constexpr void report_opaque_word()
{
}
#endif

/**
 * @brief Issue a compile-time instruction word as one immediate instruction (replaces TTI_INSN).
 *
 * @tparam Word: Encoded instruction.
 */
template <std::uint32_t Word>
inline __attribute__((always_inline)) void issue()
{
#if HAL_COMPILER_VISIBLE_ISSUE
    constexpr std::uint32_t id = visible::id_of(Word);
    if constexpr (id != 0)
    {
        visible::Emit<id>::template word<Word>();
    }
    else
#endif
    {
#if defined(HAL_REPORT_OPAQUE_ISSUE) && HAL_COMPILER_VISIBLE_ISSUE
        report_opaque_word<Word>();
#endif
        TTI_INSN(Word);
    }
}

/**
 * @brief Issue an instruction word that is a constant after inlining as one immediate instruction (replaces TTI_INSN
 *        where the word is not a constant expression).
 *
 * @param word: Encoded instruction; must fold to a constant.
 */
inline __attribute__((always_inline)) void issue_immediate(const std::uint32_t word)
{
#if HAL_COMPILER_VISIBLE_ISSUE
    if (__builtin_constant_p(word) && visible::issue_word(word, true))
    {
        return;
    }
#endif
    TTI_INSN(word);
}

/**
 * @brief Issue an instruction word through the instruction buffer (replaces TT_INSN). A word that folds to a constant
 *        after inlining, or whose opcode does, is issued through its instruction's builtin instead; a word whose
 *        opcode is a run-time value stays opaque.
 *
 * @param word: Encoded instruction.
 */
inline __attribute__((always_inline)) void issue_buffered(const std::uint32_t word)
{
#if HAL_COMPILER_VISIBLE_ISSUE
    // A constant word is issued as its builtin when it round-trips through the instruction's fields. A run-time word
    // of a compile-time instruction is issued as its builtin with run-time field operands, so the compiler still knows
    // which instruction it is and what state it touches. One switch on the opcode, on scalars only (see the table).
    if (__builtin_constant_p(word >> 24) && visible::issue_word(word, __builtin_constant_p(word)))
    {
        return;
    }
#if defined(HAL_REPORT_OPAQUE_ISSUE)
    extern void hal_report_opaque_runtime_word(std::uint32_t) __attribute__((warning("opaque runtime instruction word")));
    hal_report_opaque_runtime_word(__builtin_constant_p(word >> 24) ? (word >> 24) : 0xffu);
#endif
#endif
    TT_INSN(word);
}

/**
 * @brief Store one state-CFG word through MMIO.
 *
 * @param cfg: State-CFG bank base.
 * @param index: Word index within the bank.
 * @param value: Complete word.
 */
inline __attribute__((always_inline)) void cfg_store(volatile std::uint32_t tt_reg_ptr* cfg, const std::uint32_t index, const std::uint32_t value)
{
#if HAL_COMPILER_VISIBLE_ISSUE
    __builtin_rvtt_cfg_store(reinterpret_cast<std::uint32_t>(cfg), index, value);
#else
    cfg[index] = value;
#endif
}

/**
 * @brief Wait until the RISC-V's earlier MopCfg writes have landed (replaces ckernel::mop_sync).
 *
 * @note The inline-asm ckernel::mop_sync hides the sync from the compiler, which then keeps every MopCfg write; the
 *       builtin's constant address tells it the store cannot reach config.
 */
inline __attribute__((always_inline)) void mop_sync()
{
#if HAL_COMPILER_VISIBLE_ISSUE
    __builtin_rvtt_store_blocking(PC_BUF_BASE + 8u, 0u); // &ckernel::pc_buf_base[2]
#else
    ckernel::mop_sync();
#endif
}

/** @brief Wait until the Tensix thread has drained (replaces ckernel::tensix_sync); see @ref mop_sync. */
inline __attribute__((always_inline)) void tensix_sync()
{
#if HAL_COMPILER_VISIBLE_ISSUE
    __builtin_rvtt_store_blocking(PC_BUF_BASE + 4u, 0u); // &ckernel::pc_buf_base[1]
#else
    ckernel::tensix_sync();
#endif
}

} // namespace hal::detail

// Field-wise issue for instructions whose fields are runtime values (the TT_* macros) or whose immediate form must stay
// an expression (the TTI_* macros). With the builtins, each is the builtin of its instruction with the same operands,
// so the compiler sees which state a runtime value goes to; without them, each is the tt-llk macro it names.
#if HAL_COMPILER_VISIBLE_ISSUE
#define HAL_TT_SETC16(Reg, Value)                 (::__instrn_buffer[0] = __builtin_rvtt_bh_setc16((Reg), (Value)))
#define HAL_TTI_SETC16(Reg, Value)                HAL_TT_SETC16(Reg, Value)
#define HAL_TT_RMWCIB0(Mask, Data, Addr)          (::__instrn_buffer[0] = __builtin_rvtt_rmwciB0((Mask), (Data), (Addr)))
#define HAL_TT_RMWCIB1(Mask, Data, Addr)          (::__instrn_buffer[0] = __builtin_rvtt_rmwciB1((Mask), (Data), (Addr)))
#define HAL_TT_RMWCIB2(Mask, Data, Addr)          (::__instrn_buffer[0] = __builtin_rvtt_rmwciB2((Mask), (Data), (Addr)))
#define HAL_TT_RMWCIB3(Mask, Data, Addr)          (::__instrn_buffer[0] = __builtin_rvtt_rmwciB3((Mask), (Data), (Addr)))
#define HAL_TTI_RMWCIB0(Mask, Data, Addr)         HAL_TT_RMWCIB0(Mask, Data, Addr)
#define HAL_TTI_RMWCIB1(Mask, Data, Addr)         HAL_TT_RMWCIB1(Mask, Data, Addr)
#define HAL_TTI_RMWCIB2(Mask, Data, Addr)         HAL_TT_RMWCIB2(Mask, Data, Addr)
#define HAL_TTI_RMWCIB3(Mask, Data, Addr)         HAL_TT_RMWCIB3(Mask, Data, Addr)
#define HAL_TT_WRCFG(Gpr, Wr128b, CfgReg)         (::__instrn_buffer[0] = __builtin_rvtt_wrcfg((Gpr), (Wr128b), (CfgReg)))
#define HAL_TTI_WRCFG(Gpr, Wr128b, CfgReg)        HAL_TT_WRCFG(Gpr, Wr128b, CfgReg)
#define HAL_TTI_RDCFG(Gpr, CfgReg)                (::__instrn_buffer[0] = __builtin_rvtt_rdcfg((Gpr), (CfgReg)))
#define HAL_TT_SETDMAREG(Payload, Value, Mode, Reg) \
    (::__instrn_buffer[0] = __builtin_rvtt_setdmareg((Payload), (Value), (Mode), (Reg)))
#define HAL_TTI_SETDMAREG(Payload, Value, Mode, Reg) HAL_TT_SETDMAREG(Payload, Value, Mode, Reg)
#define HAL_TT_STALLWAIT(Stall, Wait)             (::__instrn_buffer[0] = __builtin_rvtt_stallwait((Stall), (Wait)))
#if __has_builtin(__builtin_rvtt_stallwait_scoped) && __has_builtin(__builtin_rvtt_stall_scope_end)
// A STALLWAIT that guards only the instructions up to HAL_STALL_SCOPE_END: the compiler deletes it when none of them
// needs it (a plain STALLWAIT holds back everything issued after it, so it stays whenever anything follows).
#define HAL_TTI_STALLWAIT_SCOPED(Stall, Wait)     (::__instrn_buffer[0] = __builtin_rvtt_stallwait_scoped((Stall), (Wait)))
#define HAL_STALL_SCOPE_END()                     __builtin_rvtt_stall_scope_end()
#else
#define HAL_TTI_STALLWAIT_SCOPED(Stall, Wait)     HAL_TT_STALLWAIT(Stall, Wait)
#define HAL_STALL_SCOPE_END()                     ((void)0)
#endif
#define HAL_TT_ATGETM(Mutex)                      (::__instrn_buffer[0] = __builtin_rvtt_atgetm((Mutex)))
#define HAL_TT_ATRELM(Mutex)                      (::__instrn_buffer[0] = __builtin_rvtt_atrelm((Mutex)))
#define HAL_TT_MOP(Type, Count, Mask)             (::__instrn_buffer[0] = __builtin_rvtt_bh_mop((Type), (Count), (Mask)))
#define HAL_TTI_MOP(Type, Count, Mask)            HAL_TT_MOP(Type, Count, Mask)
#define HAL_TT_MOP_CFG(MaskHigh)                  (::__instrn_buffer[0] = __builtin_rvtt_mopcfg((MaskHigh)))
#define HAL_TTI_MOP_CFG(MaskHigh)                 HAL_TT_MOP_CFG(MaskHigh)
#define HAL_TTI_NOP                               (::__instrn_buffer[0] = __builtin_rvtt_ttnop())
#else
#define HAL_TT_SETC16(Reg, Value)                 TT_SETC16(Reg, Value)
#define HAL_TTI_SETC16(Reg, Value)                TTI_SETC16(Reg, Value)
#define HAL_TT_RMWCIB0(Mask, Data, Addr)          TT_RMWCIB0(Mask, Data, Addr)
#define HAL_TT_RMWCIB1(Mask, Data, Addr)          TT_RMWCIB1(Mask, Data, Addr)
#define HAL_TT_RMWCIB2(Mask, Data, Addr)          TT_RMWCIB2(Mask, Data, Addr)
#define HAL_TT_RMWCIB3(Mask, Data, Addr)          TT_RMWCIB3(Mask, Data, Addr)
#define HAL_TTI_RMWCIB0(Mask, Data, Addr)         TTI_RMWCIB0(Mask, Data, Addr)
#define HAL_TTI_RMWCIB1(Mask, Data, Addr)         TTI_RMWCIB1(Mask, Data, Addr)
#define HAL_TTI_RMWCIB2(Mask, Data, Addr)         TTI_RMWCIB2(Mask, Data, Addr)
#define HAL_TTI_RMWCIB3(Mask, Data, Addr)         TTI_RMWCIB3(Mask, Data, Addr)
#define HAL_TT_WRCFG(Gpr, Wr128b, CfgReg)         TT_WRCFG(Gpr, Wr128b, CfgReg)
#define HAL_TTI_WRCFG(Gpr, Wr128b, CfgReg)        TTI_WRCFG(Gpr, Wr128b, CfgReg)
#define HAL_TTI_RDCFG(Gpr, CfgReg)                TTI_RDCFG(Gpr, CfgReg)
#define HAL_TT_SETDMAREG(Payload, Value, Mode, Reg)  TT_SETDMAREG(Payload, Value, Mode, Reg)
#define HAL_TTI_SETDMAREG(Payload, Value, Mode, Reg) TTI_SETDMAREG(Payload, Value, Mode, Reg)
#define HAL_TT_STALLWAIT(Stall, Wait)             TT_STALLWAIT(Stall, Wait)
#define HAL_TTI_STALLWAIT_SCOPED(Stall, Wait)     TTI_STALLWAIT(Stall, Wait)
#define HAL_STALL_SCOPE_END()                     ((void)0)
#define HAL_TT_ATGETM(Mutex)                      TT_ATGETM(Mutex)
#define HAL_TT_ATRELM(Mutex)                      TT_ATRELM(Mutex)
#define HAL_TT_MOP(Type, Count, Mask)             TT_MOP(Type, Count, Mask)
#define HAL_TTI_MOP(Type, Count, Mask)            TTI_MOP(Type, Count, Mask)
#define HAL_TT_MOP_CFG(MaskHigh)                  TT_MOP_CFG(MaskHigh)
#define HAL_TTI_MOP_CFG(MaskHigh)                 TTI_MOP_CFG(MaskHigh)
#define HAL_TTI_NOP                               TTI_NOP
#endif
