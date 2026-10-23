// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if __riscv_xtttensixqsr || (__clang__ && defined(ARCH_QUASAR) && defined(COMPILE_FOR_TRISC))
#if __clang__
// User is likely using static analysis tool.  Provide a declaration.
extern "C" void __builtin_rvtt_ttinsn(volatile unsigned long *, unsigned, unsigned, unsigned, unsigned) __attribute__((nothrow));
#endif
#define TT_IP_INSN(IP, ENC, MOD, DST, SRC) __builtin_rvtt_ttinsn((IP), (ENC), MOD, DST, SRC)
#elif defined(ARCH_QUASAR) && defined(LLK_BOOT_BRISC)
// The llk test infra uses TTI macros on brisc and somehow executes
// it. So icky. See #58141
#define TT_IP_INSN(IP, ENC, MOD, DST, SRC)                                                                            \
    (((IP) ? (({ __asm__ __volatile__(".error \"TT_IP_INSN in non-tensix code\""); }), void(ENC))                     \
           : ({ __asm__ __volatile__(".4byte ((%0 >> 30) & 3) | ((%0 & 0x3fffffff) << 2)" ::"n"(unsigned(ENC))); })), \
     void(MOD),                                                                                                       \
     void(DST),                                                                                                       \
     void(SRC))
#else
// We can't just make these compile errors, because our headers are
// not sufficiently clean.  Turn them into assembler errors, should they
// ever get that far.
#define TT_IP_INSN(IP, ENC, MOD, DST, SRC) \
    (({ __asm__ __volatile__(".error \"TT_IP_INSN in non-tensix code\""); }), void(IP), void(ENC), void(MOD), void(DST), void(SRC))
#endif

#define TT_INSN(ENC)          TT_IP_INSN(::ckernel::instrn_buffer, (ENC), 0, 0, 0)
#define TTI_INSN(ENC)         TT_IP_INSN(nullptr, (ENC), 0, 0, 0)
#define TT_OP(opcode, params) ((opcode << 24) + params)

#define TT_OP_ADDGPR(OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_OP(0x58, (((OpB_is_Const) << 23) + ((Result_GPR_Index) << 12) + ((OpB_GPR_Index) << 6) + ((OpA_GPR_Index) << 0)))
#define TT_IP_ADDGPR(IP, OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_INSN(IP, TT_OP_ADDGPR(OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index), 0, 0, 0)
#define TT_ADDGPR(OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_ADDGPR(::ckernel::instrn_buffer, OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index)
#define TTI_ADDGPR(OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_ADDGPR(nullptr, OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index)

#define TT_OP_ATCAS(SwapVal, CmpVal, Sel32b, Addr_GPR_Index)     TT_OP(0x64, (((SwapVal) << 18) + ((CmpVal) << 14) + ((Sel32b) << 12) + ((Addr_GPR_Index) << 0)))
#define TT_IP_ATCAS(IP, SwapVal, CmpVal, Sel32b, Addr_GPR_Index) TT_IP_INSN(IP, TT_OP_ATCAS(SwapVal, CmpVal, Sel32b, Addr_GPR_Index), 0, 0, 0)
#define TT_ATCAS(SwapVal, CmpVal, Sel32b, Addr_GPR_Index)        TT_IP_ATCAS(::ckernel::instrn_buffer, SwapVal, CmpVal, Sel32b, Addr_GPR_Index)
#define TTI_ATCAS(SwapVal, CmpVal, Sel32b, Addr_GPR_Index)       TT_IP_ATCAS(nullptr, SwapVal, CmpVal, Sel32b, Addr_GPR_Index)

#define TT_OP_ATGETM(mutex_index)          TT_OP(0xa0, (((mutex_index) << 0)))
#define TT_IP_ATGETM_IPTR(IP, mutex_index) TT_IP_INSN(IP, TT_OP_ATGETM(mutex_index), 0, 0, 0)
#define TT_ATGETM(mutex_index)             TT_IP_ATGETM(::ckernel::instrn_buffer, mutex_index)
#define TTI_ATGETM(mutex_index)            TT_IP_ATGETM(nullptr, mutex_index)

#define TT_OP_ATINCGET(WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index) \
    TT_OP(0x61, (((WrapVal) << 14) + ((Sel32b) << 12) + ((Data_GPR_Index) << 6) + ((Addr_GPR_Index) << 0)))
#define TT_IP_ATINCGET(IP, WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index) \
    TT_IP_INSN(IP, TT_OP_ATINCGET(WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index), 0, 0, 0)
#define TT_ATINCGET(WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index) \
    TT_IP_ATINCGET_IPTR(::ckernel::instrn_buffer, WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index)
#define TTI_ATINCGET(WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index) TT_IP_ATINCGET_IPTR(nullptr, WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index)

#define TT_OP_ATINCGETPTR(NoIncr, IncrVal, WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index) \
    TT_OP(0x62, (((NoIncr) << 22) + ((IncrVal) << 18) + ((WrapVal) << 14) + ((Sel32b) << 12) + ((Data_GPR_Index) << 6) + ((Addr_GPR_Index) << 0)))
#define TT_IP_ATINCGETPTR(IP, NoIncr, IncrVal, WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index) \
    TT_IP_INSN(IP, TT_OP_ATINCGETPTR(NoIncr, IncrVal, WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index), 0, 0, 0)
#define TT_ATINCGETPTR(NoIncr, IncrVal, WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index) \
    TT_IP_ATINCGETPTR(::ckernel::instrn_buffer, NoIncr, IncrVal, WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index)
#define TTI_ATINCGETPTR(NoIncr, IncrVal, WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index) \
    TT_IP_ATINCGETPTR(nullptr, NoIncr, IncrVal, WrapVal, Sel32b, Data_GPR_Index, Addr_GPR_Index)

#define TT_OP_ATRELM(mutex_index)     TT_OP(0xa1, (((mutex_index) << 0)))
#define TT_IP_ATRELM(IP, mutex_index) TT_IP_INSN(IP, TT_OP_ATRELM(mutex_index), 0, 0, 0)
#define TT_ATRELM(mutex_index)        TT_OP_ATRELM(mutex_index)
#define TTI_ATRELM(mutex_index)       TT_OP_ATRELM(nullptr, mutex_index)

#define TT_OP_ATSWAP(SwapMask, Data_GPR_Index, Addr_GPR_Index)     TT_OP(0x63, (((SwapMask) << 14) + ((Data_GPR_Index) << 6) + ((Addr_GPR_Index) << 0)))
#define TT_IP_ATSWAP(IP, SwapMask, Data_GPR_Index, Addr_GPR_Index) TT_IP_INSN(IP, TT_OP_ATSWAP(SwapMask, Data_GPR_Index, Addr_GPR_Index), 0, 0, 0)
#define TT_ATSWAP(SwapMask, Data_GPR_Index, Addr_GPR_Index)        TT_IP_ATSWAP(::ckernel::instrn_buffer, SwapMask, Data_GPR_Index, Addr_GPR_Index)
#define TTI_ATSWAP(SwapMask, Data_GPR_Index, Addr_GPR_Index)       TT_IP_ATSWAP(nullptr, SwapMask, Data_GPR_Index, Addr_GPR_Index)

#define TT_OP_BITWOPGPR(OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_OP(0x5b, (((OpB_is_Const) << 23) + ((OpSel) << 18) + ((Result_GPR_Index) << 12) + ((OpB_GPR_Index) << 6) + ((OpA_GPR_Index) << 0)))
#define TT_IP_BITWOPGPR(IP, OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_INSN(IP, TT_OP_BITWOPGPR(OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index), 0, 0, 0)
#define TT_BITWOPGPR(OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_BITWOPGPR(::ckernel::instrn_buffer, OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index)
#define TTI_BITWOPGPR(OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_BITWOPGPR(nullptr, OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index)

#define TT_OP_CFGSHIFTMASK(CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel)                       \
    TT_OP(                                                                                                                                  \
        0xba,                                                                                                                               \
        (((CfgRegAddr) << 16) + ((disable_mask_on_old_val) << 15) + ((operation) << 12) + ((mask_width) << 7) + ((right_cshift_amt) << 2) + \
         ((scratch_sel) << 0)))
#define TT_IP_CFGSHIFTMASK(IP, CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel) \
    TT_IP_INSN(IP, TT_OP_CFGSHIFTMASK(CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel), 0, 0, 0)
#define TT_CFGSHIFTMASK(CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel) \
    TT_IP_CFGSHIFTMASK(::ckernel::instrn_buffer, CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel)
#define TTI_CFGSHIFTMASK(CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel) \
    TT_IP_CFGSHIFTMASK(nullptr, CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel)

#define TT_OP_CFGSHIFTMASK_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(                                                              \
    CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel)                                              \
    TT_OP(                                                                                                                                  \
        0xbb,                                                                                                                               \
        (((CfgRegAddr) << 16) + ((disable_mask_on_old_val) << 15) + ((operation) << 12) + ((mask_width) << 7) + ((right_cshift_amt) << 2) + \
         ((scratch_sel) << 0)))
#define TT_IP_CFGSHIFTMASK_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(                          \
    IP, CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel)      \
    TT_IP_INSN(                                                                                         \
        IP,                                                                                             \
        TT_OP_CFGSHIFTMASK_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(                          \
            CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel), \
        0,                                                                                              \
        0,                                                                                              \
        0)
#define TT_CFGSHIFTMASK_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(                    \
    CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel) \
    TT_IP_CFGSHIFTMASK_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(                     \
        ::ckernel::instrn_buffer, CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel)
#define TTI_CFGSHIFTMASK_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(                   \
    CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel) \
    TT_IP_CFGSHIFTMASK_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(                     \
        nullptr, CfgRegAddr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel)

#define TT_OP_CLEARDVALID(cleardvalid, cleardvalid_S, dest_dvalid_reset, dest_dvalid_client_bank_reset, dest_pulse_last, reset)                              \
    TT_OP(                                                                                                                                                   \
        0x36,                                                                                                                                                \
        (((cleardvalid) << 22) + ((cleardvalid_S) << 20) + ((dest_dvalid_reset) << 10) + ((dest_dvalid_client_bank_reset) << 6) + ((dest_pulse_last) << 2) + \
         ((reset) << 0)))
#define TT_IP_CLEARDVALID(IP, cleardvalid, cleardvalid_S, dest_dvalid_reset, dest_dvalid_client_bank_reset, dest_pulse_last, reset) \
    TT_IP_INSN(IP, TT_OP_CLEARDVALID(cleardvalid, cleardvalid_S, dest_dvalid_reset, dest_dvalid_client_bank_reset, dest_pulse_last, reset), 0, 0, 0)
#define TT_CLEARDVALID(cleardvalid, cleardvalid_S, dest_dvalid_reset, dest_dvalid_client_bank_reset, dest_pulse_last, reset) \
    TT_IP_CLEARDVALID(::ckernel::instrn_buffer, cleardvalid, cleardvalid_S, dest_dvalid_reset, dest_dvalid_client_bank_reset, dest_pulse_last, reset)
#define TTI_CLEARDVALID(cleardvalid, cleardvalid_S, dest_dvalid_reset, dest_dvalid_client_bank_reset, dest_pulse_last, reset) \
    TT_IP_CLEARDVALID(nullptr, cleardvalid, cleardvalid_S, dest_dvalid_reset, dest_dvalid_client_bank_reset, dest_pulse_last, reset)

#define TT_OP_CMPGPR(OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_OP(0x5d, (((OpB_is_Const) << 23) + ((OpSel) << 18) + ((Result_GPR_Index) << 12) + ((OpB_GPR_Index) << 6) + ((OpA_GPR_Index) << 0)))
#define TT_IP_CMPGPR(IP, OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_INSN(IP, TT_OP_CMPGPR(OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index), 0, 0, 0)
#define TT_CMPGPR(OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_CMPGPR(::ckernel::instrn_buffer, OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index)
#define TTI_CMPGPR(OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_CMPGPR(nullptr, OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index)

#define TT_OP_COMMIT_SHADOW(force_commit)     TT_OP(0x41, (((force_commit) << 0)))
#define TT_IP_COMMIT_SHADOW(IP, force_commit) TT_IP_INSN(IP, TT_OP_COMMIT_SHADOW(force_commit), 0, 0, 0)
#define TT_COMMIT_SHADOW(force_commit)        TT_IP_COMMIT_SHADOW(::ckernel::instrn_buffer, force_commit)
#define TTI_COMMIT_SHADOW(force_commit)       TT_IP_COMMIT_SHADOW(nullptr, force_commit)

#define TT_OP_DMANOP     TT_OP(0x60, 0)
#define TT_IP_DMANOP(IP) TT_IP_INSN(IP, TT_OP_DMANOP, 0, 0, 0)
#define TT_DMANOP        TT_IP_DMANOP(::ckernel::instrn_buffer)
#define TTI_DMANOP       TT_IP_DMANOP(nullptr)

#define TT_OP_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_OP(0x28, (((clear_dvalid) << 22) + ((dest_accum_en) << 21) + ((instr_mod19) << 19) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_IP_ELWADD(IP, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_INSN(IP, TT_OP_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst), 0, 0, 0)
#define TT_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_ELWADD(::ckernel::instrn_buffer, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)
#define TTI_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) TT_IP_ELWADD(nullptr, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)

#define TT_OP_ELWADDDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_OP(0x31, (((clear_dvalid) << 22) + ((ins_mod) << 19) + ((srcb_addr) << 15) + ((srca_addr) << 11) + ((addr_mode) << 8) + ((dst) << 0)))
#define TT_IP_ELWADDDI(IP, clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_IP_INSN(IP, TT_OP_ELWADDDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst), 0, 0, 0)
#define TT_ELWADDDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_IP_ELWADDDI(::ckernel::instrn_buffer, clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst)
#define TTI_ELWADDDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_IP_ELWADDDI(nullptr, clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst)

#define TT_OP_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_OP(0x27, (((clear_dvalid) << 22) + ((dest_accum_en) << 21) + ((instr_mod19) << 19) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_IP_ELWMUL(IP, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_INSN(IP, TT_OP_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst), 0, 0, 0)
#define TT_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_ELWMUL(::ckernel::instrn_buffer, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)
#define TTI_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) TT_IP_ELWMUL(nullptr, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)

#define TT_OP_ELWMULDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_OP(0x3a, (((clear_dvalid) << 22) + ((ins_mod) << 19) + ((srcb_addr) << 15) + ((srca_addr) << 11) + ((addr_mode) << 8) + ((dst) << 0)))
#define TT_IP_ELWMULDI(IP, clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_IP_INSN(IP, TT_OP_ELWMULDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst), 0, 0, 0)
#define TT_ELWMULDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_IP_ELWMULDI(::ckernel::instrn_buffer, clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst)
#define TTI_ELWMULDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_IP_ELWMULDI(nullptr, clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst)

#define TT_OP_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_OP(0x30, (((clear_dvalid) << 22) + ((dest_accum_en) << 21) + ((instr_mod19) << 19) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_IP_ELWSUB(IP, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_INSN(IP, TT_OP_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst), 0, 0, 0)
#define TT_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_ELWSUB(::ckernel::instrn_buffer, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)
#define TTI_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) TT_IP_ELWSUB(nullptr, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)

#define TT_OP_ELWSUBDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_OP(0x32, (((clear_dvalid) << 22) + ((ins_mod) << 19) + ((srcb_addr) << 15) + ((srca_addr) << 11) + ((addr_mode) << 8) + ((dst) << 0)))
#define TT_IP_ELWSUBDI(IP, clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_IP_INSN(IP, TT_OP_ELWSUBDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst), 0, 0, 0)
#define TT_ELWSUBDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_IP_ELWSUBDI(::ckernel::instrn_buffer, clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst)
#define TTI_ELWSUBDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_IP_ELWSUBDI(nullptr, clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst)

#define TT_OP_GAPOOL(clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst) \
    TT_OP(0x34, (((clear_dvalid) << 22) + ((instr_mod19) << 19) + ((pool_addr_mode) << 15) + ((rsvd) << 14) + ((dst) << 0)))
#define TT_IP_GAPOOL(IP, clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst) \
    TT_IP_INSN(IP, TT_OP_GAPOOL(clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst), 0, 0, 0)
#define TT_GAPOOL(clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst) \
    TT_IP_GAPOOL(::ckernel::instrn_buffer, clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst)
#define TTI_GAPOOL(clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst) TT_IP_GAPOOL(nullptr, clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst)

#define TT_OP_GMPOOL(clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst) \
    TT_OP(0x33, (((clear_dvalid) << 22) + ((instr_mod19) << 19) + ((pool_addr_mode) << 15) + ((rsvd) << 14) + ((dst) << 0)))
#define TT_IP_GMPOOL(IP, clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst) \
    TT_IP_INSN(IP, TT_OP_GMPOOL(clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst), 0, 0, 0)
#define TT_GMPOOL(clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst) \
    TT_IP_GMPOOL(::ckernel::instrn_buffer, clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst)
#define TTI_GMPOOL(clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst) TT_IP_GMPOOL(nullptr, clear_dvalid, instr_mod19, pool_addr_mode, rsvd, dst)

#define TT_OP_HALT     TT_OP(0x23, 0)
#define TT_IP_HALT(IP) TT_IP_INSN(IP, TT_OP_HALT, 0, 0, 0)
#define TT_HALT        TT_OP_HALT(::ckernel::instrn_buffer)
#define TTI_HALT       TT_OP_HALT(nullptr)

#define TT_OP_INCRWC(rwc_cr, rwc_a, rwc_b, rwc_d)     TT_OP(0x38, (((rwc_cr) << 18) + ((rwc_a) << 13) + ((rwc_b) << 8) + ((rwc_d) << 0)))
#define TT_IP_INCRWC(IP, rwc_cr, rwc_a, rwc_b, rwc_d) TT_IP_INSN(IP, TT_OP_INCRWC(rwc_cr, rwc_a, rwc_b, rwc_d), 0, 0, 0)
#define TT_INCRWC(rwc_cr, rwc_a, rwc_b, rwc_d)        TT_IP_INCRWC(::ckernel::instrn_buffer, rwc_cr, rwc_a, rwc_b, rwc_d)
#define TTI_INCRWC(rwc_cr, rwc_a, rwc_b, rwc_d)       TT_IP_INCRWC(nullptr, rwc_cr, rwc_a, rwc_b, rwc_d)

#define TT_OP_INC_DST_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value) TT_OP(0x0e, (((Tile_Face_Row_Sel) << 21) + ((EngineSel) << 18) + ((Value) << 0)))
#define TT_INC_DST_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value) \
    TT_IP_INC_DST_TILE_FACE_ROW_IDX(::ckernel::instrn_buffer, Tile_Face_Row_Sel, EngineSel, Value)
#define TTI_INC_DST_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value) TT_IP_INC_DST_TILE_FACE_ROW_IDX(nullptr, Tile_Face_Row_Sel, EngineSel, Value)

#define TT_OP_INC_SRC_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value) TT_OP(0x07, (((Tile_Face_Row_Sel) << 21) + ((EngineSel) << 18) + ((Value) << 0)))
#define TT_IP_INC_SRC_TILE_FACE_ROW_IDX(IP, Tile_Face_Row_Sel, EngineSel, Value) \
    TT_IP_INSN(IP, TT_OP_INC_SRC_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value), 0, 0, 0)
#define TT_INC_SRC_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value) \
    TT_IP_INC_SRC_TILE_FACE_ROW_IDX(::ckernel::instrn_buffer, Tile_Face_Row_Sel, EngineSel, Value)
#define TTI_INC_SRC_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value) TT_IP_INC_SRC_TILE_FACE_ROW_IDX(nullptr, Tile_Face_Row_Sel, EngineSel, Value)

#define TT_OP_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index) \
    TT_OP(0x49, (((SizeSel) << 22) + ((OffsetIndex) << 14) + ((AutoIncSpec) << 12) + ((Data_GPR_Index) << 6) + ((Addr_GPR_Index) << 0)))
#define TT_IP_LOADIND(IP, SizeSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index) \
    TT_IP_INSN(IP, TT_OP_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index), 0, 0, 0)
#define TT_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index) \
    TT_IP_LOADIND(::ckernel::instrn_buffer, SizeSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index)
#define TTI_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index) \
    TT_IP_LOADIND(nullptr, SizeSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index)

#define TT_OP_LOADREG(Data_GPR_Index, RegAddr) TT_OP(0x68, (((Data_GPR_Index) << 18) + ((RegAddr) << 0)))
#define TT_IP_LOADREG(Data_GPR_Index, RegAddr) TT_IP_INSN(IP, TT_OP_LOADREG(Data_GPR_Index, RegAddr), 0, 0, 0)
#define TT_LOADREG(Data_GPR_Index, RegAddr)    TT_IP_LOADREG(::ckernel::instrn_buffer, Data_GPR_Index, RegAddr)
#define TTI_LOADREG(Data_GPR_Index, RegAddr)   TT_IP_LOADREG(nullptr, Data_GPR_Index, RegAddr)

#define TT_OP_MOP(mop_type, done, loop_count, zmask_lo8_or_loop_count) \
    TT_OP(0x01, (((mop_type) << 23) + ((done) << 22) + ((loop_count) << 15) + ((zmask_lo8_or_loop_count) << 0)))
#define TT_IP_MOP(IP, mop_type, done, loop_count, zmask_lo8_or_loop_count) \
    TT_IP_INSN(IP, TT_OP_MOP(mop_type, done, loop_count, zmask_lo8_or_loop_count), 0, 0, 0)
#define TT_MOP(mop_type, done, loop_count, zmask_lo8_or_loop_count)  TT_IP_MOP(::ckernel::instrn_buffer, mop_type, done, loop_count, zmask_lo8_or_loop_count)
#define TTI_MOP(mop_type, done, loop_count, zmask_lo8_or_loop_count) TT_IP_MOP(nullptr, mop_type, done, loop_count, zmask_lo8_or_loop_count)

#define TT_OP_MOP_CFG(zmask_hi24)     TT_OP(0x03, (((zmask_hi24) << 0)))
#define TT_IP_MOP_CFG(IP, zmask_hi24) TT_IP_INSN(IP, TT_OP_MOP_CFG(zmask_hi24), 0, 0, 0)
#define TT_MOP_CFG(zmask_hi24)        TT_IP_MOP_CFG(::ckernel::instrn_buffer, zmask_hi24)
#define TTI_MOP_CFG(zmask_hi24)       TT_IP_MOP_CFG(nullptr, zmask_hi24)

#define TT_OP_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst) \
    TT_OP(0x12, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 14) + ((instr_mod) << 12) + ((dst) << 0)))
#define TT_IP_MOVA2D(IP, dest_32b_lo, src, addr_mode, instr_mod, dst) TT_IP_INSN(IP, TT_OP_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst), 0, 0, 0)
#define TT_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst)        TT_IP_MOVA2D(::ckernel::instrn_buffer, dest_32b_lo, src, addr_mode, instr_mod, dst)
#define TTI_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst)       TT_IP_MOVA2D(nullptr, dest_32b_lo, src, addr_mode, instr_mod, dst)

#define TT_OP_MOVB2A(srca, addr_mode, instr_mod, srcb)     TT_OP(0x0b, (((srca) << 17) + ((addr_mode) << 14) + ((instr_mod) << 12) + ((srcb) << 0)))
#define TT_IP_MOVB2A(IP, srca, addr_mode, instr_mod, srcb) TT_IP_INSN(IP, TT_OP_MOVB2A(srca, addr_mode, instr_mod, srcb), 0, 0, 0)
#define TT_MOVB2A(srca, addr_mode, instr_mod, srcb)        TT_IP_MOVB2A(::ckernel::instrn_buffer, srca, addr_mode, instr_mod, srcb)
#define TTI_MOVB2A(srca, addr_mode, instr_mod, srcb)       TT_IP_MOVB2A(nullptr, srca, addr_mode, instr_mod, srcb)

#define TT_OP_MOVB2D(dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst) \
    TT_OP(0x13, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 14) + ((transfer_sz) << 12) + ((bcast_datum0) << 11) + ((dst) << 0)))
#define TT_IP_MOVB2D(IP, dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst) \
    TT_IP_INSN(IP, TT_OP_MOVB2D(dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst), 0, 0, 0)
#define TT_MOVB2D(dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst) \
    TT_IP_MOVB2D(::ckernel::instrn_buffer, dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst)
#define TTI_MOVB2D(dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst) \
    TT_IP_MOVB2D(nullptr, dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst)

#define TT_OP_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst) \
    TT_OP(0x08, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 14) + ((instr_mod) << 12) + ((dst) << 0)))
#define TT_IP_MOVD2A(IP, dest_32b_lo, src, addr_mode, instr_mod, dst) TT_IP_INSN(IP, TT_OP_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst), 0, 0, 0)
#define TT_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst)        TT_IP_MOVD2A(::ckernel::instrn_buffer, dest_32b_lo, src, addr_mode, instr_mod, dst)
#define TTI_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst)       TT_IP_MOVD2A(nullptr, dest_32b_lo, src, addr_mode, instr_mod, dst)

#define TT_OP_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, transpose, dst) \
    TT_OP(0x0a, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 14) + ((instr_mod) << 12) + ((transpose) << 11) + ((dst) << 0)))
#define TT_IP_MOVD2B(IP, dest_32b_lo, src, addr_mode, instr_mod, transpose, dst) \
    TT_IP_INSN(IP, TT_OP_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, transpose, dst), 0, 0, 0)
#define TT_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, transpose, dst) \
    TT_IP_MOVD2B(::ckernel::instrn_buffer, dest_32b_lo, src, addr_mode, instr_mod, transpose, dst)
#define TTI_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, transpose, dst) TT_IP_MOVD2B(nullptr, dest_32b_lo, src, addr_mode, instr_mod, transpose, dst)

#define TT_OP_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst) \
    TT_OP(0x09, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 14) + ((instr_mod) << 12) + ((dst) << 0)))
#define TT_IP_MOVDBGA2D(IP, dest_32b_lo, src, addr_mode, instr_mod, dst) TT_IP_INSN(IP, TT_OP_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst), 0, 0, 0)
#define TT_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst)        TT_IP_MOVDBGA2D(::ckernel::instrn_buffer, dest_32b_lo, src, addr_mode, instr_mod, dst)
#define TTI_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst)       TT_IP_MOVDBGA2D(nullptr, dest_32b_lo, src, addr_mode, instr_mod, dst)

#define TT_OP_MOVDBGB2D(dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst) \
    TT_OP(0x0c, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 14) + ((transfer_sz) << 12) + ((bcast_datum0) << 11) + ((dst) << 0)))
#define TT_IP_MOVDBGB2D(IP, dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst) \
    TT_IP_INSN(IP, TT_OP_MOVDBGB2D(dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst), 0, 0, 0)
#define TT_MOVDBGB2D(dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst) \
    TT_IP_MOVDBGB2D(::ckernel::instrn_buffer, dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst)
#define TTI_MOVDBGB2D(dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst) \
    TT_IP_MOVDBGB2D(nullptr, dest_32b_lo, src, addr_mode, transfer_sz, bcast_datum0, dst)

#define TT_OP_MULGPR(OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_OP(0x5a, (((OpB_is_Const) << 23) + ((Result_GPR_Index) << 12) + ((OpB_GPR_Index) << 6) + ((OpA_GPR_Index) << 0)))
#define TT_IP_MULGPR(IP, OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_INSN(IP, TT_OP_MULGPR(OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index), 0, 0, 0)
#define TT_MULGPR(OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_MULGPR(::ckernel::instrn_buffer, OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index)
#define TTI_MULGPR(OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_MULGPR(nullptr, OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index)

#define TT_OP_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst) \
    TT_OP(0x26, (((clear_dvalid) << 22) + ((instr_mod19) << 19) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_IP_MVMUL(IP, clear_dvalid, instr_mod19, addr_mode, dst) TT_IP_INSN(IP, TT_OP_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst), 0, 0, 0)
#define TT_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst)        TT_IP_MVMUL(::ckernel::instrn_buffer, clear_dvalid, instr_mod19, addr_mode, dst)
#define TTI_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst)       TT_IP_MVMUL(nullptr, clear_dvalid, instr_mod19, addr_mode, dst)

#define TT_OP_MVMULDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_OP(0x25, (((clear_dvalid) << 22) + ((ins_mod) << 19) + ((srcb_addr) << 15) + ((srca_addr) << 11) + ((addr_mode) << 8) + ((dst) << 0)))
#define TT_IP_MVMULDI(IP, clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_IP_INSN(IP, TT_OP_MVMULDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst), 0, 0, 0)
#define TT_MVMULDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_IP_MVMULDI(::ckernel::instrn_buffer, clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst)
#define TTI_MVMULDI(clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst) \
    TT_IP_MVMULDI(nullptr, clear_dvalid, ins_mod, srcb_addr, srca_addr, addr_mode, dst)

#define TT_OP_NOP     TT_OP(0x02, 0)
#define TT_IP_NOP(IP) TT_IP_INSN(IP, TT_OP_NOP, 0, 0, 0)
#define TT_NOP        TT_IP_NOP(::ckernel::instrn_buffer)
#define TTI_NOP       TT_IP_NOP(nullptr)

#define TT_OP_PACR0_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_OP(                                                                                                                                       \
        0x1f,                                                                                                                                    \
        (((Dst_Face_Idx) << 14) + ((Src_Face_Idx) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 9) + ((Src_Tile_Offset_Idx_Inc) << 7) +                 \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IP_PACR0_FACE(IP, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_INSN(                                                                                                                                      \
        IP, TT_OP_PACR0_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid), 0, 0, 0)
#define TT_PACR0_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR0_FACE(                                                                                                                         \
        ::ckernel::instrn_buffer, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid)
#define TTI_PACR0_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR0_FACE(nullptr, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid)

#define TT_OP_PACR0_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_OP(                                                                                                                                                   \
        0x20,                                                                                                                                                \
        (((Dst_Face_Idx_Inc) << 14) + ((Src_Face_Idx_Inc) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 9) + ((Src_Tile_Offset_Idx_Inc) << 7) +                     \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IP_PACR0_FACE_INC(                                                                                                                                 \
    IP, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid)                       \
    TT_IP_INSN(                                                                                                                                               \
        IP,                                                                                                                                                   \
        TT_OP_PACR0_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid), \
        0,                                                                                                                                                    \
        0,                                                                                                                                                    \
        0)
#define TT_PACR0_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR0_FACE_INC(                                                                                                                                 \
        ::ckernel::instrn_buffer,                                                                                                                         \
        Dst_Face_Idx_Inc,                                                                                                                                 \
        Src_Face_Idx_Inc,                                                                                                                                 \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                          \
        Src_Tile_Offset_Idx_Inc,                                                                                                                          \
        Buffer_Descriptor_Table_Sel,                                                                                                                      \
        ClrDatValid)
#define TTI_PACR0_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR0_FACE_INC(                                                                                                                                  \
        nullptr, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid)

#define TT_OP_PACR0_ROW(                                                                                                                              \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_OP(                                                                                                                                            \
        0x2a,                                                                                                                                         \
        (((Dst_Row_Idx) << 20) + ((Src_Row_Idx) << 16) + ((Dst_Face_Idx) << 14) + ((Src_Face_Idx) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 9) +         \
         ((Src_Tile_Offset_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IPPACR0_ROW(                                                                                                                                        \
    IP, Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid)      \
    TT_IP_INSN(                                                                                                                                                \
        IP,                                                                                                                                                    \
        TT_OP_PACR0_ROW(                                                                                                                                       \
            Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid), \
        0,                                                                                                                                                     \
        0,                                                                                                                                                     \
        0)
#define TT_PACR0_ROW(                                                                                                                                 \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR0_ROW(                                                                                                                                  \
        ::ckernel::instrn_buffer,                                                                                                                     \
        Dst_Row_Idx,                                                                                                                                  \
        Src_Row_Idx,                                                                                                                                  \
        Dst_Face_Idx,                                                                                                                                 \
        Src_Face_Idx,                                                                                                                                 \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                      \
        Src_Tile_Offset_Idx_Inc,                                                                                                                      \
        Buffer_Descriptor_Table_Sel,                                                                                                                  \
        ClrDatValid)
#define TTI_PACR0_ROW(                                                                                                                                \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR0_ROW(                                                                                                                                  \
        nullptr,                                                                                                                                      \
        Dst_Row_Idx,                                                                                                                                  \
        Src_Row_Idx,                                                                                                                                  \
        Dst_Face_Idx,                                                                                                                                 \
        Src_Face_Idx,                                                                                                                                 \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                      \
        Src_Tile_Offset_Idx_Inc,                                                                                                                      \
        Buffer_Descriptor_Table_Sel,                                                                                                                  \
        ClrDatValid)

#define TT_OP_PACR0_ROW_INC(                                                                                                                                  \
    Dst_Row_Idx_Inc,                                                                                                                                          \
    Src_Row_Idx_Inc,                                                                                                                                          \
    Dst_Face_Idx_Inc,                                                                                                                                         \
    Src_Face_Idx_Inc,                                                                                                                                         \
    Dst_Tile_Offset_Idx_Inc,                                                                                                                                  \
    Src_Tile_Offset_Idx_Inc,                                                                                                                                  \
    Buffer_Descriptor_Table_Sel,                                                                                                                              \
    ClrDatValid)                                                                                                                                              \
    TT_OP(                                                                                                                                                    \
        0x2b,                                                                                                                                                 \
        (((Dst_Row_Idx_Inc) << 20) + ((Src_Row_Idx_Inc) << 16) + ((Dst_Face_Idx_Inc) << 14) + ((Src_Face_Idx_Inc) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 9) + \
         ((Src_Tile_Offset_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IP_PACR0_ROW_INC(             \
    IP,                                  \
    Dst_Row_Idx_Inc,                     \
    Src_Row_Idx_Inc,                     \
    Dst_Face_Idx_Inc,                    \
    Src_Face_Idx_Inc,                    \
    Dst_Tile_Offset_Idx_Inc,             \
    Src_Tile_Offset_Idx_Inc,             \
    Buffer_Descriptor_Table_Sel,         \
    ClrDatValid)                         \
    TT_IP_INSN(                          \
        IP,                              \
        TT_OP_PACR0_ROW_INC(             \
            Dst_Row_Idx_Inc,             \
            Src_Row_Idx_Inc,             \
            Dst_Face_Idx_Inc,            \
            Src_Face_Idx_Inc,            \
            Dst_Tile_Offset_Idx_Inc,     \
            Src_Tile_Offset_Idx_Inc,     \
            Buffer_Descriptor_Table_Sel, \
            ClrDatValid),                \
        0,                               \
        0,                               \
        0)
#define TT_PACR0_ROW_INC(            \
    Dst_Row_Idx_Inc,                 \
    Src_Row_Idx_Inc,                 \
    Dst_Face_Idx_Inc,                \
    Src_Face_Idx_Inc,                \
    Dst_Tile_Offset_Idx_Inc,         \
    Src_Tile_Offset_Idx_Inc,         \
    Buffer_Descriptor_Table_Sel,     \
    ClrDatValid)                     \
    TT_IP_PACR0_ROW_INC(             \
        ::ckernel::instrn_buffer,    \
        Dst_Row_Idx_Inc,             \
        Src_Row_Idx_Inc,             \
        Dst_Face_Idx_Inc,            \
        Src_Face_Idx_Inc,            \
        Dst_Tile_Offset_Idx_Inc,     \
        Src_Tile_Offset_Idx_Inc,     \
        Buffer_Descriptor_Table_Sel, \
        ClrDatValid)
#define TTI_PACR0_ROW_INC(           \
    Dst_Row_Idx_Inc,                 \
    Src_Row_Idx_Inc,                 \
    Dst_Face_Idx_Inc,                \
    Src_Face_Idx_Inc,                \
    Dst_Tile_Offset_Idx_Inc,         \
    Src_Tile_Offset_Idx_Inc,         \
    Buffer_Descriptor_Table_Sel,     \
    ClrDatValid)                     \
    TT_IP_PACR0_ROW_INC(             \
        nullptr,                     \
        Dst_Row_Idx_Inc,             \
        Src_Row_Idx_Inc,             \
        Dst_Face_Idx_Inc,            \
        Src_Face_Idx_Inc,            \
        Dst_Tile_Offset_Idx_Inc,     \
        Src_Tile_Offset_Idx_Inc,     \
        Buffer_Descriptor_Table_Sel, \
        ClrDatValid)

#define TT_OP_PACR0_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_OP(0x0f, (((Dst_Tile_Idx) << 16) + ((Src_Tile_Idx) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IP_PACR0_TILE(IP, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_INSN(IP, TT_OP_PACR0_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid), 0, 0, 0)
#define TT_PACR0_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR0_TILE(::ckernel::instrn_buffer, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid)
#define TTI_PACR0_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR0_TILE(nullptr, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid)

#define TT_OP_PACR0_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_OP(0x19, (((Dst_Tile_Idx_Inc) << 16) + ((Src_Tile_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IP_PACR0_TILE_INC(IP, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_INSN(IP, TT_OP_PACR0_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid), 0, 0, 0)
#define TT_PACR0_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR0_TILE_INC(::ckernel::instrn_buffer, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid))
#define TTI_PACR0_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR0_TILE_INC(nullptr, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid))

#define TT_OP_PACR1_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_OP(                                                                                                                                       \
        0x2e,                                                                                                                                    \
        (((Dst_Face_Idx) << 14) + ((Src_Face_Idx) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 9) + ((Src_Tile_Offset_Idx_Inc) << 7) +                 \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IP_PACR1_FACE(IP, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_INSN(                                                                                                                                      \
        IP, TT_OP_PACR1_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid), 0, 0, 0)
#define TT_PACR1_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR1_FACE(                                                                                                                         \
        ::ckernel::instrn_buffer, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid)
#define TTI_PACR1_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR1_FACE(nullptr, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid)

#define TT_OP_PACR1_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_OP(                                                                                                                                                   \
        0x2f,                                                                                                                                                \
        (((Dst_Face_Idx_Inc) << 14) + ((Src_Face_Idx_Inc) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 9) + ((Src_Tile_Offset_Idx_Inc) << 7) +                     \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IP_PACR1_FACE_INC(                                                                                                                                 \
    IP, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid)                       \
    TT_IP_INSN(                                                                                                                                               \
        IP,                                                                                                                                                   \
        TT_OP_PACR1_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid), \
        0,                                                                                                                                                    \
        0,                                                                                                                                                    \
        0)
#define TT_PACR1_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR1_FACE_INC(::ckernel::instrn_buffer, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid))
#define TTI_PACR1_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR1_FACE_INC(nullptr, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid))

#define TT_OP_PACR1_ROW(                                                                                                                              \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_OP(                                                                                                                                            \
        0x3b,                                                                                                                                         \
        (((Dst_Row_Idx) << 20) + ((Src_Row_Idx) << 16) + ((Dst_Face_Idx) << 14) + ((Src_Face_Idx) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 9) +         \
         ((Src_Tile_Offset_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IP_PACR1_ROW(                                                                                                                                       \
    IP, Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid)      \
    TT_IP_INSN(                                                                                                                                                \
        IP,                                                                                                                                                    \
        TT_OP_PACR1_ROW(                                                                                                                                       \
            Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid), \
        0,                                                                                                                                                     \
        0,                                                                                                                                                     \
        0)
#define TT_PACR1_ROW(                                                                                                                                 \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR1_ROW(::ckernel::instrn_buffer,                                                                                                                           \
        Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid))
#define TTI_PACR1_ROW(                                                                                                                                \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR1_ROW(nullptr,                                                                                                                          \
        Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid))

#define TT_OP_PACR1_ROW_INC(                                                                                                                                  \
    Dst_Row_Idx_Inc,                                                                                                                                          \
    Src_Row_Idx_Inc,                                                                                                                                          \
    Dst_Face_Idx_Inc,                                                                                                                                         \
    Src_Face_Idx_Inc,                                                                                                                                         \
    Dst_Tile_Offset_Idx_Inc,                                                                                                                                  \
    Src_Tile_Offset_Idx_Inc,                                                                                                                                  \
    Buffer_Descriptor_Table_Sel,                                                                                                                              \
    ClrDatValid)                                                                                                                                              \
    TT_OP(                                                                                                                                                    \
        0x6e,                                                                                                                                                 \
        (((Dst_Row_Idx_Inc) << 20) + ((Src_Row_Idx_Inc) << 16) + ((Dst_Face_Idx_Inc) << 14) + ((Src_Face_Idx_Inc) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 9) + \
         ((Src_Tile_Offset_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IP_PACR1_ROW_INC(             \
    IP,                                  \
    Dst_Row_Idx_Inc,                     \
    Src_Row_Idx_Inc,                     \
    Dst_Face_Idx_Inc,                    \
    Src_Face_Idx_Inc,                    \
    Dst_Tile_Offset_Idx_Inc,             \
    Src_Tile_Offset_Idx_Inc,             \
    Buffer_Descriptor_Table_Sel,         \
    ClrDatValid)                         \
    TT_IP_INSN(                          \
        IP,                              \
        TT_OP_PACR1_ROW_INC(             \
            Dst_Row_Idx_Inc,             \
            Src_Row_Idx_Inc,             \
            Dst_Face_Idx_Inc,            \
            Src_Face_Idx_Inc,            \
            Dst_Tile_Offset_Idx_Inc,     \
            Src_Tile_Offset_Idx_Inc,     \
            Buffer_Descriptor_Table_Sel, \
            ClrDatValid),                \
        0,                               \
        0,                               \
        0)
#define TT_PACR1_ROW_INC(            \
    Dst_Row_Idx_Inc,                 \
    Src_Row_Idx_Inc,                 \
    Dst_Face_Idx_Inc,                \
    Src_Face_Idx_Inc,                \
    Dst_Tile_Offset_Idx_Inc,         \
    Src_Tile_Offset_Idx_Inc,         \
    Buffer_Descriptor_Table_Sel,     \
    ClrDatValid)                     \
    TT_IP_PACR1_ROW_INC(             \
        ::ckernel::instrn_buffer,    \
        Dst_Row_Idx_Inc,             \
        Src_Row_Idx_Inc,             \
        Dst_Face_Idx_Inc,            \
        Src_Face_Idx_Inc,            \
        Dst_Tile_Offset_Idx_Inc,     \
        Src_Tile_Offset_Idx_Inc,     \
        Buffer_Descriptor_Table_Sel, \
        ClrDatValid)
#define TTI_PACR1_ROW_INC(           \
    Dst_Row_Idx_Inc,                 \
    Src_Row_Idx_Inc,                 \
    Dst_Face_Idx_Inc,                \
    Src_Face_Idx_Inc,                \
    Dst_Tile_Offset_Idx_Inc,         \
    Src_Tile_Offset_Idx_Inc,         \
    Buffer_Descriptor_Table_Sel,     \
    ClrDatValid)                     \
    TT_IP_PACR1_ROW_INC(             \
        nullptr,                     \
        Dst_Row_Idx_Inc,             \
        Src_Row_Idx_Inc,             \
        Dst_Face_Idx_Inc,            \
        Src_Face_Idx_Inc,            \
        Dst_Tile_Offset_Idx_Inc,     \
        Src_Tile_Offset_Idx_Inc,     \
        Buffer_Descriptor_Table_Sel, \
        ClrDatValid)

#define TT_OP_PACR1_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_OP(0x2c, (((Dst_Tile_Idx) << 16) + ((Src_Tile_Idx) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IP_PACR1_TILE(IP, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_INSN(IP, TT_OP_PACR1_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid), 0, 0, 0)
#define TT_PACR1_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR1_TILE(::ckernel::instrn_buffer, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid))
#define TTI_PACR1_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR1_TILE(nullptr, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, ClrDatValid))

#define TT_OP_PACR1_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_OP(0x2d, (((Dst_Tile_Idx_Inc) << 16) + ((Src_Tile_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IP_PACR1_TILE_INC(IP, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_INSN(IP, TT_OP_PACR1_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid), 0, 0, 0)
#define TT_PACR1_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR1_TILE_INC(::ckernel::instrn_buffer, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid)
#define TTI_PACR1_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR1_TILE_INC(nullptr, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, ClrDatValid)

#define TT_OP_PACR_STRIDE(                                                                                                                \
    Src_Row_Idx_or_Inc_Mul4,                                                                                                              \
    Src_Row_Idx_Inc,                                                                                                                      \
    L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                          \
    Tile_Idx_Inc,                                                                                                                         \
    L1_16datums_Row_Index,                                                                                                                \
    Buffer_Descriptor_Table_Sel,                                                                                                          \
    PackerSel,                                                                                                                            \
    ClrDatValid)                                                                                                                          \
    TT_OP(                                                                                                                                \
        0x1d,                                                                                                                             \
        (((Src_Row_Idx_or_Inc_Mul4) << 18) + ((Src_Row_Idx_Inc) << 17) + ((L1_Tile_Idx_or_Tile_Idx_Inc) << 14) + ((Tile_Idx_Inc) << 13) + \
         ((L1_16datums_Row_Index) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((PackerSel) << 1) + ((ClrDatValid) << 0)))
#define TT_IP_PACR_STRIDE(               \
    IP,                                  \
    Src_Row_Idx_or_Inc_Mul4,             \
    Src_Row_Idx_Inc,                     \
    L1_Tile_Idx_or_Tile_Idx_Inc,         \
    Tile_Idx_Inc,                        \
    L1_16datums_Row_Index,               \
    Buffer_Descriptor_Table_Sel,         \
    PackerSel,                           \
    ClrDatValid)                         \
    TT_IP_INSN(                          \
        IP,                              \
        TT_OP_PACR_STRIDE(               \
            Src_Row_Idx_or_Inc_Mul4,     \
            Src_Row_Idx_Inc,             \
            L1_Tile_Idx_or_Tile_Idx_Inc, \
            Tile_Idx_Inc,                \
            L1_16datums_Row_Index,       \
            Buffer_Descriptor_Table_Sel, \
            PackerSel,                   \
            ClrDatValid),                \
        0,                               \
        0,                               \
        0)
#define TT_PACR_STRIDE(              \
    Src_Row_Idx_or_Inc_Mul4,         \
    Src_Row_Idx_Inc,                 \
    L1_Tile_Idx_or_Tile_Idx_Inc,     \
    Tile_Idx_Inc,                    \
    L1_16datums_Row_Index,           \
    Buffer_Descriptor_Table_Sel,     \
    PackerSel,                       \
    ClrDatValid)                     \
    TT_IP_PACR_STRIDE(               \
        ::ckernel::instrn_buffer,    \
        Src_Row_Idx_or_Inc_Mul4,     \
        Src_Row_Idx_Inc,             \
        L1_Tile_Idx_or_Tile_Idx_Inc, \
        Tile_Idx_Inc,                \
        L1_16datums_Row_Index,       \
        Buffer_Descriptor_Table_Sel, \
        PackerSel,                   \
        ClrDatValid)
#define TTI_PACR_STRIDE(             \
    Src_Row_Idx_or_Inc_Mul4,         \
    Src_Row_Idx_Inc,                 \
    L1_Tile_Idx_or_Tile_Idx_Inc,     \
    Tile_Idx_Inc,                    \
    L1_16datums_Row_Index,           \
    Buffer_Descriptor_Table_Sel,     \
    PackerSel,                       \
    ClrDatValid)                     \
    TT_IP_PACR_STRIDE(               \
        nullptr,                     \
        Src_Row_Idx_or_Inc_Mul4,     \
        Src_Row_Idx_Inc,             \
        L1_Tile_Idx_or_Tile_Idx_Inc, \
        Tile_Idx_Inc,                \
        L1_16datums_Row_Index,       \
        Buffer_Descriptor_Table_Sel, \
        PackerSel,                   \
        ClrDatValid)

#define TT_OP_PACR_UNTILIZE(Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Packer_Sel, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_OP(                                                                                                                                   \
        0x42,                                                                                                                                \
        (((Reserved) << 14) + ((Cntr_Reset_mask) << 12) + ((Dst_Z_Cntr_inc) << 10) + ((Src_Z_Cntr_inc) << 8) + ((Packer_Sel) << 7) +         \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((ClrDatValid) << 1)))
#define TT_IP_PACR_UNTILIZE(IP, Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Packer_Sel, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_INSN(                                                                                                                                  \
        IP, TT_OP_PACR_UNTILIZE(Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Packer_Sel, Buffer_Descriptor_Table_Sel, ClrDatValid), 0, 0, 0)
#define TT_PACR_UNTILIZE(Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Packer_Sel, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR_UNTILIZE(                                                                                                                  \
        ::ckernel::instrn_buffer, Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Packer_Sel, Buffer_Descriptor_Table_Sel, ClrDatValid)
#define TTI_PACR_UNTILIZE(Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Packer_Sel, Buffer_Descriptor_Table_Sel, ClrDatValid) \
    TT_IP_PACR_UNTILIZE(nullptr, Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Packer_Sel, Buffer_Descriptor_Table_Sel, ClrDatValid)

#define TT_OP_POP_TILES(unpacker_rd_done_wait_mask, num_tiles, buffer_sel) \
    TT_OP(0x3e, (((unpacker_rd_done_wait_mask) << 15) + ((num_tiles) << 5) + ((buffer_sel) << 0)))
#define TT_IP_POP_TILES(IP, unpacker_rd_done_wait_mask, num_tiles, buffer_sel) \
    TT_IP_INSN(IP, TT_OP_POP_TILES(unpacker_rd_done_wait_mask, num_tiles, buffer_sel), 0, 0, 0)
#define TT_POP_TILES(unpacker_rd_done_wait_mask, num_tiles, buffer_sel) \
    TT_IP_POP_TILES(::ckernel::instrn_buffer, unpacker_rd_done_wait_mask, num_tiles, buffer_sel)
#define TTI_POP_TILES(unpacker_rd_done_wait_mask, num_tiles, buffer_sel) TT_IP_POP_TILES(nullptr, unpacker_rd_done_wait_mask, num_tiles, buffer_sel)

#define TT_OP_PUSH_TILES(packer_wr_done_wait_mask, num_tiles, buffer_sel) \
    TT_OP(0x3d, (((packer_wr_done_wait_mask) << 15) + ((num_tiles) << 5) + ((buffer_sel) << 0)))
#define TT_IP_PUSH_TILES(IP, packer_wr_done_wait_mask, num_tiles, buffer_sel) \
    TT_IP_INSN(IP, TT_OP_PUSH_TILES(packer_wr_done_wait_mask, num_tiles, buffer_sel), 0, 0, 0)
#define TT_PUSH_TILES(packer_wr_done_wait_mask, num_tiles, buffer_sel) \
    TT_IP_PUSH_TILES(::ckernel::instrn_buffer, packer_wr_done_wait_mask, num_tiles, buffer_sel)
#define TTI_PUSH_TILES(packer_wr_done_wait_mask, num_tiles, buffer_sel) TT_IP_PUSH_TILES(nullptr, packer_wr_done_wait_mask, num_tiles, buffer_sel)

#define TT_OP_RDCFG(GprAddress, CfgReg)     TT_OP(0xb1, (((GprAddress) << 16) + ((CfgReg) << 0)))
#define TT_IP_RDCFG(IP, GprAddress, CfgReg) TT_IP_INSN(IP, TT_OP_RDCFG(GprAddress, CfgReg), 0, 0, 0)
#define TT_RDCFG(GprAddress, CfgReg)        TT_IP_RDCFG(::ckernel::instrn_buffer, GprAddress, CfgReg)
#define TTI_RDCFG(GprAddress, CfgReg)       TT_IP_RDCFG(nullptr, GprAddress, CfgReg)

#define TT_OP_REPLAY(start_idx, len, last, set_mutex, execute_while_loading, load_mode) \
    TT_OP(0x04, (((start_idx) << 14) + ((len) << 4) + ((last) << 3) + ((set_mutex) << 2) + ((execute_while_loading) << 1) + ((load_mode) << 0)))
#define TT_IP_REPLAY(IP, start_idx, len, last, set_mutex, execute_while_loading, load_mode) \
    TT_IP_INSN(IP, TT_OP_REPLAY(start_idx, len, last, set_mutex, execute_while_loading, load_mode), 0, 0, 0)
#define TT_REPLAY(start_idx, len, last, set_mutex, execute_while_loading, load_mode) \
    TT_IP_REPLAY(::ckernel::instrn_buffer, start_idx, len, last, set_mutex, execute_while_loading, load_mode)
#define TTI_REPLAY(start_idx, len, last, set_mutex, execute_while_loading, load_mode) \
    TT_IP_REPLAY(nullptr, start_idx, len, last, set_mutex, execute_while_loading, load_mode)

#define TT_OP_RESOURCEDECL(linger_time, resources, op_class)     TT_OP(0x05, (((linger_time) << 15) + ((resources) << 5) + ((op_class) << 0)))
#define TT_IP_RESOURCEDECL(IP, linger_time, resources, op_class) TT_IP_INSN(IP, TT_OP_RESOURCEDECL(linger_time, resources, op_class), 0, 0, 0)
#define TT_RESOURCEDECL(linger_time, resources, op_class)        TT_IP_RESOURCEDECL(::ckernel::instrn_buffer, linger_time, resources, op_class)
#define TTI_RESOURCEDECL(linger_time, resources, op_class)       TT_IP_RESOURCEDECL(nullptr, linger_time, resources, op_class)

#define TT_OP_RMWCIB0(CfgRegAddr, Mask, Data)     TT_OP(0xb2, (((CfgRegAddr) << 16) + ((Mask) << 8) + ((Data) << 0)))
#define TT_IP_RMWCIB0(IP, CfgRegAddr, Mask, Data) TT_IP_INSN(IP, TT_OP_RMWCIB0(CfgRegAddr, Mask, Data), 0, 0, 0)
#define TT_RMWCIB0(CfgRegAddr, Mask, Data)        TT_IP_RMWCIB0(::ckernel::instrn_buffer, CfgRegAddr, Mask, Data)
#define TTI_RMWCIB0(CfgRegAddr, Mask, Data)       TT_IP_RMWCIB0(nullptr, CfgRegAddr, Mask, Data)

#define TT_OP_RMWCIB0_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data) \
    TT_OP(0xb3, (((CfgRegAddr) << 16) + ((Mask) << 8) + ((Data) << 0)))
#define TT_IP_RMWCIB0_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(IP, CfgRegAddr, Mask, Data) \
    TT_IP_INSN(IP, TT_OP_RMWCIB0_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data), 0, 0, 0)
#define TT_RMWCIB0_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data) \
    TT_IP_RMWCIB0_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(::ckernel::instrn_buffer, CfgRegAddr, Mask, Data)
#define TTI_RMWCIB0_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data) \
    TT_IP_RMWCIB0_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(nullptr, CfgRegAddr, Mask, Data)

#define TT_OP_RMWCIB1(CfgRegAddr, Mask, Data)     TT_OP(0xb4, (((CfgRegAddr) << 16) + ((Mask) << 8) + ((Data) << 0)))
#define TT_IP_RMWCIB1(IP, CfgRegAddr, Mask, Data) TT_IP_INSN(IP, TT_OP_RMWCIB1(CfgRegAddr, Mask, Data), 0, 0, 0)
#define TT_RMWCIB1(CfgRegAddr, Mask, Data)        TT_IP_RMWCIB1(::ckernel::instrn_buffer, CfgRegAddr, Mask, Data)
#define TTI_RMWCIB1(CfgRegAddr, Mask, Data)       TT_IP_RMWCIB1(nullptr, CfgRegAddr, Mask, Data)

#define TT_OP_RMWCIB1_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data) \
    TT_OP(0xb5, (((CfgRegAddr) << 16) + ((Mask) << 8) + ((Data) << 0)))
#define TT_IP_RMWCIB1_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(IP, CfgRegAddr, Mask, Data) \
    TT_IP_INSN(IP, TT_OP_RMWCIB1_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data), 0, 0, 0)
#define TT_RMWCIB1_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data) \
    TT_IP_RMWCIB1_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(::ckernel::instrn_buffer, CfgRegAddr, Mask, Data)
#define TTI_RMWCIB1_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data) \
    TT_IP_RMWCIB1_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(nullptr, CfgRegAddr, Mask, Data)

#define TT_OP_RMWCIB2(CfgRegAddr, Mask, Data)     TT_OP(0xb6, (((CfgRegAddr) << 16) + ((Mask) << 8) + ((Data) << 0)))
#define TT_IP_RMWCIB2(IP, CfgRegAddr, Mask, Data) TT_IP_INSN(IP, TT_OP_RMWCIB2(CfgRegAddr, Mask, Data), 0, 0, 0)
#define TT_RMWCIB2(CfgRegAddr, Mask, Data)        TT_IP_RMWCIB2(::ckernel::instrn_buffer, CfgRegAddr, Mask, Data)
#define TTI_RMWCIB2(CfgRegAddr, Mask, Data)       TT_IP_RMWCIB2(nullptr, CfgRegAddr, Mask, Data)

#define TT_OP_RMWCIB2_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data) \
    TT_OP(0xb7, (((CfgRegAddr) << 16) + ((Mask) << 8) + ((Data) << 0)))
#define TT_IP_RMWCIB2_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(IP, CfgRegAddr, Mask, Data) \
    TT_IP_INSN(IP, TT_OP_RMWCIB2_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data), 0, 0, 0)
#define TT_RMWCIB2_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data) \
    TT_IP_RMWCIB2_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(::ckernel::instrn_buffer, CfgRegAddr, Mask, Data)
#define TTI_RMWCIB2_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data) \
    TT_IP_RMWCIB2_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(nullptr, CfgRegAddr, Mask, Data)

#define TT_OP_RMWCIB3(CfgRegAddr, Mask, Data)     TT_OP(0xb8, (((CfgRegAddr) << 16) + ((Mask) << 8) + ((Data) << 0)))
#define TT_IP_RMWCIB3(IP, CfgRegAddr, Mask, Data) TT_IP_INSN(IP, TT_OP_RMWCIB3(CfgRegAddr, Mask, Data), 0, 0, 0)
#define TT_RMWCIB3(CfgRegAddr, Mask, Data)        TT_IP_RMWCIB3(::ckernel::instrn_buffer, CfgRegAddr, Mask, Data)
#define TTI_RMWCIB3(CfgRegAddr, Mask, Data)       TT_IP_RMWCIB3(nullptr, CfgRegAddr, Mask, Data)

#define TT_OP_RMWCIB3_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data) \
    TT_OP(0xb9, (((CfgRegAddr) << 16) + ((Mask) << 8) + ((Data) << 0)))
#define TT_IP_RMWCIB3_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(IP, CfgRegAddr, Mask, Data) \
    TT_IP_INSN(IP, TT_OP_RMWCIB3_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data), 0, 0, 0)
#define TT_RMWCIB3_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data) \
    TT_IP_RMWCIB3_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(::ckernel::instrn_buffer, CfgRegAddr, Mask, Data)
#define TTI_RMWCIB3_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(CfgRegAddr, Mask, Data) \
    TT_IP_RMWCIB3_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE(nullptr, CfgRegAddr, Mask, Data)

#define TT_OP_RV_PACR(reg_idx2, reg_idx1, reg_idx0)     TT_OP(0x35, (((reg_idx2) << 10) + ((reg_idx1) << 5) + ((reg_idx0) << 0)))
#define TT_IP_RV_PACR(IP, reg_idx2, reg_idx1, reg_idx0) TT_IP_INSN(IP, TT_OP_RV_PACR(reg_idx2, reg_idx1, reg_idx0), 0, 0, 0)
#define TT_RV_PACR(reg_idx2, reg_idx1, reg_idx0)        TT_IP_RV_PACR(::ckernel::instrn_buffer, reg_idx2, reg_idx1, reg_idx0)
#define TTI_RV_PACR(reg_idx2, reg_idx1, reg_idx0)       TT_IP_RV_PACR(nullptr, reg_idx2, reg_idx1, reg_idx0)

#define TT_OP_RV_UNPACR(reg_idx2, reg_idx1, reg_idx0)     TT_OP(0x39, (((reg_idx2) << 10) + ((reg_idx1) << 5) + ((reg_idx0) << 0)))
#define TT_IP_RV_UNPACR(IP, reg_idx2, reg_idx1, reg_idx0) TT_IP_INSN(IP, TT_OP_RV_UNPACR(reg_idx2, reg_idx1, reg_idx0), 0, 0, 0)
#define TT_RV_UNPACR(reg_idx2, reg_idx1, reg_idx0)        TT_IP_RV_UNPACR(::ckernel::instrn_buffer, reg_idx2, reg_idx1, reg_idx0)
#define TTI_RV_UNPACR(reg_idx2, reg_idx1, reg_idx0)       TT_IP_RV_UNPACR(nullptr, reg_idx2, reg_idx1, reg_idx0)

#define TT_OP_RV_WRCFG(byte_mask, write_64b, index_of_reg_containing_cfg_index, index_of_reg_containing_wrdata_msbs, index_of_reg_containing_wrdata_lsbs) \
    TT_OP(                                                                                                                                                \
        0x54,                                                                                                                                             \
        (((byte_mask) << 16) + ((write_64b) << 15) + ((index_of_reg_containing_cfg_index) << 10) + ((index_of_reg_containing_wrdata_msbs) << 5) +         \
         ((index_of_reg_containing_wrdata_lsbs) << 0)))
#define TT_IP_RV_WRCFG(IP, byte_mask, write_64b, index_of_reg_containing_cfg_index, index_of_reg_containing_wrdata_msbs, index_of_reg_containing_wrdata_lsbs) \
    TT_IP_INSN(                                                                                                                                               \
        IP,                                                                                                                                                   \
        TT_OP_RV_WRCFG(byte_mask, write_64b, index_of_reg_containing_cfg_index, index_of_reg_containing_wrdata_msbs, index_of_reg_containing_wrdata_lsbs),    \
        0,                                                                                                                                                    \
        0,                                                                                                                                                    \
        0)
#define TT_RV_WRCFG(byte_mask, write_64b, index_of_reg_containing_cfg_index, index_of_reg_containing_wrdata_msbs, index_of_reg_containing_wrdata_lsbs) \
    TT_IP_RV_WRCFG(                                                                                                                                    \
        ::ckernel::instrn_buffer,                                                                                                                      \
        byte_mask,                                                                                                                                     \
        write_64b,                                                                                                                                     \
        index_of_reg_containing_cfg_index,                                                                                                             \
        index_of_reg_containing_wrdata_msbs,                                                                                                           \
        index_of_reg_containing_wrdata_lsbs)
#define TTI_RV_WRCFG(byte_mask, write_64b, index_of_reg_containing_cfg_index, index_of_reg_containing_wrdata_msbs, index_of_reg_containing_wrdata_lsbs) \
    TT_IP_RV_WRCFG(nullptr, byte_mask, write_64b, index_of_reg_containing_cfg_index, index_of_reg_containing_wrdata_msbs, index_of_reg_containing_wrdata_lsbs)

#define TT_OP_SEMGET(sem_bank_sel, sem_sel)     TT_OP(0xa5, (((sem_bank_sel) << 8) + ((sem_sel) << 0)))
#define TT_IP_SEMGET(IP, sem_bank_sel, sem_sel) TT_IP_INSN(IP, TT_OP_SEMGET(sem_bank_sel, sem_sel), 0, 0, 0)
#define TT_SEMGET(sem_bank_sel, sem_sel)        TT_IP_SEMGET(::ckernel::instrn_buffer, sem_bank_sel, sem_sel)
#define TTI_SEMGET(sem_bank_sel, sem_sel)       TT_IP_SEMGET(nullptr, sem_bank_sel, sem_sel)

#define TT_OP_SEMINIT(max_value, init_value, sem_bank_sel, sem_sel) \
    TT_OP(0xa3, (((max_value) << 20) + ((init_value) << 16) + ((sem_bank_sel) << 8) + ((sem_sel) << 0)))
#define TT_IP_SEMINIT(IP, max_value, init_value, sem_bank_sel, sem_sel) TT_IP_INSN(IP, TT_OP_SEMINIT(max_value, init_value, sem_bank_sel, sem_sel), 0, 0, 0)
#define TT_SEMINIT(max_value, init_value, sem_bank_sel, sem_sel)        TT_IP_SEMINIT(::ckernel::instrn_buffer, max_value, init_value, sem_bank_sel, sem_sel)
#define TTI_SEMINIT(max_value, init_value, sem_bank_sel, sem_sel)       TT_IP_SEMINIT(nullptr, max_value, init_value, sem_bank_sel, sem_sel)

#define TT_OP_SEMPOST(sem_bank_sel, sem_sel)     TT_OP(0xa4, (((sem_bank_sel) << 8) + ((sem_sel) << 0)))
#define TT_IP_SEMPOST(IP, sem_bank_sel, sem_sel) TT_IP_INSN(IP, TT_OP_SEMPOST(sem_bank_sel, sem_sel), 0, 0, 0)
#define TT_SEMPOST(sem_bank_sel, sem_sel)        TT_IP_SEMPOST(::ckernel::instrn_buffer, sem_bank_sel, sem_sel)
#define TTI_SEMPOST(sem_bank_sel, sem_sel)       TT_IP_SEMPOST(nullptr, sem_bank_sel, sem_sel)

#define TT_OP_SEMWAIT(stall_res, wait_sem_cond, sem_bank_sel, sem_sel) \
    TT_OP(0xa6, (((stall_res) << 15) + ((wait_sem_cond) << 13) + ((sem_bank_sel) << 8) + ((sem_sel) << 0)))
#define TT_IP_SEMWAIT(IP, stall_res, wait_sem_cond, sem_bank_sel, sem_sel) \
    TT_IP_INSN(IP, TT_OP_SEMWAIT(stall_res, wait_sem_cond, sem_bank_sel, sem_sel), 0, 0, 0)
#define TT_SEMWAIT(stall_res, wait_sem_cond, sem_bank_sel, sem_sel)  TT_IP_SEMWAIT(::ckernel::instrn_buffer, stall_res, wait_sem_cond, sem_bank_sel, sem_sel)
#define TTI_SEMWAIT(stall_res, wait_sem_cond, sem_bank_sel, sem_sel) TT_IP_SEMWAIT(nullptr, stall_res, wait_sem_cond, sem_bank_sel, sem_sel)

#define TT_OP_SETDVALID(setvalid) TT_OP(0x57, (((setvalid) << 0)))
#define TT_IP_SETDVALID(setvalid) TT_IP_INSN(IP, TT_OP_SETDVALID(setvalid), 0, 0, 0)
#define TT_SETDVALID(setvalid)    TT_IP_SETDVALID(::ckernel::instrn_buffer, setvalid)
#define TTI_SETDVALID(setvalid)   TT_IP_SETDVALID(nullptr, setvalid)

#define TT_OP_SETGPR(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, GPR_Index16b) \
    TT_OP(0x45, (((Payload_SigSelSize) << 22) + ((Payload_SigSel) << 8) + ((SetSignalsMode) << 7) + ((GPR_Index16b) << 0)))
#define TT_IP_SETGPR(IP, Payload_SigSelSize, Payload_SigSel, SetSignalsMode, GPR_Index16b) \
    TT_IP_INSN(IP, TT_OP_SETGPR(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, GPR_Index16b), 0, 0, 0)
#define TT_SETGPR(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, GPR_Index16b) \
    TT_IP_SETGPR(::ckernel::instrn_buffer, Payload_SigSelSize, Payload_SigSel, SetSignalsMode, GPR_Index16b)
#define TTI_SETGPR(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, GPR_Index16b) \
    TT_IP_SETGPR(nullptr, Payload_SigSelSize, Payload_SigSel, SetSignalsMode, GPR_Index16b)

#define TT_OP_SETRWC(clear_ab_vld, rwc_cr, rwc_val, BitMask)     TT_OP(0x37, (((clear_ab_vld) << 22) + ((rwc_cr) << 18) + ((rwc_val) << 6) + ((BitMask) << 0)))
#define TT_IP_SETRWC(IP, clear_ab_vld, rwc_cr, rwc_val, BitMask) TT_IP_INSN(IP, TT_OP_SETRWC(clear_ab_vld, rwc_cr, rwc_val, BitMask), 0, 0, 0)
#define TT_SETRWC(clear_ab_vld, rwc_cr, rwc_val, BitMask)        TT_IP_SETRWC(::ckernel::instrn_buffer, clear_ab_vld, rwc_cr, rwc_val, BitMask)
#define TTI_SETRWC(clear_ab_vld, rwc_cr, rwc_val, BitMask)       TT_IP_SETRWC(nullptr, clear_ab_vld, rwc_cr, rwc_val, BitMask)

#define TT_OP_SET_DST_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value) TT_OP(0x0d, (((Tile_Face_Row_Sel) << 21) + ((EngineSel) << 18) + ((Value) << 0)))
#define TT_IP_SET_DST_TILE_FACE_ROW_IDX(IP, Tile_Face_Row_Sel, EngineSel, Value) \
    TT_IP_INSN(IP, TT_OP_SET_DST_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value), 0, 0, 0)
#define TT_SET_DST_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value) \
    TT_IP_SET_DST_TILE_FACE_ROW_IDX(::ckernel::instrn_buffer, Tile_Face_Row_Sel, EngineSel, Value)
#define TTI_SET_DST_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value) TT_IP_SET_DST_TILE_FACE_ROW_IDX(nullptr, Tile_Face_Row_Sel, EngineSel, Value)

#define TT_OP_SET_SRC_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value) TT_OP(0x06, (((Tile_Face_Row_Sel) << 21) + ((EngineSel) << 18) + ((Value) << 0)))
#define TT_IP_SET_SRC_TILE_FACE_ROW_IDX(IP, Tile_Face_Row_Sel, EngineSel, Value) \
    TT_IP_INSN(IP, TT_OP_SET_SRC_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value), 0, 0, 0)
#define TT_SET_SRC_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value) \
    TT_IP_SET_SRC_TILE_FACE_ROW_IDX(::ckernel::instrn_buffer, Tile_Face_Row_Sel, EngineSel, Value)
#define TTI_SET_SRC_TILE_FACE_ROW_IDX(Tile_Face_Row_Sel, EngineSel, Value) TT_IP_SET_SRC_TILE_FACE_ROW_IDX(nullptr, Tile_Face_Row_Sel, EngineSel, Value)

#define TT_OP_SFPABS(lreg_c, lreg_dest, instr_mod1)     TT_OP(0x7d, (((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPABS(IP, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPABS(lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPABS(lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPABS(::ckernel::instrn_buffer, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPABS(lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPABS(nullptr, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPADD(lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x85, (((lreg_a) << 16) + ((lreg_b) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPADD(IP, lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPADD(lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPADD(lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPADD(::ckernel::instrn_buffer, lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPADD(lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPADD(nullptr, lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPADDI(imm16_math, lreg_dest, instr_mod1)     TT_OP(0x75, (((imm16_math) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPADDI(IP, imm16_math, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPADDI(imm16_math, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPADDI(imm16_math, lreg_dest, instr_mod1)        TT_IP_SFPADDI(::ckernel::instrn_buffer, imm16_math, lreg_dest, instr_mod1)
#define TTI_SFPADDI(imm16_math, lreg_dest, instr_mod1)       TT_IP_SFPADDI(nullptr, imm16_math, lreg_dest, instr_mod1)

#define TT_OP_SFPAND(lreg_c, lreg_dest)     TT_OP(0x7e, (((lreg_c) << 8) + ((lreg_dest) << 4)))
#define TT_IP_SFPAND(IP, lreg_c, lreg_dest) TT_IP_INSN(IP, TT_OP_SFPAND(lreg_c, lreg_dest), 0, 0, 0)
#define TT_SFPAND(lreg_c, lreg_dest)        TT_IP_SFPAND(::ckernel::instrn_buffer, lreg_c, lreg_dest)
#define TTI_SFPAND(lreg_c, lreg_dest)       TT_IP_SFPAND(nullptr, lreg_c, lreg_dest)

#define TT_OP_SFPCAST(lreg_c, lreg_dest, instr_mod1)     TT_OP(0x90, (((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPCAST(IP, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPCAST(lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPCAST(lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPCAST(::ckernel::instrn_buffer, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPCAST(lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPCAST(nullptr, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPCOMPC     TT_OP(0x8b, 0)
#define TT_IP_SFPCOMPC(IP) TT_IP_INSN(IP, TT_OP_SFPCOMPC, 0, 0, 0)
#define TT_SFPCOMPC        TT_IP_SFPCOMPC(::ckernel::instrn_buffer)
#define TTI_SFPCOMPC       TT_IP_SFPCOMPC(nullptr)

#define TT_OP_SFPCONFIG(imm16_math, config_dest, instr_mod1)     TT_OP(0x91, (((imm16_math) << 8) + ((config_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPCONFIG(IP, imm16_math, config_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPCONFIG(imm16_math, config_dest, instr_mod1), 0, 0, 0)
#define TT_SFPCONFIG(imm16_math, config_dest, instr_mod1)        TT_IP_SFPCONFIG(::ckernel::instrn_buffer, imm16_math, config_dest, instr_mod1)
#define TTI_SFPCONFIG(imm16_math, config_dest, instr_mod1)       TT_IP_SFPCONFIG(nullptr, imm16_math, config_dest, instr_mod1)

#define TT_OP_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x76, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPDIVP2(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPDIVP2(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPDIVP2(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPENCC(imm12_math, instr_mod1)     TT_OP(0x8a, (((imm12_math) << 12) + ((instr_mod1) << 0)))
#define TT_IP_SFPENCC(IP, imm12_math, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPENCC(imm12_math, instr_mod1), 0, 0, 0)
#define TT_SFPENCC(imm12_math, instr_mod1)        TT_IP_SFPENCC(::ckernel::instrn_buffer, imm12_math, instr_mod1)
#define TTI_SFPENCC(imm12_math, instr_mod1)       TT_IP_SFPENCC(nullptr, imm12_math, instr_mod1)

#define TT_OP_SFPEXEXP(lreg_c, lreg_dest, instr_mod1)     TT_OP(0x77, (((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPEXEXP(IP, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPEXEXP(lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPEXEXP(lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPEXEXP(::ckernel::instrn_buffer, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPEXEXP(lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPEXEXP(nullptr, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPEXMAN(lreg_c, lreg_dest, instr_mod1)     TT_OP(0x78, (((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPEXMAN(IP, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPEXMAN(lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPEXMAN(lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPEXMAN(::ckernel::instrn_buffer, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPEXMAN(lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPEXMAN(nullptr, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPGT(imm12_math, lreg_c, lreg_dest, instr_mod1)     TT_OP(0x97, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPGT(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPGT(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPGT(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPGT(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPGT(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPGT(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x79, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPIADD(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPIADD(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPIADD(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPLE(imm12_math, lreg_c, lreg_dest, instr_mod1)     TT_OP(0x96, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPLE(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPLE(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPLE(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPLE(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPLE(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPLE(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr) \
    TT_OP(0x70, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((sfpu_addr_mode) << 13) + ((done) << 11) + ((dest_reg_addr) << 0)))
#define TT_IP_SFPLOAD(IP, lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr) \
    TT_IP_INSN(IP, TT_OP_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr), 0, 0, 0)
#define TT_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr) \
    TT_IP_SFPLOAD(::ckernel::instrn_buffer, lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr)
#define TTI_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr) TT_IP_SFPLOAD(nullptr, lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr)

#define TT_OP_SFPLOADI(lreg_ind, instr_mod0, imm16)     TT_OP(0x71, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((imm16) << 0)))
#define TT_IP_SFPLOADI(IP, lreg_ind, instr_mod0, imm16) TT_IP_INSN(IP, TT_OP_SFPLOADI(lreg_ind, instr_mod0, imm16), 0, 0, 0)
#define TT_SFPLOADI(lreg_ind, instr_mod0, imm16)        TT_IP_SFPLOADI(::ckernel::instrn_buffer, lreg_ind, instr_mod0, imm16)
#define TTI_SFPLOADI(lreg_ind, instr_mod0, imm16)       TT_IP_SFPLOADI(nullptr, lreg_ind, instr_mod0, imm16)

#define TT_OP_SFPLOADMACRO(seq_id, lreg_ind_lo, instr_mod0, sfpu_addr_mode, done, dest_reg_addr, lreg_ind_hi)                                   \
    TT_OP(                                                                                                                                      \
        0x93,                                                                                                                                   \
        (((seq_id) << 22) + ((lreg_ind_lo) << 20) + ((instr_mod0) << 16) + ((sfpu_addr_mode) << 13) + ((done) << 11) + ((dest_reg_addr) << 1) + \
         ((lreg_ind_hi) << 0)))
#define TT_IP_SFPLOADMACRO(IP, seq_id, lreg_ind_lo, instr_mod0, sfpu_addr_mode, done, dest_reg_addr, lreg_ind_hi) \
    TT_IP_INSN(IP, TT_OP_SFPLOADMACRO(seq_id, lreg_ind_lo, instr_mod0, sfpu_addr_mode, done, dest_reg_addr, lreg_ind_hi), 0, 0, 0)
#define TT_SFPLOADMACRO(seq_id, lreg_ind_lo, instr_mod0, sfpu_addr_mode, done, dest_reg_addr, lreg_ind_hi) \
    TT_IP_SFPLOADMACRO(::ckernel::instrn_buffer, seq_id, lreg_ind_lo, instr_mod0, sfpu_addr_mode, done, dest_reg_addr, lreg_ind_hi)
#define TTI_SFPLOADMACRO(seq_id, lreg_ind_lo, instr_mod0, sfpu_addr_mode, done, dest_reg_addr, lreg_ind_hi) \
    TT_IP_SFPLOADMACRO(nullptr, seq_id, lreg_ind_lo, instr_mod0, sfpu_addr_mode, done, dest_reg_addr, lreg_ind_hi)

#define TT_OP_SFPLUT(lreg_ind, instr_mod0)     TT_OP(0x73, (((lreg_ind) << 20) + ((instr_mod0) << 16)))
#define TT_IP_SFPLUT(IP, lreg_ind, instr_mod0) TT_IP_INSN(IP, TT_OP_SFPLUT(lreg_ind, instr_mod0), 0, 0, 0)
#define TT_SFPLUT(lreg_ind, instr_mod0)        TT_IP_SFPLUT(::ckernel::instrn_buffer, lreg_ind, instr_mod0)
#define TTI_SFPLUT(lreg_ind, instr_mod0)       TT_IP_SFPLUT(nullptr, lreg_ind, instr_mod0)

#define TT_OP_SFPLUTFP32(lreg_dest, instr_mod1)     TT_OP(0x95, (((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPLUTFP32(IP, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPLUTFP32(lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPLUTFP32(lreg_dest, instr_mod1)        TT_IP_SFPLUTFP32(::ckernel::instrn_buffer, lreg_dest, instr_mod1)
#define TTI_SFPLUTFP32(lreg_dest, instr_mod1)       TT_IP_SFPLUTFP32(nullptr, lreg_dest, instr_mod1)

#define TT_OP_SFPLZ(lreg_c, lreg_dest, instr_mod1)     TT_OP(0x81, (((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPLZ(IP, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPLZ(lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPLZ(lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPLZ(::ckernel::instrn_buffer, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPLZ(lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPLZ(nullptr, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPMAD(lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x84, (((lreg_a) << 16) + ((lreg_b) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPMAD(IP, lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPMAD(lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPMAD(lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPMAD(::ckernel::instrn_buffer, lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPMAD(lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPMAD(nullptr, lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPMOV(lreg_c, lreg_dest, instr_mod1)     TT_OP(0x7c, (((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPMOV(IP, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPMOV(lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPMOV(lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPMOV(::ckernel::instrn_buffer, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPMOV(lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPMOV(nullptr, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPMUL(lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x86, (((lreg_a) << 16) + ((lreg_b) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPMUL(IP, lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPMUL(lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPMUL(lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPMUL(::ckernel::instrn_buffer, lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPMUL(lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPMUL(nullptr, lreg_a, lreg_b, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPMUL24(lreg_a, lreg_b, lreg_dest, instr_mod1) \
    TT_OP(0x98, (((lreg_a) << 16) + ((lreg_b) << 12) + (9 << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPMUL24(IP, lreg_a, lreg_b, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPMUL24(lreg_a, lreg_b, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPMUL24(lreg_a, lreg_b, lreg_dest, instr_mod1)        TT_IP_SFPMUL24(::ckernel::instrn_buffer, lreg_a, lreg_b, lreg_dest, instr_mod1)
#define TTI_SFPMUL24(lreg_a, lreg_b, lreg_dest, instr_mod1)       TT_IP_SFPMUL24(nullptr, lreg_a, lreg_b, lreg_dest, instr_mod1)

#define TT_OP_SFPMULI(imm16_math, lreg_dest, instr_mod1)     TT_OP(0x74, (((imm16_math) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPMULI(IP, imm16_math, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPMULI(imm16_math, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPMULI(imm16_math, lreg_dest, instr_mod1)        TT_IP_SFPMULI(::ckernel::instrn_buffer, imm16_math, lreg_dest, instr_mod1)
#define TTI_SFPMULI(imm16_math, lreg_dest, instr_mod1)       TT_IP_SFPMULI(nullptr, imm16_math, lreg_dest, instr_mod1)

#define TT_OP_SFPNONLINEAR(lreg_c, lreg_dest, instr_mod1)     TT_OP(0x99, (((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPNONLINEAR(IP, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPNONLINEAR(lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPNONLINEAR(lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPNONLINEAR(::ckernel::instrn_buffer, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPNONLINEAR(lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPNONLINEAR(nullptr, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPNOP(srcs_wr_done, srcs_rd_done, dest_done)     TT_OP(0x8f, (((srcs_wr_done) << 2) + ((srcs_rd_done) << 1) + ((dest_done) << 0)))
#define TT_IP_SFPNOP(IP, srcs_wr_done, srcs_rd_done, dest_done) TT_IP_INSN(IP, TT_OP_SFPNOP(srcs_wr_done, srcs_rd_done, dest_done), 0, 0, 0)
#define TT_SFPNOP(srcs_wr_done, srcs_rd_done, dest_done)        TT_IP_SFPNOP(::ckernel::instrn_buffer, srcs_wr_done, srcs_rd_done, dest_done)
#define TTI_SFPNOP(srcs_wr_done, srcs_rd_done, dest_done)       TT_IP_SFPNOP(nullptr, srcs_wr_done, srcs_rd_done, dest_done)

#define TT_OP_SFPNOT(lreg_c, lreg_dest)     TT_OP(0x80, (((lreg_c) << 8) + ((lreg_dest) << 4)))
#define TT_IP_SFPNOT(IP, lreg_c, lreg_dest) TT_IP_INSN(IP, TT_OP_SFPNOT(lreg_c, lreg_dest), 0, 0, 0)
#define TT_SFPNOT(lreg_c, lreg_dest)        TT_IP_SFPNOT(::ckernel::instrn_buffer, lreg_c, lreg_dest)
#define TTI_SFPNOT(lreg_c, lreg_dest)       TT_IP_SFPNOT(nullptr, lreg_c, lreg_dest)

#define TT_OP_SFPOR(lreg_c, lreg_dest)     TT_OP(0x7f, (((lreg_c) << 8) + ((lreg_dest) << 4)))
#define TT_IP_SFPOR(IP, lreg_c, lreg_dest) TT_IP_INSN(IP, TT_OP_SFPOR(lreg_c, lreg_dest), 0, 0, 0)
#define TT_SFPOR(lreg_c, lreg_dest)        TT_IP_SFPOR(::ckernel::instrn_buffer, lreg_c, lreg_dest)
#define TTI_SFPOR(lreg_c, lreg_dest)       TT_IP_SFPOR(nullptr, lreg_c, lreg_dest)

#define TT_OP_SFPPOPC(instr_mod1)     TT_OP(0x88, (((instr_mod1) << 0)))
#define TT_IP_SFPPOPC(IP, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPPOPC(instr_mod1), 0, 0, 0)
#define TT_SFPPOPC(instr_mod1)        TT_IP_SFPPOPC(::ckernel::instrn_buffer, instr_mod1)
#define TTI_SFPPOPC(instr_mod1)       TT_IP_SFPPOPC(nullptr, instr_mod1)

#define TT_OP_SFPPUSHC(instr_mod1)     TT_OP(0x87, (((instr_mod1) << 0)))
#define TT_IP_SFPPUSHC(IP, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPPUSHC(instr_mod1), 0, 0, 0)
#define TT_SFPPUSHC(instr_mod1)        TT_IP_SFPPUSHC(::ckernel::instrn_buffer, instr_mod1)
#define TTI_SFPPUSHC(instr_mod1)       TT_IP_SFPPUSHC(nullptr, instr_mod1)

#define TT_OP_SFPSETCC(imm12_math, lreg_c, instr_mod1)     TT_OP(0x7b, (((imm12_math) << 12) + ((lreg_c) << 8) + ((instr_mod1) << 0)))
#define TT_IP_SFPSETCC(IP, imm12_math, lreg_c, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSETCC(imm12_math, lreg_c, instr_mod1), 0, 0, 0)
#define TT_SFPSETCC(imm12_math, lreg_c, instr_mod1)        TT_IP_SFPSETCC(::ckernel::instrn_buffer, imm12_math, lreg_c, instr_mod1)
#define TTI_SFPSETCC(imm12_math, lreg_c, instr_mod1)       TT_IP_SFPSETCC(nullptr, imm12_math, lreg_c, instr_mod1)

#define TT_OP_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x82, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPSETEXP(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPSETEXP(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPSETEXP(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x83, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPSETMAN(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPSETMAN(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPSETMAN(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x89, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPSETSGN(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPSETSGN(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPSETSGN(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x7a, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPSHFT(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPSHFT(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPSHFT(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPSHFT2(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x94, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPSHFT2(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSHFT2(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSHFT2(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPSHFT2(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSHFT2(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPSHFT2(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr) \
    TT_OP(0x72, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((sfpu_addr_mode) << 13) + ((done) << 11) + ((dest_reg_addr) << 0)))
#define TT_IP_SFPSTORE(IP, lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr) \
    TT_IP_INSN(IP, TT_OP_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr), 0, 0, 0)
#define TT_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr) \
    TT_IP_SFPSTORE(::ckernel::instrn_buffer, lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr)
#define TTI_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr) \
    TT_IP_SFPSTORE(nullptr, lreg_ind, instr_mod0, sfpu_addr_mode, done, dest_reg_addr)

#define TT_OP_SFPSWAP(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x92, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFPSWAP(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSWAP(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSWAP(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPSWAP(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSWAP(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPSWAP(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPTRANSP     TT_OP(0x8c, 0)
#define TT_IP_SFPTRANSP(IP) TT_IP_INSN(IP, TT_OP_SFPTRANSP, 0, 0, 0)
#define TT_SFPTRANSP        TT_IP_SFPTRANSP(::ckernel::instrn_buffer)
#define TTI_SFPTRANSP       TT_IP_SFPTRANSP(nullptr)

#define TT_OP_SFPXOR(lreg_c, lreg_dest)     TT_OP(0x8d, (((lreg_c) << 8) + ((lreg_dest) << 4)))
#define TT_IP_SFPXOR(IP, lreg_c, lreg_dest) TT_IP_INSN(IP, TT_OP_SFPXOR(lreg_c, lreg_dest), 0, 0, 0)
#define TT_SFPXOR(lreg_c, lreg_dest)        TT_IP_SFPXOR(::ckernel::instrn_buffer, lreg_c, lreg_dest)
#define TTI_SFPXOR(lreg_c, lreg_dest)       TT_IP_SFPXOR(nullptr, lreg_c, lreg_dest)

#define TT_OP_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_b, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x8e, (((rnd_mode) << 21) + ((imm8_math) << 16) + ((lreg_b) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_IP_SFP_STOCH_RND(IP, rnd_mode, imm8_math, lreg_b, lreg_c, lreg_dest, instr_mod1) \
    TT_IP_INSN(IP, TT_OP_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_b, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_b, lreg_c, lreg_dest, instr_mod1) \
    TT_IP_SFP_STOCH_RND(::ckernel::instrn_buffer, rnd_mode, imm8_math, lreg_b, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_b, lreg_c, lreg_dest, instr_mod1) \
    TT_IP_SFP_STOCH_RND(nullptr, rnd_mode, imm8_math, lreg_b, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SHIFTGPR(OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_OP(0x5c, (((OpB_is_Const) << 23) + ((OpSel) << 18) + ((Result_GPR_Index) << 12) + ((OpB_GPR_Index) << 6) + ((OpA_GPR_Index) << 0)))
#define TT_IP_SHIFTGPR(IP, OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_INSN(IP, TT_OP_SHIFTGPR(OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index), 0, 0, 0)
#define TT_SHIFTGPR(OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_SHIFTGPR(::ckernel::instrn_buffer, OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index)
#define TTI_SHIFTGPR(OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_SHIFTGPR(nullptr, OpB_is_Const, OpSel, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index)

#define TT_OP_SHIFTXB(addr_mode, rot_shift, shift_row)     TT_OP(0x18, (((addr_mode) << 14) + ((rot_shift) << 10) + ((shift_row) << 0)))
#define TT_IP_SHIFTXB(IP, addr_mode, rot_shift, shift_row) TT_IP_INSN(IP, TT_OP_SHIFTXB(addr_mode, rot_shift, shift_row), 0, 0, 0)
#define TT_SHIFTXB(addr_mode, rot_shift, shift_row)        TT_IP_SHIFTXB(::ckernel::instrn_buffer, addr_mode, rot_shift, shift_row)
#define TTI_SHIFTXB(addr_mode, rot_shift, shift_row)       TT_IP_SHIFTXB(nullptr, addr_mode, rot_shift, shift_row)

#define TT_OP_STALLWAIT(stall_res, wait_res_idx_2, wait_res_idx_1, wait_res_idx_0) \
    TT_OP(0xa2, (((stall_res) << 15) + ((wait_res_idx_2) << 10) + ((wait_res_idx_1) << 5) + ((wait_res_idx_0) << 0)))
#define TT_IP_STALLWAIT(IP, stall_res, wait_res_idx_2, wait_res_idx_1, wait_res_idx_0) \
    TT_IP_INSN(IP, TT_OP_STALLWAIT(stall_res, wait_res_idx_2, wait_res_idx_1, wait_res_idx_0), 0, 0, 0)
#define TT_STALLWAIT(stall_res, wait_res_idx_2, wait_res_idx_1, wait_res_idx_0) \
    TT_IP_STALLWAIT(::ckernel::instrn_buffer, stall_res, wait_res_idx_2, wait_res_idx_1, wait_res_idx_0)
#define TTI_STALLWAIT(stall_res, wait_res_idx_2, wait_res_idx_1, wait_res_idx_0) \
    TT_IP_STALLWAIT(nullptr, stall_res, wait_res_idx_2, wait_res_idx_1, wait_res_idx_0)

#define TT_OP_STOREIND(SizeSel, MemSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index) \
    TT_OP(0x66, (((SizeSel) << 22) + ((MemSel) << 21) + ((OffsetIndex) << 14) + ((AutoIncSpec) << 12) + ((Data_GPR_Index) << 6) + ((Addr_GPR_Index) << 0)))
#define TT_IP_STOREIND(IP, SizeSel, MemSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index) \
    TT_IP_INSN(IP, TT_OP_STOREIND(SizeSel, MemSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index), 0, 0, 0)
#define TT_STOREIND(SizeSel, MemSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index) \
    TT_IP_STOREIND(::ckernel::instrn_buffer, SizeSel, MemSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index)
#define TTI_STOREIND(SizeSel, MemSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index) \
    TT_IP_STOREIND(nullptr, SizeSel, MemSel, OffsetIndex, AutoIncSpec, Data_GPR_Index, Addr_GPR_Index)

#define TT_OP_STOREREG(Data_GPR_Index, RegAddr)     TT_OP(0x67, (((Data_GPR_Index) << 18) + ((RegAddr) << 0)))
#define TT_IP_STOREREG(IP, Data_GPR_Index, RegAddr) TT_IP_INSN(IP, TT_OP_STOREREG(Data_GPR_Index, RegAddr), 0, 0, 0)
#define TT_STOREREG(Data_GPR_Index, RegAddr)        TT_IP_STOREREG(::ckernel::instrn_buffer, Data_GPR_Index, RegAddr)
#define TTI_STOREREG(Data_GPR_Index, RegAddr)       TT_IP_STOREREG(nullptr, Data_GPR_Index, RegAddr)

#define TT_OP_SUBGPR(OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_OP(0x59, (((OpB_is_Const) << 23) + ((Result_GPR_Index) << 12) + ((OpB_GPR_Index) << 6) + ((OpA_GPR_Index) << 0)))
#define TT_IP_SUBGPR(IP, OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_INSN(IP, TT_OP_SUBGPR(OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index), 0, 0, 0)
#define TT_SUBGPR(OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_SUBGPR(::ckernel::instrn_buffer, OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index)
#define TTI_SUBGPR(OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index) \
    TT_IP_SUBGPR(nullptr, OpB_is_Const, Result_GPR_Index, OpB_GPR_Index, OpA_GPR_Index)

#define TT_OP_UNPACR0_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                         \
        0x47,                                                                                                                                      \
        (((Dst_Face_Idx) << 14) + ((Src_Face_Idx) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) + ((Src_Tile_Offset_Idx_Inc) << 7) +                  \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR0_FACE(IP, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(                                                                                                                                        \
        IP,                                                                                                                                            \
        TT_OP_UNPACR0_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid),    \
        0,                                                                                                                                             \
        0,                                                                                                                                             \
        0)
#define TT_UNPACR0_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR0_FACE(                                                                                                                         \
        ::ckernel::instrn_buffer, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR0_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR0_FACE(nullptr, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR0_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                                     \
        0x3f,                                                                                                                                                  \
        (((Dst_Face_Idx_Inc) << 14) + ((Src_Face_Idx_Inc) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) + ((Src_Tile_Offset_Idx_Inc) << 7) +                      \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR0_FACE_INC(                                                                                                              \
    IP, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)      \
    TT_IP_INSN(                                                                                                                              \
        IP,                                                                                                                                  \
        TT_OP_UNPACR0_FACE_INC(                                                                                                              \
            Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid), \
        0,                                                                                                                                   \
        0,                                                                                                                                   \
        0)
#define TT_UNPACR0_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR0_FACE_INC(                                                                                                                                 \
        ::ckernel::instrn_buffer,                                                                                                                           \
        Dst_Face_Idx_Inc,                                                                                                                                   \
        Src_Face_Idx_Inc,                                                                                                                                   \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                            \
        Src_Tile_Offset_Idx_Inc,                                                                                                                            \
        Buffer_Descriptor_Table_Sel,                                                                                                                        \
        SetDatValid)
#define TTI_UNPACR0_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR0_FACE_INC(                                                                                                                                  \
        nullptr, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR0_ROW(                                                                                                                            \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                            \
        0x4b,                                                                                                                                         \
        (((Dst_Row_Idx) << 20) + ((Src_Row_Idx) << 16) + ((Dst_Face_Idx) << 14) + ((Src_Face_Idx) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) +        \
         ((Src_Tile_Offset_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR0_ROW(                                                                                                                                     \
    IP, Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)      \
    TT_IP_INSN(                                                                                                                                                \
        IP,                                                                                                                                                    \
        TT_OP_UNPACR0_ROW(                                                                                                                                     \
            Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid), \
        0,                                                                                                                                                     \
        0,                                                                                                                                                     \
        0)
#define TT_UNPACR0_ROW(                                                                                                                               \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR0_ROW(                                                                                                                                \
        ::ckernel::instrn_buffer,                                                                                                                     \
        Dst_Row_Idx,                                                                                                                                  \
        Src_Row_Idx,                                                                                                                                  \
        Dst_Face_Idx,                                                                                                                                 \
        Src_Face_Idx,                                                                                                                                 \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                      \
        Src_Tile_Offset_Idx_Inc,                                                                                                                      \
        Buffer_Descriptor_Table_Sel,                                                                                                                  \
        SetDatValid)
#define TTI_UNPACR0_ROW(                                                                                                                              \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR0_ROW(                                                                                                                                \
        nullptr,                                                                                                                                      \
        Dst_Row_Idx,                                                                                                                                  \
        Src_Row_Idx,                                                                                                                                  \
        Dst_Face_Idx,                                                                                                                                 \
        Src_Face_Idx,                                                                                                                                 \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                      \
        Src_Tile_Offset_Idx_Inc,                                                                                                                      \
        Buffer_Descriptor_Table_Sel,                                                                                                                  \
        SetDatValid)

#define TT_OP_UNPACR0_ROW_INC(                                                                                                                                 \
    Dst_Row_Idx_Inc,                                                                                                                                           \
    Src_Row_Idx_Inc,                                                                                                                                           \
    Dst_Face_Idx_Inc,                                                                                                                                          \
    Src_Face_Idx_Inc,                                                                                                                                          \
    Dst_Tile_Offset_Idx_Inc,                                                                                                                                   \
    Src_Tile_Offset_Idx_Inc,                                                                                                                                   \
    Buffer_Descriptor_Table_Sel,                                                                                                                               \
    SetDatValid)                                                                                                                                               \
    TT_OP(                                                                                                                                                     \
        0x4c,                                                                                                                                                  \
        (((Dst_Row_Idx_Inc) << 20) + ((Src_Row_Idx_Inc) << 16) + ((Dst_Face_Idx_Inc) << 14) + ((Src_Face_Idx_Inc) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) + \
         ((Src_Tile_Offset_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR0_ROW_INC(           \
    IP,                                  \
    Dst_Row_Idx_Inc,                     \
    Src_Row_Idx_Inc,                     \
    Dst_Face_Idx_Inc,                    \
    Src_Face_Idx_Inc,                    \
    Dst_Tile_Offset_Idx_Inc,             \
    Src_Tile_Offset_Idx_Inc,             \
    Buffer_Descriptor_Table_Sel,         \
    SetDatValid)                         \
    TT_IP_INSN(                          \
        IP,                              \
        TT_OP_UNPACR0_ROW_INC(           \
            Dst_Row_Idx_Inc,             \
            Src_Row_Idx_Inc,             \
            Dst_Face_Idx_Inc,            \
            Src_Face_Idx_Inc,            \
            Dst_Tile_Offset_Idx_Inc,     \
            Src_Tile_Offset_Idx_Inc,     \
            Buffer_Descriptor_Table_Sel, \
            SetDatValid),                \
        0,                               \
        0,                               \
        0)
#define TT_UNPACR0_ROW_INC(          \
    Dst_Row_Idx_Inc,                 \
    Src_Row_Idx_Inc,                 \
    Dst_Face_Idx_Inc,                \
    Src_Face_Idx_Inc,                \
    Dst_Tile_Offset_Idx_Inc,         \
    Src_Tile_Offset_Idx_Inc,         \
    Buffer_Descriptor_Table_Sel,     \
    SetDatValid)                     \
    TT_IP_UNPACR0_ROW_INC(           \
        ::ckernel::instrn_buffer,    \
        Dst_Row_Idx_Inc,             \
        Src_Row_Idx_Inc,             \
        Dst_Face_Idx_Inc,            \
        Src_Face_Idx_Inc,            \
        Dst_Tile_Offset_Idx_Inc,     \
        Src_Tile_Offset_Idx_Inc,     \
        Buffer_Descriptor_Table_Sel, \
        SetDatValid)
#define TTI_UNPACR0_ROW_INC(         \
    Dst_Row_Idx_Inc,                 \
    Src_Row_Idx_Inc,                 \
    Dst_Face_Idx_Inc,                \
    Src_Face_Idx_Inc,                \
    Dst_Tile_Offset_Idx_Inc,         \
    Src_Tile_Offset_Idx_Inc,         \
    Buffer_Descriptor_Table_Sel,     \
    SetDatValid)                     \
    TT_IP_UNPACR0_ROW_INC(           \
        nullptr,                     \
        Dst_Row_Idx_Inc,             \
        Src_Row_Idx_Inc,             \
        Dst_Face_Idx_Inc,            \
        Src_Face_Idx_Inc,            \
        Dst_Tile_Offset_Idx_Inc,     \
        Src_Tile_Offset_Idx_Inc,     \
        Buffer_Descriptor_Table_Sel, \
        SetDatValid)

#define TT_OP_UNPACR0_STRIDE(                                                                                                                          \
    Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                             \
        0x6f,                                                                                                                                          \
        (((Src_Reg_Y_Cntr_Incr) << 20) + ((L1_Tile_Idx_or_Tile_Idx_Inc) << 17) + ((Tile_Idx_Inc) << 16) + ((Row_Mask_Reg_Sel) << 13) +                 \
         ((L1_16datums_Row_Index) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR0_STRIDE(                                                                                                                              \
    IP, Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(                                                                                                                                            \
        IP,                                                                                                                                                \
        TT_OP_UNPACR0_STRIDE(                                                                                                                              \
            Src_Reg_Y_Cntr_Incr,                                                                                                                           \
            L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                                   \
            Tile_Idx_Inc,                                                                                                                                  \
            Row_Mask_Reg_Sel,                                                                                                                              \
            L1_16datums_Row_Index,                                                                                                                         \
            Buffer_Descriptor_Table_Sel,                                                                                                                   \
            SetDatValid),                                                                                                                                  \
        0,                                                                                                                                                 \
        0,                                                                                                                                                 \
        0)
#define TT_UNPACR0_STRIDE(                                                                                                                             \
    Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR0_STRIDE(                                                                                                                              \
        ::ckernel::instrn_buffer,                                                                                                                      \
        Src_Reg_Y_Cntr_Incr,                                                                                                                           \
        L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                                   \
        Tile_Idx_Inc,                                                                                                                                  \
        Row_Mask_Reg_Sel,                                                                                                                              \
        L1_16datums_Row_Index,                                                                                                                         \
        Buffer_Descriptor_Table_Sel,                                                                                                                   \
        SetDatValid)
#define TTI_UNPACR0_STRIDE(                                                                                                                            \
    Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR0_STRIDE(                                                                                                                              \
        nullptr,                                                                                                                                       \
        Src_Reg_Y_Cntr_Incr,                                                                                                                           \
        L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                                   \
        Tile_Idx_Inc,                                                                                                                                  \
        Row_Mask_Reg_Sel,                                                                                                                              \
        L1_16datums_Row_Index,                                                                                                                         \
        Buffer_Descriptor_Table_Sel,                                                                                                                   \
        SetDatValid)

#define TT_OP_UNPACR0_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(0x3c, (((Dst_Tile_Idx) << 15) + ((Src_Tile_Idx) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR0_TILE(IP, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(IP, TT_OP_UNPACR0_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid), 0, 0, 0)
#define TT_UNPACR0_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR0_TILE(::ckernel::instrn_buffer, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR0_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR0_TILE(nullptr, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR0_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(0x44, (((Dst_Tile_Idx_Inc) << 15) + ((Src_Tile_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR0_TILE_INC(IP, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(IP, TT_OP_UNPACR0_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid), 0, 0, 0)
#define TT_UNPACR0_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR0_TILE_INC(::ckernel::instrn_buffer, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR0_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR0_TILE_INC(nullptr, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR1_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                         \
        0x6a,                                                                                                                                      \
        (((Dst_Face_Idx) << 14) + ((Src_Face_Idx) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) + ((Src_Tile_Offset_Idx_Inc) << 7) +                  \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR1_FACE(IP, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(                                                                                                                                        \
        IP,                                                                                                                                            \
        TT_OP_UNPACR1_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid),    \
        0,                                                                                                                                             \
        0,                                                                                                                                             \
        0)
#define TT_UNPACR1_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR1_FACE(                                                                                                                         \
        ::ckernel::instrn_buffer, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR1_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR1_FACE(nullptr, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR1_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                                     \
        0x6b,                                                                                                                                                  \
        (((Dst_Face_Idx_Inc) << 14) + ((Src_Face_Idx_Inc) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) + ((Src_Tile_Offset_Idx_Inc) << 7) +                      \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR1_FACE_INC(                                                                                                              \
    IP, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)      \
    TT_IP_INSN(                                                                                                                              \
        IP,                                                                                                                                  \
        TT_OP_UNPACR1_FACE_INC(                                                                                                              \
            Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid), \
        0,                                                                                                                                   \
        0,                                                                                                                                   \
        0)
#define TT_UNPACR1_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR1_FACE_INC(                                                                                                                                 \
        ::ckernel::instrn_buffer,                                                                                                                           \
        Dst_Face_Idx_Inc,                                                                                                                                   \
        Src_Face_Idx_Inc,                                                                                                                                   \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                            \
        Src_Tile_Offset_Idx_Inc,                                                                                                                            \
        Buffer_Descriptor_Table_Sel,                                                                                                                        \
        SetDatValid)
#define TTI_UNPACR1_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR1_FACE_INC(                                                                                                                                  \
        nullptr, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR1_ROW(                                                                                                                            \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                            \
        0x6c,                                                                                                                                         \
        (((Dst_Row_Idx) << 20) + ((Src_Row_Idx) << 16) + ((Dst_Face_Idx) << 14) + ((Src_Face_Idx) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) +        \
         ((Src_Tile_Offset_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR1_ROW(                                                                                                                                     \
    IP, Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)      \
    TT_IP_INSN(                                                                                                                                                \
        IP,                                                                                                                                                    \
        TT_OP_UNPACR1_ROW(                                                                                                                                     \
            Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid), \
        0,                                                                                                                                                     \
        0,                                                                                                                                                     \
        0)
#define TT_UNPACR1_ROW(                                                                                                                               \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR1_ROW(                                                                                                                                \
        ::ckernel::instrn_buffer,                                                                                                                     \
        Dst_Row_Idx,                                                                                                                                  \
        Src_Row_Idx,                                                                                                                                  \
        Dst_Face_Idx,                                                                                                                                 \
        Src_Face_Idx,                                                                                                                                 \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                      \
        Src_Tile_Offset_Idx_Inc,                                                                                                                      \
        Buffer_Descriptor_Table_Sel,                                                                                                                  \
        SetDatValid)
#define TTI_UNPACR1_ROW(                                                                                                                              \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR1_ROW(                                                                                                                                \
        nullptr,                                                                                                                                      \
        Dst_Row_Idx,                                                                                                                                  \
        Src_Row_Idx,                                                                                                                                  \
        Dst_Face_Idx,                                                                                                                                 \
        Src_Face_Idx,                                                                                                                                 \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                      \
        Src_Tile_Offset_Idx_Inc,                                                                                                                      \
        Buffer_Descriptor_Table_Sel,                                                                                                                  \
        SetDatValid)

#define TT_OP_UNPACR1_ROW_INC(                                                                                                                                 \
    Dst_Row_Idx_Inc,                                                                                                                                           \
    Src_Row_Idx_Inc,                                                                                                                                           \
    Dst_Face_Idx_Inc,                                                                                                                                          \
    Src_Face_Idx_Inc,                                                                                                                                          \
    Dst_Tile_Offset_Idx_Inc,                                                                                                                                   \
    Src_Tile_Offset_Idx_Inc,                                                                                                                                   \
    Buffer_Descriptor_Table_Sel,                                                                                                                               \
    SetDatValid)                                                                                                                                               \
    TT_OP(                                                                                                                                                     \
        0x6d,                                                                                                                                                  \
        (((Dst_Row_Idx_Inc) << 20) + ((Src_Row_Idx_Inc) << 16) + ((Dst_Face_Idx_Inc) << 14) + ((Src_Face_Idx_Inc) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) + \
         ((Src_Tile_Offset_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR1_ROW_INC(           \
    IP,                                  \
    Dst_Row_Idx_Inc,                     \
    Src_Row_Idx_Inc,                     \
    Dst_Face_Idx_Inc,                    \
    Src_Face_Idx_Inc,                    \
    Dst_Tile_Offset_Idx_Inc,             \
    Src_Tile_Offset_Idx_Inc,             \
    Buffer_Descriptor_Table_Sel,         \
    SetDatValid)                         \
    TT_IP_INSN(                          \
        IP,                              \
        TT_OP_UNPACR1_ROW_INC(           \
            Dst_Row_Idx_Inc,             \
            Src_Row_Idx_Inc,             \
            Dst_Face_Idx_Inc,            \
            Src_Face_Idx_Inc,            \
            Dst_Tile_Offset_Idx_Inc,     \
            Src_Tile_Offset_Idx_Inc,     \
            Buffer_Descriptor_Table_Sel, \
            SetDatValid),                \
        0,                               \
        0,                               \
        0)
#define TT_UNPACR1_ROW_INC(          \
    Dst_Row_Idx_Inc,                 \
    Src_Row_Idx_Inc,                 \
    Dst_Face_Idx_Inc,                \
    Src_Face_Idx_Inc,                \
    Dst_Tile_Offset_Idx_Inc,         \
    Src_Tile_Offset_Idx_Inc,         \
    Buffer_Descriptor_Table_Sel,     \
    SetDatValid)                     \
    TT_IP_UNPACR1_ROW_INC(           \
        ::ckernel::instrn_buffer,    \
        Dst_Row_Idx_Inc,             \
        Src_Row_Idx_Inc,             \
        Dst_Face_Idx_Inc,            \
        Src_Face_Idx_Inc,            \
        Dst_Tile_Offset_Idx_Inc,     \
        Src_Tile_Offset_Idx_Inc,     \
        Buffer_Descriptor_Table_Sel, \
        SetDatValid)
#define TTI_UNPACR1_ROW_INC(         \
    Dst_Row_Idx_Inc,                 \
    Src_Row_Idx_Inc,                 \
    Dst_Face_Idx_Inc,                \
    Src_Face_Idx_Inc,                \
    Dst_Tile_Offset_Idx_Inc,         \
    Src_Tile_Offset_Idx_Inc,         \
    Buffer_Descriptor_Table_Sel,     \
    SetDatValid)                     \
    TT_IP_UNPACR1_ROW_INC(           \
        nullptr,                     \
        Dst_Row_Idx_Inc,             \
        Src_Row_Idx_Inc,             \
        Dst_Face_Idx_Inc,            \
        Src_Face_Idx_Inc,            \
        Dst_Tile_Offset_Idx_Inc,     \
        Src_Tile_Offset_Idx_Inc,     \
        Buffer_Descriptor_Table_Sel, \
        SetDatValid)

#define TT_OP_UNPACR1_STRIDE(                                                                                                                          \
    Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                             \
        0xaa,                                                                                                                                          \
        (((Src_Reg_Y_Cntr_Incr) << 20) + ((L1_Tile_Idx_or_Tile_Idx_Inc) << 17) + ((Tile_Idx_Inc) << 16) + ((Row_Mask_Reg_Sel) << 13) +                 \
         ((L1_16datums_Row_Index) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR1_STRIDE(                                                                                                                              \
    IP, Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(                                                                                                                                            \
        IP,                                                                                                                                                \
        TT_OP_UNPACR1_STRIDE(                                                                                                                              \
            Src_Reg_Y_Cntr_Incr,                                                                                                                           \
            L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                                   \
            Tile_Idx_Inc,                                                                                                                                  \
            Row_Mask_Reg_Sel,                                                                                                                              \
            L1_16datums_Row_Index,                                                                                                                         \
            Buffer_Descriptor_Table_Sel,                                                                                                                   \
            SetDatValid),                                                                                                                                  \
        0,                                                                                                                                                 \
        0,                                                                                                                                                 \
        0)
#define TT_UNPACR1_STRIDE(                                                                                                                             \
    Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR1_STRIDE(                                                                                                                              \
        ::ckernel::instrn_buffer,                                                                                                                      \
        Src_Reg_Y_Cntr_Incr,                                                                                                                           \
        L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                                   \
        Tile_Idx_Inc,                                                                                                                                  \
        Row_Mask_Reg_Sel,                                                                                                                              \
        L1_16datums_Row_Index,                                                                                                                         \
        Buffer_Descriptor_Table_Sel,                                                                                                                   \
        SetDatValid)
#define TTI_UNPACR1_STRIDE(                                                                                                                            \
    Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR1_STRIDE(                                                                                                                              \
        nullptr,                                                                                                                                       \
        Src_Reg_Y_Cntr_Incr,                                                                                                                           \
        L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                                   \
        Tile_Idx_Inc,                                                                                                                                  \
        Row_Mask_Reg_Sel,                                                                                                                              \
        L1_16datums_Row_Index,                                                                                                                         \
        Buffer_Descriptor_Table_Sel,                                                                                                                   \
        SetDatValid)

#define TT_OP_UNPACR1_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(0x5f, (((Dst_Tile_Idx) << 15) + ((Src_Tile_Idx) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR1_TILE(IP, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(IP, TT_OP_UNPACR1_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid), 0, 0, 0)
#define TT_UNPACR1_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR1_TILE(::ckernel::instrn_buffer, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR1_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR1_TILE(nullptr, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR1_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(0x69, (((Dst_Tile_Idx_Inc) << 15) + ((Src_Tile_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR1_TILE_INC(IP, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(IP, TT_OP_UNPACR1_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid), 0, 0, 0)
#define TT_UNPACR1_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR1_TILE_INC(::ckernel::instrn_buffer, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR1_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR1_TILE_INC(nullptr, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR2_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                         \
        0x9d,                                                                                                                                      \
        (((Dst_Face_Idx) << 14) + ((Src_Face_Idx) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) + ((Src_Tile_Offset_Idx_Inc) << 7) +                  \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR2_FACE(IP, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(                                                                                                                                        \
        IP,                                                                                                                                            \
        TT_OP_UNPACR2_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid),    \
        0,                                                                                                                                             \
        0,                                                                                                                                             \
        0)
#define TT_UNPACR2_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR2_FACE(                                                                                                                         \
        ::ckernel::instrn_buffer, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR2_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR2_FACE(nullptr, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR2_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                                     \
        0x9e,                                                                                                                                                  \
        (((Dst_Face_Idx_Inc) << 14) + ((Src_Face_Idx_Inc) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) + ((Src_Tile_Offset_Idx_Inc) << 7) +                      \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR2_FACE_INC(                                                                                                              \
    IP, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)      \
    TT_IP_INSN(                                                                                                                              \
        IP,                                                                                                                                  \
        TT_OP_UNPACR2_FACE_INC(                                                                                                              \
            Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid), \
        0,                                                                                                                                   \
        0,                                                                                                                                   \
        0)
#define TT_UNPACR2_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR2_FACE_INC(                                                                                                                                 \
        ::ckernel::instrn_buffer,                                                                                                                           \
        Dst_Face_Idx_Inc,                                                                                                                                   \
        Src_Face_Idx_Inc,                                                                                                                                   \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                            \
        Src_Tile_Offset_Idx_Inc,                                                                                                                            \
        Buffer_Descriptor_Table_Sel,                                                                                                                        \
        SetDatValid)
#define TTI_UNPACR2_FACE_INC(Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR2_FACE_INC(                                                                                                                                  \
        nullptr, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR2_ROW(                                                                                                                            \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                            \
        0x9f,                                                                                                                                         \
        (((Dst_Row_Idx) << 20) + ((Src_Row_Idx) << 16) + ((Dst_Face_Idx) << 14) + ((Src_Face_Idx) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) +        \
         ((Src_Tile_Offset_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR2_ROW(                                                                                                                                     \
    IP, Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)      \
    TT_IP_INSN(                                                                                                                                                \
        IP,                                                                                                                                                    \
        TT_OP_UNPACR2_ROW(                                                                                                                                     \
            Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid), \
        0,                                                                                                                                                     \
        0,                                                                                                                                                     \
        0)
#define TT_UNPACR2_ROW(                                                                                                                               \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR2_ROW(                                                                                                                                \
        ::ckernel::instrn_buffer,                                                                                                                     \
        Dst_Row_Idx,                                                                                                                                  \
        Src_Row_Idx,                                                                                                                                  \
        Dst_Face_Idx,                                                                                                                                 \
        Src_Face_Idx,                                                                                                                                 \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                      \
        Src_Tile_Offset_Idx_Inc,                                                                                                                      \
        Buffer_Descriptor_Table_Sel,                                                                                                                  \
        SetDatValid)
#define TTI_UNPACR2_ROW(                                                                                                                              \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR2_ROW(                                                                                                                                \
        nullptr,                                                                                                                                      \
        Dst_Row_Idx,                                                                                                                                  \
        Src_Row_Idx,                                                                                                                                  \
        Dst_Face_Idx,                                                                                                                                 \
        Src_Face_Idx,                                                                                                                                 \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                      \
        Src_Tile_Offset_Idx_Inc,                                                                                                                      \
        Buffer_Descriptor_Table_Sel,                                                                                                                  \
        SetDatValid)

#define TT_OP_UNPACR2_ROW_INC(                                                                                                                                 \
    Dst_Row_Idx_Inc,                                                                                                                                           \
    Src_Row_Idx_Inc,                                                                                                                                           \
    Dst_Face_Idx_Inc,                                                                                                                                          \
    Src_Face_Idx_Inc,                                                                                                                                          \
    Dst_Tile_Offset_Idx_Inc,                                                                                                                                   \
    Src_Tile_Offset_Idx_Inc,                                                                                                                                   \
    Buffer_Descriptor_Table_Sel,                                                                                                                               \
    SetDatValid)                                                                                                                                               \
    TT_OP(                                                                                                                                                     \
        0xa8,                                                                                                                                                  \
        (((Dst_Row_Idx_Inc) << 20) + ((Src_Row_Idx_Inc) << 16) + ((Dst_Face_Idx_Inc) << 14) + ((Src_Face_Idx_Inc) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) + \
         ((Src_Tile_Offset_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR2_ROW_INC(           \
    IP,                                  \
    Dst_Row_Idx_Inc,                     \
    Src_Row_Idx_Inc,                     \
    Dst_Face_Idx_Inc,                    \
    Src_Face_Idx_Inc,                    \
    Dst_Tile_Offset_Idx_Inc,             \
    Src_Tile_Offset_Idx_Inc,             \
    Buffer_Descriptor_Table_Sel,         \
    SetDatValid)                         \
    TT_IP_INSN(                          \
        IP,                              \
        TT_OP_UNPACR2_ROW_INC(           \
            Dst_Row_Idx_Inc,             \
            Src_Row_Idx_Inc,             \
            Dst_Face_Idx_Inc,            \
            Src_Face_Idx_Inc,            \
            Dst_Tile_Offset_Idx_Inc,     \
            Src_Tile_Offset_Idx_Inc,     \
            Buffer_Descriptor_Table_Sel, \
            SetDatValid),                \
        0,                               \
        0,                               \
        0)
#define TT_UNPACR2_ROW_INC(          \
    Dst_Row_Idx_Inc,                 \
    Src_Row_Idx_Inc,                 \
    Dst_Face_Idx_Inc,                \
    Src_Face_Idx_Inc,                \
    Dst_Tile_Offset_Idx_Inc,         \
    Src_Tile_Offset_Idx_Inc,         \
    Buffer_Descriptor_Table_Sel,     \
    SetDatValid)                     \
    TT_IP_UNPACR2_ROW_INC(           \
        ::ckernel::instrn_buffer,    \
        Dst_Row_Idx_Inc,             \
        Src_Row_Idx_Inc,             \
        Dst_Face_Idx_Inc,            \
        Src_Face_Idx_Inc,            \
        Dst_Tile_Offset_Idx_Inc,     \
        Src_Tile_Offset_Idx_Inc,     \
        Buffer_Descriptor_Table_Sel, \
        SetDatValid)
#define TTI_UNPACR2_ROW_INC(         \
    Dst_Row_Idx_Inc,                 \
    Src_Row_Idx_Inc,                 \
    Dst_Face_Idx_Inc,                \
    Src_Face_Idx_Inc,                \
    Dst_Tile_Offset_Idx_Inc,         \
    Src_Tile_Offset_Idx_Inc,         \
    Buffer_Descriptor_Table_Sel,     \
    SetDatValid)                     \
    TT_IP_UNPACR2_ROW_INC(           \
        nullptr,                     \
        Dst_Row_Idx_Inc,             \
        Src_Row_Idx_Inc,             \
        Dst_Face_Idx_Inc,            \
        Src_Face_Idx_Inc,            \
        Dst_Tile_Offset_Idx_Inc,     \
        Src_Tile_Offset_Idx_Inc,     \
        Buffer_Descriptor_Table_Sel, \
        SetDatValid)

#define TT_OP_UNPACR2_STRIDE(                                                                                                                          \
    Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                             \
        0x4e,                                                                                                                                          \
        (((Src_Reg_Y_Cntr_Incr) << 20) + ((L1_Tile_Idx_or_Tile_Idx_Inc) << 17) + ((Tile_Idx_Inc) << 16) + ((Row_Mask_Reg_Sel) << 13) +                 \
         ((L1_16datums_Row_Index) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR2_STRIDE(                                                                                                                              \
    IP, Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(                                                                                                                                            \
        IP,                                                                                                                                                \
        TT_OP_UNPACR2_STRIDE(                                                                                                                              \
            Src_Reg_Y_Cntr_Incr,                                                                                                                           \
            L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                                   \
            Tile_Idx_Inc,                                                                                                                                  \
            Row_Mask_Reg_Sel,                                                                                                                              \
            L1_16datums_Row_Index,                                                                                                                         \
            Buffer_Descriptor_Table_Sel,                                                                                                                   \
            SetDatValid),                                                                                                                                  \
        0,                                                                                                                                                 \
        0,                                                                                                                                                 \
        0)
#define TT_UNPACR2_STRIDE(                                                                                                                             \
    Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR2_STRIDE(                                                                                                                              \
        ::ckernel::instrn_buffer,                                                                                                                      \
        Src_Reg_Y_Cntr_Incr,                                                                                                                           \
        L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                                   \
        Tile_Idx_Inc,                                                                                                                                  \
        Row_Mask_Reg_Sel,                                                                                                                              \
        L1_16datums_Row_Index,                                                                                                                         \
        Buffer_Descriptor_Table_Sel,                                                                                                                   \
        SetDatValid)
#define TTI_UNPACR2_STRIDE(                                                                                                                            \
    Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR2_STRIDE(                                                                                                                              \
        nullptr,                                                                                                                                       \
        Src_Reg_Y_Cntr_Incr,                                                                                                                           \
        L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                                   \
        Tile_Idx_Inc,                                                                                                                                  \
        Row_Mask_Reg_Sel,                                                                                                                              \
        L1_16datums_Row_Index,                                                                                                                         \
        Buffer_Descriptor_Table_Sel,                                                                                                                   \
        SetDatValid)

#define TT_OP_UNPACR2_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(0x9b, (((Dst_Tile_Idx) << 15) + ((Src_Tile_Idx) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR2_TILE(IP, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(IP, TT_OP_UNPACR2_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid), 0, 0, 0)
#define TT_UNPACR2_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR2_TILE(::ckernel::instrn_buffer, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR2_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR2_TILE(nullptr, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR2_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(0x9c, (((Dst_Tile_Idx_Inc) << 15) + ((Src_Tile_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR2_TILE_INC(IP, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(IP, TT_OP_UNPACR2_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid), 0, 0, 0)
#define TT_UNPACR2_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR2_TILE_INC(::ckernel::instrn_buffer, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR2_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR2_TILE_INC(nullptr, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR_DEST_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                             \
        0xae,                                                                                                                                          \
        (((Dst_Face_Idx) << 14) + ((Src_Face_Idx) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) + ((Src_Tile_Offset_Idx_Inc) << 7) +                      \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR_DEST_FACE(IP, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(                                                                                                                                            \
        IP,                                                                                                                                                \
        TT_OP_UNPACR_DEST_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid),    \
        0,                                                                                                                                                 \
        0,                                                                                                                                                 \
        0)
#define TT_UNPACR_DEST_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_DEST_FACE(                                                                                                                         \
        ::ckernel::instrn_buffer, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR_DEST_FACE(Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_DEST_FACE(nullptr, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid))

#define TT_OP_UNPACR_DEST_FACE_INC(                                                                                                       \
    Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)       \
    TT_OP(                                                                                                                                \
        0xaf,                                                                                                                             \
        (((Dst_Face_Idx_Inc) << 14) + ((Src_Face_Idx_Inc) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) + ((Src_Tile_Offset_Idx_Inc) << 7) + \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR_DEST_FACE_INC(                                                                                                          \
    IP, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)      \
    TT_IP_INSN(                                                                                                                              \
        IP,                                                                                                                                  \
        TT_OP_UNPACR_DEST_FACE_INC(                                                                                                          \
            Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid), \
        0,                                                                                                                                   \
        0,                                                                                                                                   \
        0)
#define TT_UNPACR_DEST_FACE_INC(                                                                                                    \
    Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_DEST_FACE_INC(                                                                                                     \
        ::ckernel::instrn_buffer,                                                                                                   \
        Dst_Face_Idx_Inc,                                                                                                           \
        Src_Face_Idx_Inc,                                                                                                           \
        Dst_Tile_Offset_Idx_Inc,                                                                                                    \
        Src_Tile_Offset_Idx_Inc,                                                                                                    \
        Buffer_Descriptor_Table_Sel,                                                                                                \
        SetDatValid)
#define TTI_UNPACR_DEST_FACE_INC(                                                                                                   \
    Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_DEST_FACE_INC(                                                                                                     \
        nullptr, Dst_Face_Idx_Inc, Src_Face_Idx_Inc, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR_DEST_ROW(                                                                                                                        \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                            \
        0xa7,                                                                                                                                         \
        (((Dst_Row_Idx) << 20) + ((Src_Row_Idx) << 16) + ((Dst_Face_Idx) << 14) + ((Src_Face_Idx) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) +        \
         ((Src_Tile_Offset_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR_DEST_ROW(                                                                                                                                 \
    IP, Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)      \
    TT_IP_INSN(                                                                                                                                                \
        IP,                                                                                                                                                    \
        TT_OP_UNPACR_DEST_ROW(                                                                                                                                 \
            Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid), \
        0,                                                                                                                                                     \
        0,                                                                                                                                                     \
        0)
#define TT_UNPACR_DEST_ROW(                                                                                                                           \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_DEST_ROW(                                                                                                                            \
        ::ckernel::instrn_buffer,                                                                                                                     \
        Dst_Row_Idx,                                                                                                                                  \
        Src_Row_Idx,                                                                                                                                  \
        Dst_Face_Idx,                                                                                                                                 \
        Src_Face_Idx,                                                                                                                                 \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                      \
        Src_Tile_Offset_Idx_Inc,                                                                                                                      \
        Buffer_Descriptor_Table_Sel,                                                                                                                  \
        SetDatValid)
#define TTI_UNPACR_DEST_ROW(                                                                                                                          \
    Dst_Row_Idx, Src_Row_Idx, Dst_Face_Idx, Src_Face_Idx, Dst_Tile_Offset_Idx_Inc, Src_Tile_Offset_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_DEST_ROW(                                                                                                                            \
        nullptr,                                                                                                                                      \
        Dst_Row_Idx,                                                                                                                                  \
        Src_Row_Idx,                                                                                                                                  \
        Dst_Face_Idx,                                                                                                                                 \
        Src_Face_Idx,                                                                                                                                 \
        Dst_Tile_Offset_Idx_Inc,                                                                                                                      \
        Src_Tile_Offset_Idx_Inc,                                                                                                                      \
        Buffer_Descriptor_Table_Sel,                                                                                                                  \
        SetDatValid)

#define TT_OP_UNPACR_DEST_ROW_INC(                                                                                                                             \
    Dst_Row_Idx_Inc,                                                                                                                                           \
    Src_Row_Idx_Inc,                                                                                                                                           \
    Dst_Face_Idx_Inc,                                                                                                                                          \
    Src_Face_Idx_Inc,                                                                                                                                          \
    Dst_Tile_Offset_Idx_Inc,                                                                                                                                   \
    Src_Tile_Offset_Idx_Inc,                                                                                                                                   \
    Buffer_Descriptor_Table_Sel,                                                                                                                               \
    SetDatValid)                                                                                                                                               \
    TT_OP(                                                                                                                                                     \
        0x65,                                                                                                                                                  \
        (((Dst_Row_Idx_Inc) << 20) + ((Src_Row_Idx_Inc) << 16) + ((Dst_Face_Idx_Inc) << 14) + ((Src_Face_Idx_Inc) << 12) + ((Dst_Tile_Offset_Idx_Inc) << 10) + \
         ((Src_Tile_Offset_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR_DEST_ROW_INC(       \
    IP,                                  \
    Dst_Row_Idx_Inc,                     \
    Src_Row_Idx_Inc,                     \
    Dst_Face_Idx_Inc,                    \
    Src_Face_Idx_Inc,                    \
    Dst_Tile_Offset_Idx_Inc,             \
    Src_Tile_Offset_Idx_Inc,             \
    Buffer_Descriptor_Table_Sel,         \
    SetDatValid)                         \
    TT_IP_INSN(                          \
        IP,                              \
        TT_OP_UNPACR_DEST_ROW_INC(       \
            Dst_Row_Idx_Inc,             \
            Src_Row_Idx_Inc,             \
            Dst_Face_Idx_Inc,            \
            Src_Face_Idx_Inc,            \
            Dst_Tile_Offset_Idx_Inc,     \
            Src_Tile_Offset_Idx_Inc,     \
            Buffer_Descriptor_Table_Sel, \
            SetDatValid),                \
        0,                               \
        0,                               \
        0)
#define TT_UNPACR_DEST_ROW_INC(      \
    Dst_Row_Idx_Inc,                 \
    Src_Row_Idx_Inc,                 \
    Dst_Face_Idx_Inc,                \
    Src_Face_Idx_Inc,                \
    Dst_Tile_Offset_Idx_Inc,         \
    Src_Tile_Offset_Idx_Inc,         \
    Buffer_Descriptor_Table_Sel,     \
    SetDatValid)                     \
    TT_IP_UNPACR_DEST_ROW_INC(       \
        ::ckernel::instrn_buffer,    \
        Dst_Row_Idx_Inc,             \
        Src_Row_Idx_Inc,             \
        Dst_Face_Idx_Inc,            \
        Src_Face_Idx_Inc,            \
        Dst_Tile_Offset_Idx_Inc,     \
        Src_Tile_Offset_Idx_Inc,     \
        Buffer_Descriptor_Table_Sel, \
        SetDatValid)
#define TTI_UNPACR_DEST_ROW_INC(     \
    Dst_Row_Idx_Inc,                 \
    Src_Row_Idx_Inc,                 \
    Dst_Face_Idx_Inc,                \
    Src_Face_Idx_Inc,                \
    Dst_Tile_Offset_Idx_Inc,         \
    Src_Tile_Offset_Idx_Inc,         \
    Buffer_Descriptor_Table_Sel,     \
    SetDatValid)                     \
    TT_IP_UNPACR_DEST_ROW_INC(       \
        nullptr,                     \
        Dst_Row_Idx_Inc,             \
        Src_Row_Idx_Inc,             \
        Dst_Face_Idx_Inc,            \
        Src_Face_Idx_Inc,            \
        Dst_Tile_Offset_Idx_Inc,     \
        Src_Tile_Offset_Idx_Inc,     \
        Buffer_Descriptor_Table_Sel, \
        SetDatValid)

#define TT_OP_UNPACR_DEST_STRIDE(                                                                                                                      \
    Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                             \
        0xbd,                                                                                                                                          \
        (((Src_Reg_Y_Cntr_Incr) << 20) + ((L1_Tile_Idx_or_Tile_Idx_Inc) << 17) + ((Tile_Idx_Inc) << 16) + ((Row_Mask_Reg_Sel) << 13) +                 \
         ((L1_16datums_Row_Index) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR_DEST_STRIDE(                                                                                                                          \
    IP, Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(                                                                                                                                            \
        IP,                                                                                                                                                \
        TT_OP_UNPACR_DEST_STRIDE(                                                                                                                          \
            Src_Reg_Y_Cntr_Incr,                                                                                                                           \
            L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                                   \
            Tile_Idx_Inc,                                                                                                                                  \
            Row_Mask_Reg_Sel,                                                                                                                              \
            L1_16datums_Row_Index,                                                                                                                         \
            Buffer_Descriptor_Table_Sel,                                                                                                                   \
            SetDatValid),                                                                                                                                  \
        0,                                                                                                                                                 \
        0,                                                                                                                                                 \
        0)
#define TT_UNPACR_DEST_STRIDE(                                                                                                                         \
    Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_DEST_STRIDE(                                                                                                                          \
        ::ckernel::instrn_buffer,                                                                                                                      \
        Src_Reg_Y_Cntr_Incr,                                                                                                                           \
        L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                                   \
        Tile_Idx_Inc,                                                                                                                                  \
        Row_Mask_Reg_Sel,                                                                                                                              \
        L1_16datums_Row_Index,                                                                                                                         \
        Buffer_Descriptor_Table_Sel,                                                                                                                   \
        SetDatValid)
#define TTI_UNPACR_DEST_STRIDE(                                                                                                                        \
    Src_Reg_Y_Cntr_Incr, L1_Tile_Idx_or_Tile_Idx_Inc, Tile_Idx_Inc, Row_Mask_Reg_Sel, L1_16datums_Row_Index, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_DEST_STRIDE(                                                                                                                          \
        nullptr,                                                                                                                                       \
        Src_Reg_Y_Cntr_Incr,                                                                                                                           \
        L1_Tile_Idx_or_Tile_Idx_Inc,                                                                                                                   \
        Tile_Idx_Inc,                                                                                                                                  \
        Row_Mask_Reg_Sel,                                                                                                                              \
        L1_16datums_Row_Index,                                                                                                                         \
        Buffer_Descriptor_Table_Sel,                                                                                                                   \
        SetDatValid)

#define TT_OP_UNPACR_DEST_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(0xac, (((Dst_Tile_Idx) << 15) + ((Src_Tile_Idx) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR_DEST_TILE(IP, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(IP, TT_OP_UNPACR_DEST_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid), 0, 0, 0)
#define TT_UNPACR_DEST_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_DEST_TILE(::ckernel::instrn_buffer, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR_DEST_TILE(Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_DEST_TILE(nullptr, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR_DEST_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(0xad, (((Dst_Tile_Idx_Inc) << 15) + ((Src_Tile_Idx_Inc) << 7) + ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR_DEST_TILE_INC(IP, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(IP, TT_OP_UNPACR_DEST_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid), 0, 0, 0)
#define TT_UNPACR_DEST_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_DEST_TILE_INC(::ckernel::instrn_buffer, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR_DEST_TILE_INC(Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_DEST_TILE_INC(nullptr, Dst_Tile_Idx_Inc, Src_Tile_Idx_Inc, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR_NOP(Unpacker_Select, Set_Dvalid, Stall_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Nop_type) \
    TT_OP(0x43, (((Unpacker_Select) << 8) + ((Set_Dvalid) << 7) + ((Stall_Cntrl) << 5) + ((Bank_Clr_Ctrl) << 4) + ((Src_ClrVal_Ctrl) << 2) + ((Nop_type) << 0)))
#define TT_IP_UNPACR_NOP(IP, Unpacker_Select, Set_Dvalid, Stall_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Nop_type) \
    TT_IP_INSN(IP, TT_OP_UNPACR_NOP(Unpacker_Select, Set_Dvalid, Stall_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Nop_type), 0, 0, 0)
#define TT_UNPACR_NOP(Unpacker_Select, Set_Dvalid, Stall_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Nop_type) \
    TT_IP_UNPACR_NOP(::ckernel::instrn_buffer, Unpacker_Select, Set_Dvalid, Stall_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Nop_type)
#define TTI_UNPACR_NOP(Unpacker_Select, Set_Dvalid, Stall_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Nop_type) \
    TT_IP_UNPACR_NOP(nullptr, Unpacker_Select, Set_Dvalid, Stall_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Nop_type)

#define TT_OP_UNPACR_TILE_MISC(Unpack_Type, Row_Bcast_Row_Idx, Tile_Idx_Inc, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                         \
        0xbf,                                                                                                                                      \
        (((Unpack_Type) << 21) + ((Row_Bcast_Row_Idx) << 15) + ((Tile_Idx_Inc) << 14) + ((Dst_Tile_Idx) << 12) + ((Src_Tile_Idx) << 7) +           \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR_TILE_MISC(IP, Unpack_Type, Row_Bcast_Row_Idx, Tile_Idx_Inc, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(OIP, TT_OP_UNPACR_TILE_MISC(Unpack_Type, Row_Bcast_Row_Idx, Tile_Idx_Inc, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid))
#define TT_UNPACR_TILE_MISC(Unpack_Type, Row_Bcast_Row_Idx, Tile_Idx_Inc, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_TILE_MISC(                                                                                                                     \
        ::ckernel::instrn_buffer, Unpack_Type, Row_Bcast_Row_Idx, Tile_Idx_Inc, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR_TILE_MISC(Unpack_Type, Row_Bcast_Row_Idx, Tile_Idx_Inc, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_TILE_MISC(nullptr, Unpack_Type, Row_Bcast_Row_Idx, Tile_Idx_Inc, Dst_Tile_Idx, Src_Tile_Idx, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_UNPACR_TILIZE(Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Unpack_Sel, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_OP(                                                                                                                                   \
        0xbe,                                                                                                                                \
        (((Reserved) << 15) + ((Cntr_Reset_mask) << 13) + ((Dst_Z_Cntr_inc) << 11) + ((Src_Z_Cntr_inc) << 9) + ((Unpack_Sel) << 7) +         \
         ((Buffer_Descriptor_Table_Sel) << 2) + ((SetDatValid) << 1)))
#define TT_IP_UNPACR_TILIZE(IP, Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Unpack_Sel, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_INSN(                                                                                                                                  \
        IP, TT_OP_UNPACR_TILIZE(Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Unpack_Sel, Buffer_Descriptor_Table_Sel, SetDatValid), 0, 0, 0)
#define TT_UNPACR_TILIZE(Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Unpack_Sel, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_TILIZE(                                                                                                                  \
        ::ckernel::instrn_buffer, Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Unpack_Sel, Buffer_Descriptor_Table_Sel, SetDatValid)
#define TTI_UNPACR_TILIZE(Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Unpack_Sel, Buffer_Descriptor_Table_Sel, SetDatValid) \
    TT_IP_UNPACR_TILIZE(nullptr, Reserved, Cntr_Reset_mask, Dst_Z_Cntr_inc, Src_Z_Cntr_inc, Unpack_Sel, Buffer_Descriptor_Table_Sel, SetDatValid)

#define TT_OP_WAIT_FREE(stall_res, num_tiles, buffer_sel)     TT_OP(0xab, (((stall_res) << 15) + ((num_tiles) << 5) + ((buffer_sel) << 0)))
#define TT_IP_WAIT_FREE(IP, stall_res, num_tiles, buffer_sel) TT_IP_INSN(IP, TT_OP_WAIT_FREE(stall_res, num_tiles, buffer_sel), 0, 0, 0)
#define TT_WAIT_FREE(stall_res, num_tiles, buffer_sel)        TT_IP_WAIT_FREE(::ckernel::instrn_buffer, stall_res, num_tiles, buffer_sel)
#define TTI_WAIT_FREE(stall_res, num_tiles, buffer_sel)       TT_IP_WAIT_FREE(nullptr, stall_res, num_tiles, buffer_sel)

#define TT_OP_WAIT_TILES(stall_res, num_tiles, buffer_sel)     TT_OP(0xa9, (((stall_res) << 15) + ((num_tiles) << 5) + ((buffer_sel) << 0)))
#define TT_IP_WAIT_TILES(IP, stall_res, num_tiles, buffer_sel) TT_IP_INSN(IP, TT_OP_WAIT_TILES(stall_res, num_tiles, buffer_sel), 0, 0, 0)
#define TT_WAIT_TILES(stall_res, num_tiles, buffer_sel)        TT_IP_WAIT_TILES(::ckernel::instrn_buffer, stall_res, num_tiles, buffer_sel)
#define TTI_WAIT_TILES(stall_res, num_tiles, buffer_sel)       TT_IP_WAIT_TILES(nullptr, stall_res, num_tiles, buffer_sel)

#define TT_OP_WRCFG(GprAddress, wr128b, CfgReg)     TT_OP(0xb0, (((GprAddress) << 16) + ((wr128b) << 15) + ((CfgReg) << 0)))
#define TT_IP_WRCFG(IP, GprAddress, wr128b, CfgReg) TT_IP_INSN(IP, TT_OP_WRCFG(GprAddress, wr128b, CfgReg), 0, 0, 0)
#define TT_WRCFG(GprAddress, wr128b, CfgReg)        TT_IP_WRCFG(::ckernel::instrn_buffer, GprAddress, wr128b, CfgReg)
#define TTI_WRCFG(GprAddress, wr128b, CfgReg)       TT_IP_WRCFG(nullptr, GprAddress, wr128b, CfgReg)

#define TT_OP_ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where) \
    TT_OP(0x10, (((clear_mode) << 19) + ((use_32_bit_mode) << 18) + ((clear_zero_flags) << 17) + ((addr_mode) << 14) + ((where) << 0)))
#define TT_IP_ZEROACC(IP, clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where) \
    TT_IP_INSN(IP, TT_OP_ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where), 0, 0, 0)
#define TT_ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where) \
    TT_IP_ZEROACC(::ckernel::instrn_buffer, clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where)
#define TTI_ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where) \
    TT_IP_ZEROACC(nullptr, clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where)

#define TT_OP_ZEROSRC(packed_fmt, int_fmt, exp_bias, zero_val, write_mode, bank_mask, src_mask) \
    TT_OP(0x11, (((packed_fmt) << 7) + ((int_fmt) << 6) + ((exp_bias) << 5) + ((zero_val) << 4) + ((write_mode) << 3) + ((bank_mask) << 2) + ((src_mask) << 0)))
#define TT_IP_ZEROSRC(IP, packed_fmt, int_fmt, exp_bias, zero_val, write_mode, bank_mask, src_mask) \
    TT_IP_INSN(IP, TT_OP_ZEROSRC(packed_fmt, int_fmt, exp_bias, zero_val, write_mode, bank_mask, src_mask), 0, 0, 0)
#define TT_ZEROSRC(packed_fmt, int_fmt, exp_bias, zero_val, write_mode, bank_mask, src_mask) \
    TT_IP_ZEROSRC(::ckernel::instrn_buffer, packed_fmt, int_fmt, exp_bias, zero_val, write_mode, bank_mask, src_mask)
#define TTI_ZEROSRC(packed_fmt, int_fmt, exp_bias, zero_val, write_mode, bank_mask, src_mask) \
    TT_IP_ZEROSRC(nullptr, packed_fmt, int_fmt, exp_bias, zero_val, write_mode, bank_mask, src_mask)
