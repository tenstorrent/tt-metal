// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if __riscv_xtttensixbh || (__clang__ && defined(ARCH_BLACKHOLE) && defined(COMPILE_FOR_TRISC))
#if __clang__
// User is likely using static analysis tool.  Provide a declaration.
extern "C" void __builtin_rvtt_ttinsn(volatile unsigned long *, unsigned, unsigned, unsigned, unsigned) __attribute__((nothrow));
#endif
#define TT_IP_INSN(IP, ENC, MOD, DST, SRC) __builtin_rvtt_ttinsn((IP), (ENC), MOD, DST, SRC)
#elif defined(ARCH_BLACKHOLE) && defined(LLK_BOOT_BRISC)
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

#define TT_OP_ADDDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_OP(0x58, (((OpBisConst) << 23) + ((ResultRegIndex) << 12) + ((OpBRegIndex) << 6) + ((OpARegIndex) << 0)))
#define TT_ADDDMAREG_VALID(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    (ckernel::is_valid(OpBisConst, 1) && ckernel::is_valid(ResultRegIndex, 11) && ckernel::is_valid(OpBRegIndex, 6) && ckernel::is_valid(OpARegIndex, 6))
#define TT_IP_ADDDMAREG(IP, OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_INSN(IP, TT_OP_ADDDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex), 0, 0, 0)
#define TT_ADDDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_ADDDMAREG(::ckernel::instrn_buffer, OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex)
#define TTI_ADDDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) TT_IP_ADDDMAREG(nullptr, OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex)

#define TT_OP_ADDRCRXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) \
    TT_OP(0x53, (((CntSetMask) << 21) + ((Ch1_Y) << 15) + ((Ch1_X) << 12) + ((Ch0_Y) << 9) + ((Ch0_X) << 6) + ((BitMask) << 0)))
#define TT_ADDRCRXY_VALID(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)                                                            \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(Ch1_Y, 6) && ckernel::is_valid(Ch1_X, 3) && ckernel::is_valid(Ch0_Y, 3) && \
     ckernel::is_valid(Ch0_X, 3) && ckernel::is_valid(BitMask, 6))
#define TT_IP_ADDRCRXY(IP, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) \
    TT_IP_INSN(IP, TT_OP_ADDRCRXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask), 0, 0, 0)
#define TT_ADDRCRXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)  TT_IP_ADDRCRXY(::ckernel::instrn_buffer, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)
#define TTI_ADDRCRXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) TT_IP_ADDRCRXY(nullptr, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)

#define TT_OP_ADDRCRZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) \
    TT_OP(0x56, (((CntSetMask) << 21) + ((Ch1_Y) << 15) + ((Ch1_X) << 12) + ((Ch0_Y) << 9) + ((Ch0_X) << 6) + ((BitMask) << 0)))
#define TT_ADDRCRZW_VALID(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)                                                            \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(Ch1_Y, 6) && ckernel::is_valid(Ch1_X, 3) && ckernel::is_valid(Ch0_Y, 3) && \
     ckernel::is_valid(Ch0_X, 3) && ckernel::is_valid(BitMask, 6))
#define TT_IP_ADDRCRZW(IP, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) \
    TT_IP_INSN(IP, TT_OP_ADDRCRZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask), 0, 0, 0)
#define TT_ADDRCRZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)  TT_IP_ADDRCRZW(::ckernel::instrn_buffer, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)
#define TTI_ADDRCRZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) TT_IP_ADDRCRZW(nullptr, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)

#define TT_OP_APOOL3S1(clear_dvalid, pool_addr_mode, index_en, dst) \
    TT_OP(0x25, (((clear_dvalid) << 22) + ((pool_addr_mode) << 15) + ((index_en) << 14) + ((dst) << 0)))
#define TT_APOOL3S1_VALID(clear_dvalid, pool_addr_mode, index_en, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(pool_addr_mode, 7) && ckernel::is_valid(index_en, 1) && ckernel::is_valid(dst, 14))
#define TT_IP_APOOL3S1(IP, clear_dvalid, pool_addr_mode, index_en, dst) TT_IP_INSN(IP, TT_OP_APOOL3S1(clear_dvalid, pool_addr_mode, index_en, dst), 0, 0, 0)
#define TT_APOOL3S1(clear_dvalid, pool_addr_mode, index_en, dst)        TT_IP_APOOL3S1(::ckernel::instrn_buffer, clear_dvalid, pool_addr_mode, index_en, dst)
#define TTI_APOOL3S1(clear_dvalid, pool_addr_mode, index_en, dst)       TT_IP_APOOL3S1(nullptr, clear_dvalid, pool_addr_mode, index_en, dst)

#define TT_OP_APOOL3S2(clear_dvalid, pool_addr_mode, index_en, dst) \
    TT_OP(0x32, (((clear_dvalid) << 22) + ((pool_addr_mode) << 15) + ((index_en) << 14) + ((dst) << 0)))
#define TT_APOOL3S2_VALID(clear_dvalid, pool_addr_mode, index_en, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(pool_addr_mode, 7) && ckernel::is_valid(index_en, 1) && ckernel::is_valid(dst, 14))
#define TT_IP_APOOL3S2(IP, clear_dvalid, pool_addr_mode, index_en, dst) TT_IP_INSN(IP, TT_OP_APOOL3S2(clear_dvalid, pool_addr_mode, index_en, dst), 0, 0, 0)
#define TT_APOOL3S2(clear_dvalid, pool_addr_mode, index_en, dst)        TT_IP_APOOL3S2(::ckernel::instrn_buffer, clear_dvalid, pool_addr_mode, index_en, dst)
#define TTI_APOOL3S2(clear_dvalid, pool_addr_mode, index_en, dst)       TT_IP_APOOL3S2(nullptr, clear_dvalid, pool_addr_mode, index_en, dst)

#define TT_OP_ATCAS(MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_OP(0x64, (((MemHierSel) << 23) + ((SwapVal) << 18) + ((CmpVal) << 14) + ((Sel32b) << 12) + ((DataRegIndex) << 6) + ((AddrRegIndex) << 0)))
#define TT_ATCAS_VALID(MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex)                                                   \
    (ckernel::is_valid(MemHierSel, 1) && ckernel::is_valid(SwapVal, 5) && ckernel::is_valid(CmpVal, 4) && ckernel::is_valid(Sel32b, 2) && \
     ckernel::is_valid(DataRegIndex, 6) && ckernel::is_valid(AddrRegIndex, 6))
#define TT_IP_ATCAS(IP, MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_IP_INSN(IP, TT_OP_ATCAS(MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex), 0, 0, 0)
#define TT_ATCAS(MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_IP_ATCAS(::ckernel::instrn_buffer, MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex)
#define TTI_ATCAS(MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_IP_ATCAS(nullptr, MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex)

#define TT_OP_ATGETM(mutex_index)     TT_OP(0xa0, (((mutex_index) << 0)))
#define TT_ATGETM_VALID(mutex_index)  (ckernel::is_valid(mutex_index, 24))
#define TT_IP_ATGETM(IP, mutex_index) TT_IP_INSN(IP, TT_OP_ATGETM(mutex_index), 0, 0, 0)
#define TT_ATGETM(mutex_index)        TT_IP_ATGETM(::ckernel::instrn_buffer, mutex_index)
#define TTI_ATGETM(mutex_index)       TT_IP_ATGETM(nullptr, mutex_index)

#define TT_OP_ATINCGET(MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_OP(0x61, (((MemHierSel) << 23) + ((WrapVal) << 14) + ((Sel32b) << 12) + ((DataRegIndex) << 6) + ((AddrRegIndex) << 0)))
#define TT_ATINCGET_VALID(MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex)                                                              \
    (ckernel::is_valid(MemHierSel, 1) && ckernel::is_valid(WrapVal, 9) && ckernel::is_valid(Sel32b, 2) && ckernel::is_valid(DataRegIndex, 6) && \
     ckernel::is_valid(AddrRegIndex, 6))
#define TT_IP_ATINCGET(IP, MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_IP_INSN(IP, TT_IP_ATINCGET(::ckernel::instrn_buffer, MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex), 0, 0, 0)
#define TT_ATINCGET(MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_IP_ATINCGET(::ckernel::instrn_buffer, MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex)
#define TTI_ATINCGET(MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) TT_IP_ATINCGET(nullptr, MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex)

#define TT_OP_ATINCGETPTR(MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_OP(                                                                                          \
        0x62,                                                                                       \
        (((MemHierSel) << 23) + ((NoIncr) << 22) + ((IncrVal) << 18) + ((WrapVal) << 14) + ((Sel32b) << 12) + ((DataRegIndex) << 6) + ((AddrRegIndex) << 0)))
#define TT_ATINCGETPTR_VALID(MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex)                                     \
    (ckernel::is_valid(MemHierSel, 1) && ckernel::is_valid(NoIncr, 1) && ckernel::is_valid(IncrVal, 4) && ckernel::is_valid(WrapVal, 4) && \
     ckernel::is_valid(Sel32b, 2) && ckernel::is_valid(DataRegIndex, 6) && ckernel::is_valid(AddrRegIndex, 6))
#define TT_IP_ATINCGETPTR(IP, MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_IP_INSN(IP, TT_OP_ATINCGETPTR(MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex), 0, 0, 0)
#define TT_ATINCGETPTR(MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_IP_ATINCGETPTR(::ckernel::instrn_buffer, MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex)
#define TTI_ATINCGETPTR(MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_IP_ATINCGETPTR(nullptr, MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex)

#define TT_OP_ATRELM(mutex_index)     TT_OP(0xa1, (((mutex_index) << 0)))
#define TT_ATRELM_VALID(mutex_index)  (ckernel::is_valid(mutex_index, 24))
#define TT_IP_ATRELM(IP, mutex_index) TT_IP_INSN(IP, TT_OP_ATRELM(mutex_index), 0, 0, 0)
#define TT_ATRELM(mutex_index)        TT_IP_ATRELM(::ckernel::instrn_buffer, mutex_index)
#define TTI_ATRELM(mutex_index)       TT_IP_ATRELM(nullptr, mutex_index)

#define TT_OP_ATSWAP(MemHierSel, SwapMask, DataRegIndex, AddrRegIndex) \
    TT_OP(0x63, (((MemHierSel) << 23) + ((SwapMask) << 14) + ((DataRegIndex) << 6) + ((AddrRegIndex) << 0)))
#define TT_ATSWAP_VALID(MemHierSel, SwapMask, DataRegIndex, AddrRegIndex) \
    (ckernel::is_valid(MemHierSel, 1) && ckernel::is_valid(SwapMask, 9) && ckernel::is_valid(DataRegIndex, 8) && ckernel::is_valid(AddrRegIndex, 6))
#define TT_IP_ATSWAP(IP, MemHierSel, SwapMask, DataRegIndex, AddrRegIndex) \
    TT_IP_INSN(IP, TT_OP_ATSWAP(MemHierSel, SwapMask, DataRegIndex, AddrRegIndex), 0, 0, 0)
#define TT_ATSWAP(MemHierSel, SwapMask, DataRegIndex, AddrRegIndex)  TT_IP_ATSWAP(::ckernel::instrn_buffer, MemHierSel, SwapMask, DataRegIndex, AddrRegIndex)
#define TTI_ATSWAP(MemHierSel, SwapMask, DataRegIndex, AddrRegIndex) TT_IP_ATSWAP(nullptr, MemHierSel, SwapMask, DataRegIndex, AddrRegIndex)

#define TT_OP_BITWOPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_OP(0x5b, (((OpBisConst) << 23) + ((OpSel) << 18) + ((ResultRegIndex) << 12) + ((OpBRegIndex) << 6) + ((OpARegIndex) << 0)))
#define TT_BITWOPDMAREG_VALID(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)                                                           \
    (ckernel::is_valid(OpBisConst, 1) && ckernel::is_valid(OpSel, 5) && ckernel::is_valid(ResultRegIndex, 6) && ckernel::is_valid(OpBRegIndex, 6) && \
     ckernel::is_valid(OpARegIndex, 6))
#define TT_IP_BITWOPDMAREG(IP, OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_INSN(IP, TT_OP_BITWOPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex), 0, 0, 0)
#define TT_BITWOPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_BITWOPDMAREG(::ckernel::instrn_buffer, OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)
#define TTI_BITWOPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_BITWOPDMAREG(nullptr, OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)

#define TT_OP_CFGSHIFTMASK(disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel, CfgReg)                             \
    TT_OP(                                                                                                                                    \
        0xb8,                                                                                                                                 \
        (((disable_mask_on_old_val) << 23) + ((operation) << 20) + ((mask_width) << 15) + ((right_cshift_amt) << 10) + ((scratch_sel) << 8) + \
         ((CfgReg) << 0)))
#define TT_CFGSHIFTMASK_VALID(disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel, CfgReg)         \
    (ckernel::is_valid(disable_mask_on_old_val, 1) && ckernel::is_valid(operation, 3) && ckernel::is_valid(mask_width, 5) && \
     ckernel::is_valid(right_cshift_amt, 5) && ckernel::is_valid(scratch_sel, 2) && ckernel::is_valid(CfgReg, 8))
#define TT_IP_CFGSHIFTMASK(IP, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel, CfgReg) \
    TT_IP_INSN(IP, TT_OP_CFGSHIFTMASK(disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel, CfgReg), 0, 0, 0)
#define TT_CFGSHIFTMASK(disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel, CfgReg) \
    TT_IP_CFGSHIFTMASK(::ckernel::instrn_buffer, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel, CfgReg)
#define TTI_CFGSHIFTMASK(disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel, CfgReg) \
    TT_IP_CFGSHIFTMASK(nullptr, disable_mask_on_old_val, operation, mask_width, right_cshift_amt, scratch_sel, CfgReg)

#define TT_OP_CLEARDVALID(cleardvalid, reset)     TT_OP(0x36, (((cleardvalid) << 22) + ((reset) << 0)))
#define TT_CLEARDVALID_VALID(cleardvalid, reset)  (ckernel::is_valid(cleardvalid, 2) && ckernel::is_valid(reset, 22))
#define TT_IP_CLEARDVALID(IP, cleardvalid, reset) TT_IP_INSN(IP, TT_OP_CLEARDVALID(cleardvalid, reset), 0, 0, 0)
#define TT_CLEARDVALID(cleardvalid, reset)        TT_IP_CLEARDVALID(::ckernel::instrn_buffer, cleardvalid, reset)
#define TTI_CLEARDVALID(cleardvalid, reset)       TT_IP_CLEARDVALID(nullptr, cleardvalid, reset)

#define TT_OP_CLREXPHIST     TT_OP(0x21, 0)
#define TT_IP_CLREXPHIST(IP) TT_IP_INSN(IP, TT_OP_CLREXPHIST, 0, 0, 0)
#define TTI_CLREXPHIST       TT_IP_CLREXPHIST(nullptr)

#define TT_OP_CMPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_OP(0x5d, (((OpBisConst) << 23) + ((OpSel) << 18) + ((ResultRegIndex) << 12) + ((OpBRegIndex) << 6) + ((OpARegIndex) << 0)))
#define TT_CMPDMAREG_VALID(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)                                                              \
    (ckernel::is_valid(OpBisConst, 1) && ckernel::is_valid(OpSel, 5) && ckernel::is_valid(ResultRegIndex, 6) && ckernel::is_valid(OpBRegIndex, 6) && \
     ckernel::is_valid(OpARegIndex, 6))
#define TT_IP_CMPDMAREG(IP, OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_INSN(IP, TT_OP_CMPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex), 0, 0, 0)
#define TT_CMPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_CMPDMAREG(::ckernel::instrn_buffer, OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)
#define TTI_CMPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_CMPDMAREG(nullptr, OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)

#define TT_OP_CONV3S1(clear_dvalid, rotate_weights, addr_mode, dst) \
    TT_OP(0x22, (((clear_dvalid) << 22) + ((rotate_weights) << 17) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_CONV3S1_VALID(clear_dvalid, rotate_weights, addr_mode, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(rotate_weights, 5) && ckernel::is_valid(addr_mode, 3) && ckernel::is_valid(dst, 14))
#define TT_IP_CONV3S1(IP, clear_dvalid, rotate_weights, addr_mode, dst) TT_IP_INSN(IP, TT_OP_CONV3S1(clear_dvalid, rotate_weights, addr_mode, dst), 0, 0, 0)
#define TT_CONV3S1(clear_dvalid, rotate_weights, addr_mode, dst)        TT_IP_CONV3S1(::ckernel::instrn_buffer, clear_dvalid, rotate_weights, addr_mode, dst)
#define TTI_CONV3S1(clear_dvalid, rotate_weights, addr_mode, dst)       TT_IP_CONV3S1(nullptr, clear_dvalid, rotate_weights, addr_mode, dst)

#define TT_OP_CONV3S2(clear_dvalid, rotate_weights, addr_mode, dst) \
    TT_OP(0x23, (((clear_dvalid) << 22) + ((rotate_weights) << 17) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_CONV3S2_VALID(clear_dvalid, rotate_weights, addr_mode, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(rotate_weights, 5) && ckernel::is_valid(addr_mode, 3) && ckernel::is_valid(dst, 14))
#define TT_IP_CONV3S2(IP, clear_dvalid, rotate_weights, addr_mode, dst) TT_IP_INSN(IP, TT_OP_CONV3S2(clear_dvalid, rotate_weights, addr_mode, dst), 0, 0, 0)
#define TT_CONV3S2(clear_dvalid, rotate_weights, addr_mode, dst)        TT_IP_CONV3S2(::ckernel::instrn_buffer, clear_dvalid, rotate_weights, addr_mode, dst)
#define TTI_CONV3S2(clear_dvalid, rotate_weights, addr_mode, dst)       TT_IP_CONV3S2(nullptr, clear_dvalid, rotate_weights, addr_mode, dst)

#define TT_OP_DMANOP     TT_OP(0x60, 0)
#define TT_IP_DMANOP(IP) TT_IP_INSN(IP, TT_OP_DMANOP, 0, 0, 0)
#define TTI_DMANOP       TT_IP_DMANOP(nullptr)

#define TT_OP_DOTPV(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_OP(0x29, (((clear_dvalid) << 22) + ((dest_accum_en) << 21) + ((instr_mod19) << 19) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_DOTPV_VALID(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)                                                                          \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(dest_accum_en, 1) && ckernel::is_valid(instr_mod19, 2) && ckernel::is_valid(addr_mode, 5) && \
     ckernel::is_valid(dst, 14))
#define TT_IP_DOTPV(IP, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_INSN(IP, TT_OP_DOTPV(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst), 0, 0, 0)
#define TT_DOTPV(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_DOTPV(::ckernel::instrn_buffer, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)
#define TTI_DOTPV(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) TT_IP_DOTPV(nullptr, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)

#define TT_OP_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_OP(0x28, (((clear_dvalid) << 22) + ((dest_accum_en) << 21) + ((instr_mod19) << 19) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_ELWADD_VALID(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)                                                                         \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(dest_accum_en, 1) && ckernel::is_valid(instr_mod19, 2) && ckernel::is_valid(addr_mode, 5) && \
     ckernel::is_valid(dst, 14))
#define TT_IP_ELWADD(IP, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_INSN(IP, TT_OP_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst), 0, 0, 0)
#define TT_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_ELWADD(::ckernel::instrn_buffer, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)
#define TTI_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) TT_IP_ELWADD(nullptr, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)

#define TT_OP_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_OP(0x27, (((clear_dvalid) << 22) + ((dest_accum_en) << 21) + ((instr_mod19) << 19) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_ELWMUL_VALID(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)                                                                         \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(dest_accum_en, 1) && ckernel::is_valid(instr_mod19, 2) && ckernel::is_valid(addr_mode, 5) && \
     ckernel::is_valid(dst, 14))
#define TT_IP_ELWMUL(IP, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_INSN(IP, TT_OP_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst), 0, 0, 0)
#define TT_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_ELWMUL(::ckernel::instrn_buffer, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)
#define TTI_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) TT_IP_ELWMUL(nullptr, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)

#define TT_OP_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_OP(0x30, (((clear_dvalid) << 22) + ((dest_accum_en) << 21) + ((instr_mod19) << 19) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_ELWSUB_VALID(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)                                                                         \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(dest_accum_en, 1) && ckernel::is_valid(instr_mod19, 2) && ckernel::is_valid(addr_mode, 5) && \
     ckernel::is_valid(dst, 14))
#define TT_IP_ELWSUB(IP, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_INSN(IP, TT_OP_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst), 0, 0, 0)
#define TT_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_IP_ELWSUB(::ckernel::instrn_buffer, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)
#define TTI_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) TT_IP_ELWSUB(nullptr, clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)

#define TT_OP_FLUSHDMA(FlushSpec)     TT_OP(0x46, (((FlushSpec) << 0)))
#define TT_FLUSHDMA_VALID(FlushSpec)  (ckernel::is_valid(FlushSpec, 24))
#define TT_IP_FLUSHDMA(IP, FlushSpec) TT_IP_INSN(IP, TT_OP_FLUSHDMA(FlushSpec), 0, 0, 0)
#define TT_FLUSHDMA(FlushSpec)        TT_IP_FLUSHDMA(::ckernel::instrn_buffer, FlushSpec)
#define TTI_FLUSHDMA(FlushSpec)       TT_IP_FLUSHDMA(nullptr, FlushSpec)

#define TT_OP_GAPOOL(clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst) \
    TT_OP(0x34, (((clear_dvalid) << 22) + ((instr_mod19) << 19) + ((pool_addr_mode) << 15) + ((max_pool_index_en) << 14) + ((dst) << 0)))
#define TT_GAPOOL_VALID(clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst)                              \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(instr_mod19, 3) && ckernel::is_valid(pool_addr_mode, 4) && \
     ckernel::is_valid(max_pool_index_en, 1) && ckernel::is_valid(dst, 14))
#define TT_IP_GAPOOL(IP, clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst) \
    TT_IP_INSN(IP, TT_OP_GAPOOL(clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst), 0, 0, 0)
#define TT_GAPOOL(clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst) \
    TT_IP_GAPOOL(::ckernel::instrn_buffer, clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst)
#define TTI_GAPOOL(clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst) \
    TT_IP_GAPOOL(nullptr, clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst)

#define TT_OP_GATESRCRST(reset_srcb_gate_control, reset_srca_gate_control) TT_OP(0x35, (((reset_srcb_gate_control) << 1) + ((reset_srca_gate_control) << 0)))
#define TT_GATESRCRST_VALID(reset_srcb_gate_control, reset_srca_gate_control) \
    (ckernel::is_valid(reset_srcb_gate_control, 23) && ckernel::is_valid(reset_srca_gate_control, 1))
#define TT_IP_GATESRCRST(IP, reset_srcb_gate_control, reset_srca_gate_control) \
    TT_IP_INSN(IP, TT_OP_GATESRCRST(reset_srcb_gate_control, reset_srca_gate_control), 0, 0, 0)
#define TT_GATESRCRST(reset_srcb_gate_control, reset_srca_gate_control) \
    TT_IP_GATESRCRST(::ckernel::instrn_buffer, reset_srcb_gate_control, reset_srca_gate_control)
#define TTI_GATESRCRST(reset_srcb_gate_control, reset_srca_gate_control) TT_IP_GATESRCRST(nullptr, reset_srcb_gate_control, reset_srca_gate_control)

#define TT_OP_GMPOOL(clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst) \
    TT_OP(0x33, (((clear_dvalid) << 22) + ((instr_mod19) << 19) + ((pool_addr_mode) << 15) + ((max_pool_index_en) << 14) + ((dst) << 0)))
#define TT_GMPOOL_VALID(clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst)                              \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(instr_mod19, 3) && ckernel::is_valid(pool_addr_mode, 4) && \
     ckernel::is_valid(max_pool_index_en, 1) && ckernel::is_valid(dst, 14))
#define TT_IP_GMPOOL(IP, clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst) \
    TT_IP_INSN(IP, TT_OP_GMPOOL(clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst), 0, 0, 0)
#define TT_GMPOOL(clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst) \
    TT_IP_GMPOOL(::ckernel::instrn_buffer, clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst)
#define TTI_GMPOOL(clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst) \
    TT_IP_GMPOOL(nullptr, clear_dvalid, instr_mod19, pool_addr_mode, max_pool_index_en, dst)

#define TT_OP_INCADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X) \
    TT_OP(0x52, (((CntSetMask) << 21) + ((Ch1_Y) << 15) + ((Ch1_X) << 12) + ((Ch0_Y) << 9) + ((Ch0_X) << 6)))
#define TT_INCADCXY_VALID(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)                                                                     \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(Ch1_Y, 6) && ckernel::is_valid(Ch1_X, 3) && ckernel::is_valid(Ch0_Y, 3) && \
     ckernel::is_valid(Ch0_X, 3))
#define TT_IP_INCADCXY(IP, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X) TT_IP_INSN(IP, TT_OP_INCADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X), 0, 0, 0)
#define TT_INCADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)        TT_IP_INCADCXY(::ckernel::instrn_buffer, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)
#define TTI_INCADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)       TT_IP_INCADCXY(nullptr, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)

#define TT_OP_INCADCZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X) \
    TT_OP(0x55, (((CntSetMask) << 21) + ((Ch1_Y) << 15) + ((Ch1_X) << 12) + ((Ch0_Y) << 9) + ((Ch0_X) << 6)))
#define TT_INCADCZW_VALID(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)                                                                     \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(Ch1_Y, 6) && ckernel::is_valid(Ch1_X, 3) && ckernel::is_valid(Ch0_Y, 3) && \
     ckernel::is_valid(Ch0_X, 3))
#define TT_IP_INCADCZW(IP, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X) TT_IP_INSN(IP, TT_OP_INCADCZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X), 0, 0, 0)
#define TT_INCADCZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)        TT_IP_INCADCZW(::ckernel::instrn_buffer, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)
#define TTI_INCADCZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)       TT_IP_INCADCZW(nullptr, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)

#define TT_OP_INCRWC(rwc_cr, rwc_d, rwc_b, rwc_a) TT_OP(0x38, (((rwc_cr) << 18) + ((rwc_d) << 14) + ((rwc_b) << 10) + ((rwc_a) << 6)))
#define TT_INCRWC_VALID(rwc_cr, rwc_d, rwc_b, rwc_a) \
    (ckernel::is_valid(rwc_cr, 6) && ckernel::is_valid(rwc_d, 4) && ckernel::is_valid(rwc_b, 4) && ckernel::is_valid(rwc_a, 4))
#define TT_IP_INCRWC(IP, rwc_cr, rwc_d, rwc_b, rwc_a) TT_IP_INSN(IP, TT_OP_INCRWC(rwc_cr, rwc_d, rwc_b, rwc_a), 0, 0, 0)
#define TT_INCRWC(rwc_cr, rwc_d, rwc_b, rwc_a)        TT_IP_INCRWC(::ckernel::instrn_buffer, rwc_cr, rwc_d, rwc_b, rwc_a)
#define TTI_INCRWC(rwc_cr, rwc_d, rwc_b, rwc_a)       TT_IP_INCRWC(nullptr, rwc_cr, rwc_d, rwc_b, rwc_a)

#define TT_OP_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex) \
    TT_OP(0x49, (((SizeSel) << 22) + ((OffsetIndex) << 14) + ((AutoIncSpec) << 12) + ((DataRegIndex) << 6) + ((AddrRegIndex) << 0)))
#define TT_LOADIND_VALID(SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex)                                                               \
    (ckernel::is_valid(SizeSel, 2) && ckernel::is_valid(OffsetIndex, 8) && ckernel::is_valid(AutoIncSpec, 2) && ckernel::is_valid(DataRegIndex, 6) && \
     ckernel::is_valid(AddrRegIndex, 6))
#define TT_IP_LOADIND(IP, SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex) \
    TT_IP_INSN(IP, TT_OP_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex), 0, 0, 0)
#define TT_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex) \
    TT_IP_LOADIND(::ckernel::instrn_buffer, SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex)
#define TTI_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex) \
    TT_IP_LOADIND(nullptr, SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex)

#define TT_OP_LOADREG(TdmaDataRegIndex, RegAddr)     TT_OP(0x68, (((TdmaDataRegIndex) << 18) + ((RegAddr) << 0)))
#define TT_LOADREG_VALID(TdmaDataRegIndex, RegAddr)  (ckernel::is_valid(TdmaDataRegIndex, 6) && ckernel::is_valid(RegAddr, 18))
#define TT_IP_LOADREG(IP, TdmaDataRegIndex, RegAddr) TT_IP_INSN(IP, TT_OP_LOADREG(TdmaDataRegIndex, RegAddr), 0, 0, 0)
#define TT_LOADREG(TdmaDataRegIndex, RegAddr)        TT_IP_LOADREG(::ckernel::instrn_buffer, TdmaDataRegIndex, RegAddr)
#define TTI_LOADREG(TdmaDataRegIndex, RegAddr)       TT_IP_LOADREG(nullptr, TdmaDataRegIndex, RegAddr)

#define TT_OP_MFCONV3S1(clear_dvalid, rotate_weights, addr_mode, dst) \
    TT_OP(0x3a, (((clear_dvalid) << 22) + ((rotate_weights) << 17) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_MFCONV3S1_VALID(clear_dvalid, rotate_weights, addr_mode, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(rotate_weights, 5) && ckernel::is_valid(addr_mode, 3) && ckernel::is_valid(dst, 14))
#define TT_IP_MFCONV3S1(IP, clear_dvalid, rotate_weights, addr_mode, dst) TT_IP_INSN(IP, TT_OP_MFCONV3S1(clear_dvalid, rotate_weights, addr_mode, dst), 0, 0, 0)
#define TT_MFCONV3S1(clear_dvalid, rotate_weights, addr_mode, dst)        TT_IP_MFCONV3S1(::ckernel::instrn_buffer, clear_dvalid, rotate_weights, addr_mode, dst)
#define TTI_MFCONV3S1(clear_dvalid, rotate_weights, addr_mode, dst)       TT_IP_MFCONV3S1(nullptr, clear_dvalid, rotate_weights, addr_mode, dst)

#define TT_OP_MOP(mop_type, loop_count, zmask_lo16_or_loop_count) TT_OP(0x01, (((mop_type) << 23) + ((loop_count) << 16) + ((zmask_lo16_or_loop_count) << 0)))
#define TT_MOP_VALID(mop_type, loop_count, zmask_lo16_or_loop_count) \
    (ckernel::is_valid(mop_type, 1) && ckernel::is_valid(loop_count, 7) && ckernel::is_valid(zmask_lo16_or_loop_count, 16))
#define TT_IP_MOP(IP, mop_type, loop_count, zmask_lo16_or_loop_count) TT_IP_INSN(IP, TT_OP_MOP(mop_type, loop_count, zmask_lo16_or_loop_count), 0, 0, 0)
#define TT_MOP(mop_type, loop_count, zmask_lo16_or_loop_count)        TT_IP_MOP(::ckernel::instrn_buffer, mop_type, loop_count, zmask_lo16_or_loop_count)
#define TTI_MOP(mop_type, loop_count, zmask_lo16_or_loop_count)       TT_IP_MOP(nullptr, mop_type, loop_count, zmask_lo16_or_loop_count)

#define TT_OP_MOP_CFG(zmask_hi16)     TT_OP(0x03, (((zmask_hi16) << 0)))
#define TT_MOP_CFG_VALID(zmask_hi16)  (ckernel::is_valid(zmask_hi16, 24))
#define TT_IP_MOP_CFG(IP, zmask_hi16) TT_IP_INSN(IP, TT_OP_MOP_CFG(zmask_hi16), 0, 0, 0)
#define TT_MOP_CFG(zmask_hi16)        TT_IP_MOP_CFG(::ckernel::instrn_buffer, zmask_hi16)
#define TTI_MOP_CFG(zmask_hi16)       TT_IP_MOP_CFG(nullptr, zmask_hi16)

#define TT_OP_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst) \
    TT_OP(0x12, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 14) + ((instr_mod) << 12) + ((dst) << 0)))
#define TT_MOVA2D_VALID(dest_32b_lo, src, addr_mode, instr_mod, dst)                                                                         \
    (ckernel::is_valid(dest_32b_lo, 1) && ckernel::is_valid(src, 6) && ckernel::is_valid(addr_mode, 3) && ckernel::is_valid(instr_mod, 2) && \
     ckernel::is_valid(dst, 12))
#define TT_IP_MOVA2D(IP, dest_32b_lo, src, addr_mode, instr_mod, dst) TT_IP_INSN(IP, TT_OP_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst), 0, 0, 0)
#define TT_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst)        TT_IP_MOVA2D(::ckernel::instrn_buffer, dest_32b_lo, src, addr_mode, instr_mod, dst)
#define TTI_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst)       TT_IP_MOVA2D(nullptr, dest_32b_lo, src, addr_mode, instr_mod, dst)

#define TT_OP_MOVB2A(srca, addr_mode, instr_mod, srcb) TT_OP(0x0b, (((srca) << 17) + ((addr_mode) << 14) + ((instr_mod) << 12) + ((srcb) << 0)))
#define TT_MOVB2A_VALID(srca, addr_mode, instr_mod, srcb) \
    (ckernel::is_valid(srca, 7) && ckernel::is_valid(addr_mode, 3) && ckernel::is_valid(instr_mod, 2) && ckernel::is_valid(srcb, 12))
#define TT_IP_MOVB2A(IP, srca, addr_mode, instr_mod, srcb) TT_IP_INSN(IP, TT_OP_MOVB2A(srca, addr_mode, instr_mod, srcb), 0, 0, 0)
#define TT_MOVB2A(srca, addr_mode, instr_mod, srcb)        TT_IP_MOVB2A(::ckernel::instrn_buffer, srca, addr_mode, instr_mod, srcb)
#define TTI_MOVB2A(srca, addr_mode, instr_mod, srcb)       TT_IP_MOVB2A(nullptr, srca, addr_mode, instr_mod, srcb)

#define TT_OP_MOVB2D(dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst) \
    TT_OP(0x13, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 14) + ((movb2d_instr_mod) << 11) + ((dst) << 0)))
#define TT_MOVB2D_VALID(dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst)                                                                         \
    (ckernel::is_valid(dest_32b_lo, 1) && ckernel::is_valid(src, 6) && ckernel::is_valid(addr_mode, 3) && ckernel::is_valid(movb2d_instr_mod, 3) && \
     ckernel::is_valid(dst, 11))
#define TT_IP_MOVB2D(IP, dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst) \
    TT_IP_INSN(IP, TT_OP_MOVB2D(dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst), 0, 0, 0)
#define TT_MOVB2D(dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst)  TT_IP_MOVB2D(::ckernel::instrn_buffer, dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst)
#define TTI_MOVB2D(dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst) TT_IP_MOVB2D(nullptr, dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst)

#define TT_OP_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst) \
    TT_OP(0x08, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 14) + ((instr_mod) << 12) + ((dst) << 0)))
#define TT_MOVD2A_VALID(dest_32b_lo, src, addr_mode, instr_mod, dst)                                                                         \
    (ckernel::is_valid(dest_32b_lo, 1) && ckernel::is_valid(src, 6) && ckernel::is_valid(addr_mode, 3) && ckernel::is_valid(instr_mod, 2) && \
     ckernel::is_valid(dst, 12))
#define TT_IP_MOVD2A(IP, dest_32b_lo, src, addr_mode, instr_mod, dst) TT_IP_INSN(IP, TT_OP_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst), 0, 0, 0)
#define TT_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst)        TT_IP_MOVD2A(::ckernel::instrn_buffer, dest_32b_lo, src, addr_mode, instr_mod, dst)
#define TTI_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst)       TT_IP_MOVD2A(nullptr, dest_32b_lo, src, addr_mode, instr_mod, dst)

#define TT_OP_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, dst) \
    TT_OP(0x0a, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 14) + ((instr_mod) << 12) + ((dst) << 0)))
#define TT_MOVD2B_VALID(dest_32b_lo, src, addr_mode, instr_mod, dst)                                                                         \
    (ckernel::is_valid(dest_32b_lo, 1) && ckernel::is_valid(src, 6) && ckernel::is_valid(addr_mode, 3) && ckernel::is_valid(instr_mod, 2) && \
     ckernel::is_valid(dst, 12))
#define TT_IP_MOVD2B(IP, dest_32b_lo, src, addr_mode, instr_mod, dst) TT_IP_INSN(IP, TT_OP_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, dst), 0, 0, 0)
#define TT_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, dst)        TT_IP_MOVD2B(::ckernel::instrn_buffer, dest_32b_lo, src, addr_mode, instr_mod, dst)
#define TTI_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, dst)       TT_IP_MOVD2B(nullptr, dest_32b_lo, src, addr_mode, instr_mod, dst)

#define TT_OP_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst) \
    TT_OP(0x09, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 14) + ((instr_mod) << 12) + ((dst) << 0)))
#define TT_MOVDBGA2D_VALID(dest_32b_lo, src, addr_mode, instr_mod, dst)                                                                      \
    (ckernel::is_valid(dest_32b_lo, 1) && ckernel::is_valid(src, 6) && ckernel::is_valid(addr_mode, 3) && ckernel::is_valid(instr_mod, 2) && \
     ckernel::is_valid(dst, 12))
#define TT_IP_MOVDBGA2D(IP, dest_32b_lo, src, addr_mode, instr_mod, dst) TT_IP_INSN(IP, TT_OP_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst), 0, 0, 0)
#define TT_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst)        TT_IP_MOVDBGA2D(::ckernel::instrn_buffer, dest_32b_lo, src, addr_mode, instr_mod, dst)
#define TTI_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst)       TT_IP_MOVDBGA2D(nullptr, dest_32b_lo, src, addr_mode, instr_mod, dst)

#define TT_OP_MOVDBGB2D(dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst) \
    TT_OP(0x0c, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 14) + ((movb2d_instr_mod) << 11) + ((dst) << 0)))
#define TT_MOVDBGB2D_VALID(dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst)                                                                      \
    (ckernel::is_valid(dest_32b_lo, 1) && ckernel::is_valid(src, 6) && ckernel::is_valid(addr_mode, 3) && ckernel::is_valid(movb2d_instr_mod, 3) && \
     ckernel::is_valid(dst, 11))
#define TT_IP_MOVDBGB2D(IP, dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst) \
    TT_IP_INSN(IP, TT_OP_MOVDBGB2D(dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst), 0, 0, 0)
#define TT_MOVDBGB2D(dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst) \
    TT_IP_MOVDBGB2D(::ckernel::instrn_buffer, dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst)
#define TTI_MOVDBGB2D(dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst) TT_IP_MOVDBGB2D(nullptr, dest_32b_lo, src, addr_mode, movb2d_instr_mod, dst)

#define TT_OP_MPOOL3S1(clear_dvalid, pool_addr_mode, index_en, dst) \
    TT_OP(0x24, (((clear_dvalid) << 22) + ((pool_addr_mode) << 15) + ((index_en) << 14) + ((dst) << 0)))
#define TT_MPOOL3S1_VALID(clear_dvalid, pool_addr_mode, index_en, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(pool_addr_mode, 7) && ckernel::is_valid(index_en, 1) && ckernel::is_valid(dst, 14))
#define TT_IP_MPOOL3S1(IP, clear_dvalid, pool_addr_mode, index_en, dst) TT_IP_INSN(IP, TT_OP_MPOOL3S1(clear_dvalid, pool_addr_mode, index_en, dst), 0, 0, 0)
#define TT_MPOOL3S1(clear_dvalid, pool_addr_mode, index_en, dst)        TT_IP_MPOOL3S1(::ckernel::instrn_buffer, clear_dvalid, pool_addr_mode, index_en, dst)
#define TTI_MPOOL3S1(clear_dvalid, pool_addr_mode, index_en, dst)       TT_IP_MPOOL3S1(nullptr, clear_dvalid, pool_addr_mode, index_en, dst)

#define TT_OP_MPOOL3S2(clear_dvalid, pool_addr_mode, index_en, dst) \
    TT_OP(0x31, (((clear_dvalid) << 22) + ((pool_addr_mode) << 15) + ((index_en) << 14) + ((dst) << 0)))
#define TT_MPOOL3S2_VALID(clear_dvalid, pool_addr_mode, index_en, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(pool_addr_mode, 7) && ckernel::is_valid(index_en, 1) && ckernel::is_valid(dst, 14))
#define TT_IP_MPOOL3S2(IP, clear_dvalid, pool_addr_mode, index_en, dst) TT_IP_INSN(IP, TT_OP_MPOOL3S2(clear_dvalid, pool_addr_mode, index_en, dst), 0, 0, 0)
#define TT_MPOOL3S2(clear_dvalid, pool_addr_mode, index_en, dst)        TT_IP_MPOOL3S2(::ckernel::instrn_buffer, clear_dvalid, pool_addr_mode, index_en, dst)
#define TTI_MPOOL3S2(clear_dvalid, pool_addr_mode, index_en, dst)       TT_IP_MPOOL3S2(nullptr, clear_dvalid, pool_addr_mode, index_en, dst)

#define TT_OP_MULDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_OP(0x5a, (((OpBisConst) << 23) + ((ResultRegIndex) << 12) + ((OpBRegIndex) << 6) + ((OpARegIndex) << 0)))
#define TT_MULDMAREG_VALID(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    (ckernel::is_valid(OpBisConst, 1) && ckernel::is_valid(ResultRegIndex, 11) && ckernel::is_valid(OpBRegIndex, 6) && ckernel::is_valid(OpARegIndex, 6))
#define TT_IP_MULDMAREG(IP, OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_INSN(IP, TT_OP_MULDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex), 0, 0, 0)
#define TT_MULDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_MULDMAREG(::ckernel::instrn_buffer, OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex)
#define TTI_MULDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) TT_IP_MULDMAREG(nullptr, OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex)

#define TT_OP_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst) \
    TT_OP(0x26, (((clear_dvalid) << 22) + ((instr_mod19) << 19) + ((addr_mode) << 14) + ((dst) << 0)))
#define TT_MVMUL_VALID(clear_dvalid, instr_mod19, addr_mode, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(instr_mod19, 3) && ckernel::is_valid(addr_mode, 5) && ckernel::is_valid(dst, 14))
#define TT_IP_MVMUL(IP, clear_dvalid, instr_mod19, addr_mode, dst) TT_IP_INSN(IP, TT_OP_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst), 0, 0, 0)
#define TT_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst)        TT_IP_MVMUL(::ckernel::instrn_buffer, clear_dvalid, instr_mod19, addr_mode, dst)
#define TTI_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst)       TT_IP_MVMUL(nullptr, clear_dvalid, instr_mod19, addr_mode, dst)

#define TT_OP_NOP     TT_OP(0x02, 0)
#define TT_IP_NOP(IP) TT_IP_INSN(IP, TT_OP_NOP, 0, 0, 0)
#define TTI_NOP       TT_IP_NOP(nullptr)

#define TT_OP_PACR(CfgContext, RowPadZero, DstAccessMode, AddrMode, AddrCntContext, ZeroWrite, ReadIntfSel, OvrdThreadId, Concat, CtxtCtrl, Flush, Last) \
    TT_OP(                                                                                                                                               \
        0x41,                                                                                                                                            \
        (((CfgContext) << 21) + ((RowPadZero) << 18) + ((DstAccessMode) << 17) + ((AddrMode) << 15) + ((AddrCntContext) << 13) + ((ZeroWrite) << 12) +   \
         ((ReadIntfSel) << 8) + ((OvrdThreadId) << 7) + ((Concat) << 4) + ((CtxtCtrl) << 2) + ((Flush) << 1) + ((Last) << 0)))
#define TT_PACR_VALID(CfgContext, RowPadZero, DstAccessMode, AddrMode, AddrCntContext, ZeroWrite, ReadIntfSel, OvrdThreadId, Concat, CtxtCtrl, Flush, Last) \
    (ckernel::is_valid(CfgContext, 3) && ckernel::is_valid(RowPadZero, 3) && ckernel::is_valid(DstAccessMode, 1) && ckernel::is_valid(AddrMode, 2) &&       \
     ckernel::is_valid(AddrCntContext, 2) && ckernel::is_valid(ZeroWrite, 1) && ckernel::is_valid(ReadIntfSel, 4) && ckernel::is_valid(OvrdThreadId, 1) &&  \
     ckernel::is_valid(Concat, 3) && ckernel::is_valid(CtxtCtrl, 2) && ckernel::is_valid(Flush, 1) && ckernel::is_valid(Last, 1))
#define TT_IP_PACR(IP, CfgContext, RowPadZero, DstAccessMode, AddrMode, AddrCntContext, ZeroWrite, ReadIntfSel, OvrdThreadId, Concat, CtxtCtrl, Flush, Last) \
    TT_IP_INSN(                                                                                                                                              \
        IP,                                                                                                                                                  \
        TT_OP_PACR(CfgContext, RowPadZero, DstAccessMode, AddrMode, AddrCntContext, ZeroWrite, ReadIntfSel, OvrdThreadId, Concat, CtxtCtrl, Flush, Last),    \
        0,                                                                                                                                                   \
        0,                                                                                                                                                   \
        0)
#define TT_PACR(CfgContext, RowPadZero, DstAccessMode, AddrMode, AddrCntContext, ZeroWrite, ReadIntfSel, OvrdThreadId, Concat, CtxtCtrl, Flush, Last) \
    TT_IP_PACR(                                                                                                                                       \
        ::ckernel::instrn_buffer,                                                                                                                     \
        CfgContext,                                                                                                                                   \
        RowPadZero,                                                                                                                                   \
        DstAccessMode,                                                                                                                                \
        AddrMode,                                                                                                                                     \
        AddrCntContext,                                                                                                                               \
        ZeroWrite,                                                                                                                                    \
        ReadIntfSel,                                                                                                                                  \
        OvrdThreadId,                                                                                                                                 \
        Concat,                                                                                                                                       \
        CtxtCtrl,                                                                                                                                     \
        Flush,                                                                                                                                        \
        Last)
#define TTI_PACR(CfgContext, RowPadZero, DstAccessMode, AddrMode, AddrCntContext, ZeroWrite, ReadIntfSel, OvrdThreadId, Concat, CtxtCtrl, Flush, Last) \
    TT_IP_PACR(nullptr, CfgContext, RowPadZero, DstAccessMode, AddrMode, AddrCntContext, ZeroWrite, ReadIntfSel, OvrdThreadId, Concat, CtxtCtrl, Flush, Last)

#define TT_OP_PACR_SETREG(Push, ModeSel, Unused, DisableStall, AddrSel, StreamId, Flush, Last)                                                    \
    TT_OP(                                                                                                                                        \
        0x4a,                                                                                                                                     \
        (((Push) << 23) + ((ModeSel) << 22) + ((Unused) << 12) + ((DisableStall) << 10) + ((AddrSel) << 8) + ((StreamId) << 2) + ((Flush) << 1) + \
         ((Last) << 0)))
#define TT_PACR_SETREG_VALID(Push, ModeSel, Unused, DisableStall, AddrSel, StreamId, Flush, Last)                                          \
    (ckernel::is_valid(Push, 1) && ckernel::is_valid(ModeSel, 1) && ckernel::is_valid(Unused, 10) && ckernel::is_valid(DisableStall, 2) && \
     ckernel::is_valid(AddrSel, 2) && ckernel::is_valid(StreamId, 6) && ckernel::is_valid(Flush, 1) && ckernel::is_valid(Last, 1))
#define TT_IP_PACR_SETREG(IP, Push, ModeSel, Unused, DisableStall, AddrSel, StreamId, Flush, Last) \
    TT_IP_INSN(IP, TT_OP_PACR_SETREG(Push, ModeSel, Unused, DisableStall, AddrSel, StreamId, Flush, Last), 0, 0, 0)
#define TT_PACR_SETREG(Push, ModeSel, Unused, DisableStall, AddrSel, StreamId, Flush, Last) \
    TT_IP_PACR_SETREG(::ckernel::instrn_buffer, Push, ModeSel, Unused, DisableStall, AddrSel, StreamId, Flush, Last)
#define TTI_PACR_SETREG(Push, ModeSel, Unused, DisableStall, AddrSel, StreamId, Flush, Last) \
    TT_IP_PACR_SETREG(nullptr, Push, ModeSel, Unused, DisableStall, AddrSel, StreamId, Flush, Last)

#define TT_OP_RAREB     TT_OP(0x15, 0)
#define TT_IP_RAREB(IP) TT_IP_INSN(IP, TT_OP_RAREB, 0, 0, 0)
#define TTI_RAREB       TT_IP_RAREB(nullptr)

#define TT_OP_RDCFG(GprAddress, CfgReg)     TT_OP(0xb1, (((GprAddress) << 16) + ((CfgReg) << 0)))
#define TT_RDCFG_VALID(GprAddress, CfgReg)  (ckernel::is_valid(GprAddress, 8) && ckernel::is_valid(CfgReg, 16))
#define TT_IP_RDCFG(IP, GprAddress, CfgReg) TT_IP_INSN(IP, TT_OP_RDCFG(GprAddress, CfgReg), 0, 0, 0)
#define TT_RDCFG(GprAddress, CfgReg)        TT_IP_RDCFG(::ckernel::instrn_buffer, GprAddress, CfgReg)
#define TTI_RDCFG(GprAddress, CfgReg)       TT_IP_RDCFG(nullptr, GprAddress, CfgReg)

#define TT_OP_REG2FLOP(SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex) \
    TT_OP(0x48, (((SizeSel) << 22) + ((TargetSel) << 20) + ((ByteOffset) << 18) + ((ContextId_2) << 16) + ((FlopIndex) << 6) + ((RegIndex) << 0)))
#define TT_REG2FLOP_VALID(SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex)                                                       \
    (ckernel::is_valid(SizeSel, 2) && ckernel::is_valid(TargetSel, 2) && ckernel::is_valid(ByteOffset, 2) && ckernel::is_valid(ContextId_2, 2) && \
     ckernel::is_valid(FlopIndex, 10) && ckernel::is_valid(RegIndex, 6))
#define TT_IP_REG2FLOP(IP, SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex) \
    TT_IP_INSN(IP, TT_OP_REG2FLOP(SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex), 0, 0, 0)
#define TT_REG2FLOP(SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex) \
    TT_IP_REG2FLOP(::ckernel::instrn_buffer, SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex)
#define TTI_REG2FLOP(SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex) \
    TT_IP_REG2FLOP(nullptr, SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex)

#define TT_OP_REPLAY(start_idx, len, execute_while_loading, load_mode) \
    TT_OP(0x04, (((start_idx) << 14) + ((len) << 4) + ((execute_while_loading) << 1) + ((load_mode) << 0)))
#define TT_REPLAY_VALID(start_idx, len, execute_while_loading, load_mode) \
    (ckernel::is_valid(start_idx, 10) && ckernel::is_valid(len, 10) && ckernel::is_valid(execute_while_loading, 3) && ckernel::is_valid(load_mode, 1))
#define TT_IP_REPLAY(IP, start_idx, len, execute_while_loading, load_mode) \
    TT_IP_INSN(IP, TT_OP_REPLAY(start_idx, len, execute_while_loading, load_mode), 0, 0, 0)
#define TT_REPLAY(start_idx, len, execute_while_loading, load_mode)  TT_IP_REPLAY(::ckernel::instrn_buffer, start_idx, len, execute_while_loading, load_mode)
#define TTI_REPLAY(start_idx, len, execute_while_loading, load_mode) TT_IP_REPLAY(nullptr, start_idx, len, execute_while_loading, load_mode)

#define TT_OP_RESOURCEDECL(linger_time, resources, op_class) TT_OP(0x05, (((linger_time) << 13) + ((resources) << 4) + ((op_class) << 0)))
#define TT_RESOURCEDECL_VALID(linger_time, resources, op_class) \
    (ckernel::is_valid(linger_time, 11) && ckernel::is_valid(resources, 9) && ckernel::is_valid(op_class, 4))
#define TT_IP_RESOURCEDECL(IP, linger_time, resources, op_class) TT_IP_INSN(IP, TT_OP_RESOURCEDECL(linger_time, resources, op_class), 0, 0, 0)
#define TT_RESOURCEDECL(linger_time, resources, op_class)        TT_IP_RESOURCEDECL(::ckernel::instrn_buffer, linger_time, resources, op_class)
#define TTI_RESOURCEDECL(linger_time, resources, op_class)       TT_IP_RESOURCEDECL(nullptr, linger_time, resources, op_class)

#define TT_OP_RMWCIB0(Mask, Data, CfgRegAddr)     TT_OP(0xb3, (((Mask) << 16) + ((Data) << 8) + ((CfgRegAddr) << 0)))
#define TT_RMWCIB0_VALID(Mask, Data, CfgRegAddr)  (ckernel::is_valid(Mask, 8) && ckernel::is_valid(Data, 8) && ckernel::is_valid(CfgRegAddr, 8))
#define TT_IP_RMWCIB0(IP, Mask, Data, CfgRegAddr) TT_IP_INSN(IP, TT_OP_RMWCIB0(Mask, Data, CfgRegAddr), 0, 0, 0)
#define TT_RMWCIB0(Mask, Data, CfgRegAddr)        TT_IP_RMWCIB0(::ckernel::instrn_buffer, Mask, Data, CfgRegAddr)
#define TTI_RMWCIB0(Mask, Data, CfgRegAddr)       TT_IP_RMWCIB0(nullptr, Mask, Data, CfgRegAddr)

#define TT_OP_RMWCIB1(Mask, Data, CfgRegAddr)     TT_OP(0xb4, (((Mask) << 16) + ((Data) << 8) + ((CfgRegAddr) << 0)))
#define TT_RMWCIB1_VALID(Mask, Data, CfgRegAddr)  (ckernel::is_valid(Mask, 8) && ckernel::is_valid(Data, 8) && ckernel::is_valid(CfgRegAddr, 8))
#define TT_IP_RMWCIB1(IP, Mask, Data, CfgRegAddr) TT_IP_INSN(IP, TT_OP_RMWCIB1(Mask, Data, CfgRegAddr), 0, 0, 0)
#define TT_RMWCIB1(Mask, Data, CfgRegAddr)        TT_IP_RMWCIB1(::ckernel::instrn_buffer, Mask, Data, CfgRegAddr)
#define TTI_RMWCIB1(Mask, Data, CfgRegAddr)       TT_IP_RMWCIB1(nullptr, Mask, Data, CfgRegAddr)

#define TT_OP_RMWCIB2(Mask, Data, CfgRegAddr)     TT_OP(0xb5, (((Mask) << 16) + ((Data) << 8) + ((CfgRegAddr) << 0)))
#define TT_RMWCIB2_VALID(Mask, Data, CfgRegAddr)  (ckernel::is_valid(Mask, 8) && ckernel::is_valid(Data, 8) && ckernel::is_valid(CfgRegAddr, 8))
#define TT_IP_RMWCIB2(IP, Mask, Data, CfgRegAddr) TT_IP_INSN(IP, TT_OP_RMWCIB2(Mask, Data, CfgRegAddr), 0, 0, 0)
#define TT_RMWCIB2(Mask, Data, CfgRegAddr)        TT_IP_RMWCIB2(::ckernel::instrn_buffer, Mask, Data, CfgRegAddr)
#define TTI_RMWCIB2(Mask, Data, CfgRegAddr)       TT_IP_RMWCIB2(nullptr, Mask, Data, CfgRegAddr)

#define TT_OP_RMWCIB3(Mask, Data, CfgRegAddr)     TT_OP(0xb6, (((Mask) << 16) + ((Data) << 8) + ((CfgRegAddr) << 0)))
#define TT_RMWCIB3_VALID(Mask, Data, CfgRegAddr)  (ckernel::is_valid(Mask, 8) && ckernel::is_valid(Data, 8) && ckernel::is_valid(CfgRegAddr, 8))
#define TT_IP_RMWCIB3(IP, Mask, Data, CfgRegAddr) TT_IP_INSN(IP, TT_OP_RMWCIB3(Mask, Data, CfgRegAddr), 0, 0, 0)
#define TT_RMWCIB3(Mask, Data, CfgRegAddr)        TT_IP_RMWCIB3(::ckernel::instrn_buffer, Mask, Data, CfgRegAddr)
#define TTI_RMWCIB3(Mask, Data, CfgRegAddr)       TT_IP_RMWCIB3(nullptr, Mask, Data, CfgRegAddr)

#define TT_OP_RSTDMA     TT_OP(0x44, 0)
#define TT_IP_RSTDMA(IP) TT_IP_INSN(IP, TT_OP_RSTDMA, 0, 0, 0)
#define TTI_RSTDMA       TT_P_RSTDMA(nullptr)

#define TT_OP_SEMGET(sem_sel)     TT_OP(0xa5, (((sem_sel) << 2)))
#define TT_SEMGET_VALID(sem_sel)  (ckernel::is_valid(sem_sel, 22))
#define TT_IP_SEMGET(IP, sem_sel) TT_IP_INSN(IP, TT_OP_SEMGET(sem_sel), 0, 0, 0)
#define TT_SEMGET(sem_sel)        TT_IP_SEMGET(::ckernel::instrn_buffer, sem_sel)
#define TTI_SEMGET(sem_sel)       TT_IP_SEMGET(nullptr, sem_sel)

#define TT_OP_SEMINIT(max_value, init_value, sem_sel)     TT_OP(0xa3, (((max_value) << 20) + ((init_value) << 16) + ((sem_sel) << 2)))
#define TT_SEMINIT_VALID(max_value, init_value, sem_sel)  (ckernel::is_valid(max_value, 4) && ckernel::is_valid(init_value, 4) && ckernel::is_valid(sem_sel, 14))
#define TT_IP_SEMINIT(IP, max_value, init_value, sem_sel) TT_IP_INSN(IP, TT_OP_SEMINIT(max_value, init_value, sem_sel), 0, 0, 0)
#define TT_SEMINIT(max_value, init_value, sem_sel)        TT_IP_SEMINIT(::ckernel::instrn_buffer, max_value, init_value, sem_sel)
#define TTI_SEMINIT(max_value, init_value, sem_sel)       TT_IP_SEMINIT(nullptr, max_value, init_value, sem_sel)

#define TT_OP_SEMPOST(sem_sel)     TT_OP(0xa4, (((sem_sel) << 2)))
#define TT_SEMPOST_VALID(sem_sel)  (ckernel::is_valid(sem_sel, 22))
#define TT_IP_SEMPOST(IP, sem_sel) TT_IP_INSN(IP, TT_OP_SEMPOST(sem_sel), 0, 0, 0)
#define TT_SEMPOST(sem_sel)        TT_IP_SEMPOST(::ckernel::instrn_buffer, sem_sel)
#define TTI_SEMPOST(sem_sel)       TT_IP_SEMPOST(nullptr, sem_sel)

#define TT_OP_SEMWAIT(stall_res, sem_sel, wait_sem_cond) TT_OP(0xa6, (((stall_res) << 15) + ((sem_sel) << 2) + ((wait_sem_cond) << 0)))
#define TT_SEMWAIT_VALID(stall_res, sem_sel, wait_sem_cond) \
    (ckernel::is_valid(stall_res, 9) && ckernel::is_valid(sem_sel, 13) && ckernel::is_valid(wait_sem_cond, 2))
#define TT_IP_SEMWAIT(IP, stall_res, sem_sel, wait_sem_cond) TT_IP_INSN(IP, TT_OP_SEMWAIT(stall_res, sem_sel, wait_sem_cond), 0, 0, 0)
#define TT_SEMWAIT(stall_res, sem_sel, wait_sem_cond)        TT_IP_SEMWAIT(::ckernel::instrn_buffer, stall_res, sem_sel, wait_sem_cond)
#define TTI_SEMWAIT(stall_res, sem_sel, wait_sem_cond)       TT_IP_SEMWAIT(nullptr, stall_res, sem_sel, wait_sem_cond)

#define TT_OP_SETADC(CntSetMask, ChannelIndex, DimensionIndex, Value) \
    TT_OP(0x50, (((CntSetMask) << 21) + ((ChannelIndex) << 20) + ((DimensionIndex) << 18) + ((Value) << 0)))
#define TT_SETADC_VALID(CntSetMask, ChannelIndex, DimensionIndex, Value) \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(ChannelIndex, 1) && ckernel::is_valid(DimensionIndex, 2) && ckernel::is_valid(Value, 18))
#define TT_IP_SETADC(IP, CntSetMask, ChannelIndex, DimensionIndex, Value) TT_IP_INSN(IP, TT_OP_SETADC(CntSetMask, ChannelIndex, DimensionIndex, Value), 0, 0, 0)
#define TT_SETADC(CntSetMask, ChannelIndex, DimensionIndex, Value)        TT_IP_SETADC(::ckernel::instrn_buffer, CntSetMask, ChannelIndex, DimensionIndex, Value)
#define TTI_SETADC(CntSetMask, ChannelIndex, DimensionIndex, Value)       TT_IP_SETADC(nullptr, CntSetMask, ChannelIndex, DimensionIndex, Value)

#define TT_OP_SETADCXX(CntSetMask, x_end2, x_start)     TT_OP(0x5e, (((CntSetMask) << 21) + ((x_end2) << 10) + ((x_start) << 0)))
#define TT_SETADCXX_VALID(CntSetMask, x_end2, x_start)  (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(x_end2, 11) && ckernel::is_valid(x_start, 10))
#define TT_IP_SETADCXX(IP, CntSetMask, x_end2, x_start) TT_IP_INSN(IP, TT_OP_SETADCXX(CntSetMask, x_end2, x_start), 0, 0, 0)
#define TT_SETADCXX(CntSetMask, x_end2, x_start)        TT_IP_SETADCXX(::ckernel::instrn_buffer, CntSetMask, x_end2, x_start)
#define TTI_SETADCXX(CntSetMask, x_end2, x_start)       TT_IP_SETADCXX(nullptr, CntSetMask, x_end2, x_start)

#define TT_OP_SETADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) \
    TT_OP(0x51, (((CntSetMask) << 21) + ((Ch1_Y) << 15) + ((Ch1_X) << 12) + ((Ch0_Y) << 9) + ((Ch0_X) << 6) + ((BitMask) << 0)))
#define TT_SETADCXY_VALID(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)                                                            \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(Ch1_Y, 6) && ckernel::is_valid(Ch1_X, 3) && ckernel::is_valid(Ch0_Y, 3) && \
     ckernel::is_valid(Ch0_X, 3) && ckernel::is_valid(BitMask, 6))
#define TT_IP_SETADCXY(IP, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) \
    TT_IP_INSN(IP, TT_OP_SETADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask), 0, 0, 0)
#define TT_SETADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)  TT_IP_SETADCXY(::ckernel::instrn_buffer, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)
#define TTI_SETADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) TT_IP_SETADCXY(nullptr, CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)

#define TT_OP_SETADCZW(CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask) \
    TT_OP(0x54, (((CntSetMask) << 21) + ((Ch1_W) << 15) + ((Ch1_Z) << 12) + ((Ch0_W) << 9) + ((Ch0_Z) << 6) + ((BitMask) << 0)))
#define TT_SETADCZW_VALID(CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask)                                                            \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(Ch1_W, 6) && ckernel::is_valid(Ch1_Z, 3) && ckernel::is_valid(Ch0_W, 3) && \
     ckernel::is_valid(Ch0_Z, 3) && ckernel::is_valid(BitMask, 6))
#define TT_IP_SETADCZW(IP, CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask) \
    TT_IP_INSN(IP, TT_OP_SETADCZW(CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask), 0, 0, 0)
#define TT_SETADCZW(CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask)  TT_IP_SETADCZW(::ckernel::instrn_buffer, CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask)
#define TTI_SETADCZW(CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask) TT_IP_SETADCZW(nullptr, CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask)

#define TT_OP_SETASHRMH(reg_mask, halo_mask)     TT_OP(0x1e, (((reg_mask) << 1) + ((halo_mask) << 0)))
#define TT_SETASHRMH_VALID(reg_mask, halo_mask)  (ckernel::is_valid(reg_mask, 23) && ckernel::is_valid(halo_mask, 1))
#define TT_IP_SETASHRMH(IP, reg_mask, halo_mask) TT_IP_INSN(IP, TT_OP_SETASHRMH(reg_mask, halo_mask), 0, 0, 0)
#define TT_SETASHRMH(reg_mask, halo_mask)        TT_IP_SETASHRMH(::ckernel::instrn_buffer, reg_mask, halo_mask)
#define TTI_SETASHRMH(reg_mask, halo_mask)       TT_IP_SETASHRMH(nullptr, reg_mask, halo_mask)

#define TT_OP_SETASHRMH0(reg_mask, halo_mask)     TT_OP(0x1a, (((reg_mask) << 1) + ((halo_mask) << 0)))
#define TT_SETASHRMH0_VALID(reg_mask, halo_mask)  (ckernel::is_valid(reg_mask, 23) && ckernel::is_valid(halo_mask, 1))
#define TT_IP_SETASHRMH0(IP, reg_mask, halo_mask) TT_IP_INSN(IP, TT_OP_SETASHRMH0(reg_mask, halo_mask), 0, 0, 0)
#define TT_SETASHRMH0(reg_mask, halo_mask)        TT_IP_SETASHRMH0(::ckernel::instrn_buffer, reg_mask, halo_mask)
#define TTI_SETASHRMH0(reg_mask, halo_mask)       TT_IP_SETASHRMH0(nullptr, reg_mask, halo_mask)

#define TT_OP_SETASHRMH1(reg_mask, halo_mask)     TT_OP(0x1b, (((reg_mask) << 1) + ((halo_mask) << 0)))
#define TT_SETASHRMH1_VALID(reg_mask, halo_mask)  (ckernel::is_valid(reg_mask, 23) && ckernel::is_valid(halo_mask, 1))
#define TT_IP_SETASHRMH1(IP, reg_mask, halo_mask) TT_IP_INSN(IP, TT_OP_SETASHRMH1(reg_mask, halo_mask), 0, 0, 0)
#define TT_SETASHRMH1(reg_mask, halo_mask)        TT_IP_SETASHRMH1(::ckernel::instrn_buffer, reg_mask, halo_mask)
#define TTI_SETASHRMH1(reg_mask, halo_mask)       TT_IP_SETASHRMH1(nullptr, reg_mask, halo_mask)

#define TT_OP_SETASHRMV(reg_mask2)     TT_OP(0x1c, (((reg_mask2) << 0)))
#define TT_SETASHRMV_VALID(reg_mask2)  (ckernel::is_valid(reg_mask2, 24))
#define TT_IP_SETASHRMV(IP, reg_mask2) TT_IP_INSN(IP, TT_OP_SETASHRMV(reg_mask2), 0, 0, 0)
#define TT_SETASHRMV(reg_mask2)        TT_IP_SETASHRMV(::ckernel::instrn_buffer, reg_mask2)
#define TTI_SETASHRMV(reg_mask2)       TT_IP_SETASHRMV(nullptr, reg_mask2)

#define TT_OP_SETC16(setc16_reg, setc16_value)     TT_OP(0xb2, (((setc16_reg) << 16) + ((setc16_value) << 0)))
#define TT_SETC16_VALID(setc16_reg, setc16_value)  (ckernel::is_valid(setc16_reg, 8) && ckernel::is_valid(setc16_value, 16))
#define TT_IP_SETC16(IP, setc16_reg, setc16_value) TT_IP_INSN(IP, TT_OP_SETC16(setc16_reg, setc16_value), 0, 0, 0)
#define TT_SETC16(setc16_reg, setc16_value)        TT_IP_SETC16(::ckernel::instrn_buffer, setc16_reg, setc16_value)
#define TTI_SETC16(setc16_reg, setc16_value)       TT_IP_SETC16(nullptr, setc16_reg, setc16_value)

#define TT_OP_SETDMAREG(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b) \
    TT_OP(0x45, (((Payload_SigSelSize) << 22) + ((Payload_SigSel) << 8) + ((SetSignalsMode) << 7) + ((RegIndex16b) << 0)))
#define TT_SETDMAREG_VALID(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b)                                       \
    (ckernel::is_valid(Payload_SigSelSize, 2) && ckernel::is_valid(Payload_SigSel, 14) && ckernel::is_valid(SetSignalsMode, 1) && \
     ckernel::is_valid(RegIndex16b, 7))
#define TT_IP_SETDMAREG(IP, Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b) \
    TT_IP_INSN(IP, TT_OP_SETDMAREG(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b), 0, 0, 0)
#define TT_SETDMAREG(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b) \
    TT_IP_SETDMAREG(::ckernel::instrn_buffer, Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b)
#define TTI_SETDMAREG(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b) \
    TT_IP_SETDMAREG(nullptr, Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b)

#define TT_OP_SETDVALID(setvalid)     TT_OP(0x57, (((setvalid) << 0)))
#define TT_SETDVALID_VALID(setvalid)  (ckernel::is_valid(setvalid, 24))
#define TT_IP_SETDVALID(IP, setvalid) TT_IP_INSN(IP, TT_OP_SETDVALID(setvalid), 0, 0, 0)
#define TT_SETDVALID(setvalid)        TT_IP_SETDVALID(::ckernel::instrn_buffer, setvalid)
#define TTI_SETDVALID(setvalid)       TT_IP_SETDVALID(nullptr, setvalid)

#define TT_OP_SETIBRWC(rwc_cr, rwc_bias, set_inc_ctrl) TT_OP(0x39, (((rwc_cr) << 18) + ((rwc_bias) << 6) + ((set_inc_ctrl) << 0)))
#define TT_SETIBRWC_VALID(rwc_cr, rwc_bias, set_inc_ctrl) \
    (ckernel::is_valid(rwc_cr, 6) && ckernel::is_valid(rwc_bias, 12) && ckernel::is_valid(set_inc_ctrl, 6))
#define TT_IP_SETIBRWC(IP, rwc_cr, rwc_bias, set_inc_ctrl) TT_IP_INSN(IP, TT_OP_SETIBRWC(rwc_cr, rwc_bias, set_inc_ctrl), 0, 0, 0)
#define TT_SETIBRWC(rwc_cr, rwc_bias, set_inc_ctrl)        TT_IP_SETIBRWC(::ckernel::instrn_buffer, rwc_cr, rwc_bias, set_inc_ctrl)
#define TTI_SETIBRWC(rwc_cr, rwc_bias, set_inc_ctrl)       TT_IP_SETIBRWC(nullptr, rwc_cr, rwc_bias, set_inc_ctrl)

#define TT_OP_SETPKEDGOF(y_end, y_start, x_end, x_start) TT_OP(0x1d, (((y_end) << 12) + ((y_start) << 8) + ((x_end) << 4) + ((x_start) << 0)))
#define TT_SETPKEDGOF_VALID(y_end, y_start, x_end, x_start) \
    (ckernel::is_valid(y_end, 12) && ckernel::is_valid(y_start, 4) && ckernel::is_valid(x_end, 4) && ckernel::is_valid(x_start, 4))
#define TT_IP_SETPKEDGOF(IP, y_end, y_start, x_end, x_start) TT_IP_INSN(IP, TT_OP_SETPKEDGOF(y_end, y_start, x_end, x_start), 0, 0, 0)
#define TT_SETPKEDGOF(y_end, y_start, x_end, x_start)        TT_IP_SETPKEDGOF(::ckernel::instrn_buffer, y_end, y_start, x_end, x_start)
#define TTI_SETPKEDGOF(y_end, y_start, x_end, x_start)       TT_IP_SETPKEDGOF(nullptr, y_end, y_start, x_end, x_start)

#define TT_OP_SETRWC(clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask) \
    TT_OP(0x37, (((clear_ab_vld) << 22) + ((rwc_cr) << 18) + ((rwc_d) << 14) + ((rwc_b) << 10) + ((rwc_a) << 6) + ((BitMask) << 0)))
#define TT_SETRWC_VALID(clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask)                                                              \
    (ckernel::is_valid(clear_ab_vld, 2) && ckernel::is_valid(rwc_cr, 4) && ckernel::is_valid(rwc_d, 4) && ckernel::is_valid(rwc_b, 4) && \
     ckernel::is_valid(rwc_a, 4) && ckernel::is_valid(BitMask, 6))
#define TT_IP_SETRWC(IP, clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask) \
    TT_IP_INSN(IP, TT_OP_SETRWC(clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask), 0, 0, 0)
#define TT_SETRWC(clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask)  TT_IP_SETRWC(::ckernel::instrn_buffer, clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask)
#define TTI_SETRWC(clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask) TT_IP_SETRWC(nullptr, clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask)

#define TT_OP_SFPABS(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x7d, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPABS_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPABS(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPABS(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPABS(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPABS(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPABS(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPABS(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPADD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x85, (((lreg_src_a) << 16) + ((lreg_src_b) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPADD_VALID(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)                                                                  \
    (ckernel::is_valid(lreg_src_a, 8) && ckernel::is_valid(lreg_src_b, 4) && ckernel::is_valid(lreg_src_c, 4) && ckernel::is_valid(lreg_dest, 4) && \
     ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPADD(IP, lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_IP_INSN(IP, TT_OP_SFPADD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPADD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_IP_SFPADD(::ckernel::instrn_buffer, lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPADD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) TT_IP_SFPADD(nullptr, lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)

#define TT_OP_SFPADDI(imm16_math, lreg_dest, instr_mod1) TT_OP(0x75, (((imm16_math) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPADDI_VALID(imm16_math, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm16_math, 16) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPADDI(IP, imm16_math, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPADDI(imm16_math, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPADDI(imm16_math, lreg_dest, instr_mod1)        TT_IP_SFPADDI(::ckernel::instrn_buffer, imm16_math, lreg_dest, instr_mod1)
#define TTI_SFPADDI(imm16_math, lreg_dest, instr_mod1)       TT_IP_SFPADDI(nullptr, imm16_math, lreg_dest, instr_mod1)

#define TT_OP_SFPAND(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x7e, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPAND_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPAND(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPAND(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPAND(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPAND(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPAND(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPAND(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPARECIP(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x99, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPARECIP_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPARECIP(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPARECIP(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPARECIP(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPARECIP(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPARECIP(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPARECIP(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPCAST(lreg_src_c, lreg_dest, instr_mod1) TT_OP(0x90, (((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPCAST_VALID(lreg_src_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(lreg_src_c, 16) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPCAST(IP, lreg_src_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPCAST(lreg_src_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPCAST(lreg_src_c, lreg_dest, instr_mod1)        TT_IP_SFPCAST(::ckernel::instrn_buffer, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPCAST(lreg_src_c, lreg_dest, instr_mod1)       TT_IP_SFPCAST(nullptr, lreg_src_c, lreg_dest, instr_mod1)

#define TT_OP_SFPCOMPC(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x8b, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPCOMPC_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPCOMPC(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPCOMPC(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPCOMPC(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPCOMPC(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPCOMPC(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPCOMPC(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPCONFIG(imm16_math, config_dest, instr_mod1) TT_OP(0x91, (((imm16_math) << 8) + ((config_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPCONFIG_VALID(imm16_math, config_dest, instr_mod1) \
    (ckernel::is_valid(imm16_math, 16) && ckernel::is_valid(config_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPCONFIG(IP, imm16_math, config_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPCONFIG(imm16_math, config_dest, instr_mod1), 0, 0, 0)
#define TT_SFPCONFIG(imm16_math, config_dest, instr_mod1)        TT_IP_SFPCONFIG(::ckernel::instrn_buffer, imm16_math, config_dest, instr_mod1)
#define TTI_SFPCONFIG(imm16_math, config_dest, instr_mod1)       TT_IP_SFPCONFIG(nullptr, imm16_math, config_dest, instr_mod1)

#define TT_OP_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x76, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPDIVP2_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPDIVP2(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPDIVP2(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPDIVP2(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPENCC(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x8a, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPENCC_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPENCC(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPENCC(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPENCC(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPENCC(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPENCC(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPENCC(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPEXEXP(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x77, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPEXEXP_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPEXEXP(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPEXEXP(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPEXEXP(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPEXEXP(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPEXEXP(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPEXEXP(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPEXMAN(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x78, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPEXMAN_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPEXMAN(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPEXMAN(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPEXMAN(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPEXMAN(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPEXMAN(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPEXMAN(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPGT(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x97, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPGT_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPGT(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPGT(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPGT(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPGT(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPGT(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPGT(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x79, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPIADD_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPIADD(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPIADD(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPIADD(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPLE(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x96, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPLE_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPLE(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPLE(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPLE(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPLE(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPLE(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPLE(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    TT_OP(0x70, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((sfpu_addr_mode) << 13) + ((dest_reg_addr) << 0)))
#define TT_SFPLOAD_VALID(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    (ckernel::is_valid(lreg_ind, 4) && ckernel::is_valid(instr_mod0, 4) && ckernel::is_valid(sfpu_addr_mode, 3) && ckernel::is_valid(dest_reg_addr, 13))
#define TT_IP_SFPLOAD(IP, lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    TT_IP_INSN(IP, TT_OP_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr), 0, 0, 0)
#define TT_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    TT_IP_SFPLOAD(::ckernel::instrn_buffer, lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr)
#define TTI_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) TT_IP_SFPLOAD(nullptr, lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr)

#define TT_OP_SFPLOADI(lreg_ind, instr_mod0, imm16)     TT_OP(0x71, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((imm16) << 0)))
#define TT_SFPLOADI_VALID(lreg_ind, instr_mod0, imm16)  (ckernel::is_valid(lreg_ind, 4) && ckernel::is_valid(instr_mod0, 4) && ckernel::is_valid(imm16, 16))
#define TT_IP_SFPLOADI(IP, lreg_ind, instr_mod0, imm16) TT_IP_INSN(IP, TT_OP_SFPLOADI(lreg_ind, instr_mod0, imm16), 0, 0, 0)
#define TT_SFPLOADI(lreg_ind, instr_mod0, imm16)        TT_IP_SFPLOADI(::ckernel::instrn_buffer, lreg_ind, instr_mod0, imm16)
#define TTI_SFPLOADI(lreg_ind, instr_mod0, imm16)       TT_IP_SFPLOADI(nullptr, lreg_ind, instr_mod0, imm16)

#define TT_OP_SFPLOADMACRO(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    TT_OP(0x93, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((sfpu_addr_mode) << 13) + ((dest_reg_addr) << 0)))
#define TT_SFPLOADMACRO_VALID(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    (ckernel::is_valid(lreg_ind, 4) && ckernel::is_valid(instr_mod0, 4) && ckernel::is_valid(sfpu_addr_mode, 3) && ckernel::is_valid(dest_reg_addr, 13))
#define TT_IP_SFPLOADMACRO(IP, lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    TT_IP_INSN(IP, TT_OP_SFPLOADMACRO(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr), 0, 0, 0)
#define TT_SFPLOADMACRO(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    TT_IP_SFPLOADMACRO(::ckernel::instrn_buffer, lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr)
#define TTI_SFPLOADMACRO(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) TT_IP_SFPLOADMACRO(nullptr, lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr)

#define TT_OP_SFPLUT(lreg_ind, instr_mod0, dest_reg_addr) TT_OP(0x73, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((dest_reg_addr) << 0)))
#define TT_SFPLUT_VALID(lreg_ind, instr_mod0, dest_reg_addr) \
    (ckernel::is_valid(lreg_ind, 4) && ckernel::is_valid(instr_mod0, 4) && ckernel::is_valid(dest_reg_addr, 16))
#define TT_IP_SFPLUT(IP, lreg_ind, instr_mod0, dest_reg_addr) TT_IP_INSN(IP, TT_OP_SFPLUT(lreg_ind, instr_mod0, dest_reg_addr), 0, 0, 0)
#define TT_SFPLUT(lreg_ind, instr_mod0, dest_reg_addr)        TT_IP_SFPLUT(::ckernel::instrn_buffer, lreg_ind, instr_mod0, dest_reg_addr)
#define TTI_SFPLUT(lreg_ind, instr_mod0, dest_reg_addr)       TT_IP_SFPLUT(nullptr, lreg_ind, instr_mod0, dest_reg_addr)

#define TT_OP_SFPLUTFP32(lreg_dest, instr_mod1)     TT_OP(0x95, (((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPLUTFP32_VALID(lreg_dest, instr_mod1)  (ckernel::is_valid(lreg_dest, 20) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPLUTFP32(IP, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPLUTFP32(lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPLUTFP32(lreg_dest, instr_mod1)        TT_IP_SFPLUTFP32(::ckernel::instrn_buffer, lreg_dest, instr_mod1)
#define TTI_SFPLUTFP32(lreg_dest, instr_mod1)       TT_IP_SFPLUTFP32(nullptr, lreg_dest, instr_mod1)

#define TT_OP_SFPLZ(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x81, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPLZ_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPLZ(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPLZ(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPLZ(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPLZ(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPLZ(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPLZ(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPMAD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x84, (((lreg_src_a) << 16) + ((lreg_src_b) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPMAD_VALID(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)                                                                  \
    (ckernel::is_valid(lreg_src_a, 8) && ckernel::is_valid(lreg_src_b, 4) && ckernel::is_valid(lreg_src_c, 4) && ckernel::is_valid(lreg_dest, 4) && \
     ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPMAD(IP, lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_IP_INSN(IP, TT_OP_SFPMAD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPMAD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_IP_SFPMAD(::ckernel::instrn_buffer, lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPMAD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) TT_IP_SFPMAD(nullptr, lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)

#define TT_OP_SFPMOV(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x7c, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPMOV_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPMOV(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPMOV(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPMOV(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPMOV(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPMOV(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPMOV(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPMUL(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x86, (((lreg_src_a) << 16) + ((lreg_src_b) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPMUL_VALID(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)                                                                  \
    (ckernel::is_valid(lreg_src_a, 8) && ckernel::is_valid(lreg_src_b, 4) && ckernel::is_valid(lreg_src_c, 4) && ckernel::is_valid(lreg_dest, 4) && \
     ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPMUL(IP, lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_IP_INSN(IP, TT_OP_SFPMUL(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPMUL(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_IP_SFPMUL(::ckernel::instrn_buffer, lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPMUL(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) TT_IP_SFPMUL(nullptr, lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)

#define TT_OP_SFPMUL24(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x98, (((lreg_src_a) << 16) + ((lreg_src_b) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPMUL24_VALID(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)                                                                \
    (ckernel::is_valid(lreg_src_a, 8) && ckernel::is_valid(lreg_src_b, 4) && ckernel::is_valid(lreg_src_c, 4) && ckernel::is_valid(lreg_dest, 4) && \
     ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPMUL24(IP, lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_IP_INSN(IP, TT_OP_SFPMUL24(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPMUL24(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_IP_SFPMUL24(::ckernel::instrn_buffer, lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPMUL24(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_IP_SFPMUL24(nullptr, lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)

#define TT_OP_SFPMULI(imm16_math, lreg_dest, instr_mod1) TT_OP(0x74, (((imm16_math) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPMULI_VALID(imm16_math, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm16_math, 16) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPMULI(IP, imm16_math, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPMULI(imm16_math, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPMULI(imm16_math, lreg_dest, instr_mod1)        TT_IP_SFPMULI(::ckernel::instrn_buffer, imm16_math, lreg_dest, instr_mod1)
#define TTI_SFPMULI(imm16_math, lreg_dest, instr_mod1)       TT_IP_SFPMULI(nullptr, imm16_math, lreg_dest, instr_mod1)

#define TT_OP_SFPNOP     TT_OP(0x8f, 0)
#define TT_IP_SFPNOP(IP) TT_IP_INSN(IP, TT_OP_SFPNOP, 0, 0, 0)
#define TTI_SFPNOP       TT_IP_SFPNOP(nullptr)

#define TT_OP_SFPNOT(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x80, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPNOT_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPNOT(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPNOT(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPNOT(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPNOT(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPNOT(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPNOT(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPOR(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x7f, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPOR_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPOR(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPOR(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPOR(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPOR(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPOR(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPOR(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPPOPC(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x88, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPPOPC_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPPOPC(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPPOPC(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPPOPC(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPPOPC(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPPOPC(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPPOPC(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPPUSHC(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x87, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPPUSHC_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPPUSHC(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPPUSHC(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPPUSHC(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPPUSHC(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPPUSHC(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPPUSHC(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPSETCC(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x7b, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSETCC_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPSETCC(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSETCC(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSETCC(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPSETCC(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSETCC(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPSETCC(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x82, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSETEXP_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPSETEXP(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPSETEXP(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPSETEXP(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x83, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSETMAN_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPSETMAN(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPSETMAN(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPSETMAN(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x89, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSETSGN_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPSETSGN(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPSETSGN(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPSETSGN(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x7a, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSHFT_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPSHFT(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPSHFT(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPSHFT(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPSHFT2(imm12_math, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x94, (((imm12_math) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSHFT2_VALID(imm12_math, lreg_src_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_src_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPSHFT2(IP, imm12_math, lreg_src_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSHFT2(imm12_math, lreg_src_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSHFT2(imm12_math, lreg_src_c, lreg_dest, instr_mod1)        TT_IP_SFPSHFT2(::ckernel::instrn_buffer, imm12_math, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPSHFT2(imm12_math, lreg_src_c, lreg_dest, instr_mod1)       TT_IP_SFPSHFT2(nullptr, imm12_math, lreg_src_c, lreg_dest, instr_mod1)

#define TT_OP_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    TT_OP(0x72, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((sfpu_addr_mode) << 13) + ((dest_reg_addr) << 0)))
#define TT_SFPSTORE_VALID(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    (ckernel::is_valid(lreg_ind, 4) && ckernel::is_valid(instr_mod0, 4) && ckernel::is_valid(sfpu_addr_mode, 3) && ckernel::is_valid(dest_reg_addr, 13))
#define TT_IP_SFPSTORE(IP, lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    TT_IP_INSN(IP, TT_OP_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr), 0, 0, 0)
#define TT_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    TT_IP_SFPSTORE(::ckernel::instrn_buffer, lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr)
#define TTI_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) TT_IP_SFPSTORE(nullptr, lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr)

#define TT_OP_SFPSWAP(imm12_math, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x92, (((imm12_math) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSWAP_VALID(imm12_math, lreg_src_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_src_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPSWAP(IP, imm12_math, lreg_src_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPSWAP(imm12_math, lreg_src_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPSWAP(imm12_math, lreg_src_c, lreg_dest, instr_mod1)        TT_IP_SFPSWAP(::ckernel::instrn_buffer, imm12_math, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPSWAP(imm12_math, lreg_src_c, lreg_dest, instr_mod1)       TT_IP_SFPSWAP(nullptr, imm12_math, lreg_src_c, lreg_dest, instr_mod1)

#define TT_OP_SFPTRANSP(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x8c, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPTRANSP_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPTRANSP(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPTRANSP(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPTRANSP(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPTRANSP(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPTRANSP(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPTRANSP(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFPXOR(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x8d, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPXOR_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFPXOR(IP, imm12_math, lreg_c, lreg_dest, instr_mod1) TT_IP_INSN(IP, TT_OP_SFPXOR(imm12_math, lreg_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFPXOR(imm12_math, lreg_c, lreg_dest, instr_mod1)        TT_IP_SFPXOR(::ckernel::instrn_buffer, imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPXOR(imm12_math, lreg_c, lreg_dest, instr_mod1)       TT_IP_SFPXOR(nullptr, imm12_math, lreg_c, lreg_dest, instr_mod1)

#define TT_OP_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x8e, (((rnd_mode) << 21) + ((imm8_math) << 16) + ((lreg_src_b) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFP_STOCH_RND_VALID(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)                                                \
    (ckernel::is_valid(rnd_mode, 3) && ckernel::is_valid(imm8_math, 5) && ckernel::is_valid(lreg_src_b, 4) && ckernel::is_valid(lreg_src_c, 4) && \
     ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_IP_SFP_STOCH_RND(IP, rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_IP_INSN(IP, TT_OP_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1), 0, 0, 0)
#define TT_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_IP_SFP_STOCH_RND(::ckernel::instrn_buffer, rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_IP_SFP_STOCH_RND(nullptr, rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)

#define TT_OP_SHIFTDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_OP(0x5c, (((OpBisConst) << 23) + ((OpSel) << 18) + ((ResultRegIndex) << 12) + ((OpBRegIndex) << 6) + ((OpARegIndex) << 0)))
#define TT_SHIFTDMAREG_VALID(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)                                                            \
    (ckernel::is_valid(OpBisConst, 1) && ckernel::is_valid(OpSel, 5) && ckernel::is_valid(ResultRegIndex, 6) && ckernel::is_valid(OpBRegIndex, 6) && \
     ckernel::is_valid(OpARegIndex, 6))
#define TT_IP_SHIFTDMAREG(IP, OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_INSN(IP, TT_OP_SHIFTDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex), 0, 0, 0)
#define TT_SHIFTDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_SHIFTDMAREG(::ckernel::instrn_buffer, OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)
#define TTI_SHIFTDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_SHIFTDMAREG(nullptr, OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)

#define TT_OP_SHIFTXA(log2_amount2, shift_mode)     TT_OP(0x17, (((log2_amount2) << 2) + ((shift_mode) << 0)))
#define TT_SHIFTXA_VALID(log2_amount2, shift_mode)  (ckernel::is_valid(log2_amount2, 22) && ckernel::is_valid(shift_mode, 2))
#define TT_IP_SHIFTXA(IP, log2_amount2, shift_mode) TT_IP_INSN(IP, TT_OP_SHIFTXA(log2_amount2, shift_mode), 0, 0, 0)
#define TT_SHIFTXA(log2_amount2, shift_mode)        TT_IP_SHIFTXA(::ckernel::instrn_buffer, log2_amount2, shift_mode)
#define TTI_SHIFTXA(log2_amount2, shift_mode)       TT_IP_SHIFTXA(nullptr, log2_amount2, shift_mode)

#define TT_OP_SHIFTXB(addr_mode, rot_shift, shift_row) TT_OP(0x18, (((addr_mode) << 14) + ((rot_shift) << 10) + ((shift_row) << 0)))
#define TT_SHIFTXB_VALID(addr_mode, rot_shift, shift_row) \
    (ckernel::is_valid(addr_mode, 10) && ckernel::is_valid(rot_shift, 4) && ckernel::is_valid(shift_row, 10))
#define TT_IP_SHIFTXB(IP, addr_mode, rot_shift, shift_row) TT_IP_INSN(IP, TT_OP_SHIFTXB(addr_mode, rot_shift, shift_row), 0, 0, 0)
#define TT_SHIFTXB(addr_mode, rot_shift, shift_row)        TT_IP_SHIFTXB(::ckernel::instrn_buffer, addr_mode, rot_shift, shift_row)
#define TTI_SHIFTXB(addr_mode, rot_shift, shift_row)       TT_IP_SHIFTXB(nullptr, addr_mode, rot_shift, shift_row)

#define TT_OP_STALLWAIT(stall_res, wait_res)     TT_OP(0xa2, (((stall_res) << 15) + ((wait_res) << 0)))
#define TT_STALLWAIT_VALID(stall_res, wait_res)  (ckernel::is_valid(stall_res, 9) && ckernel::is_valid(wait_res, 15))
#define TT_IP_STALLWAIT(IP, stall_res, wait_res) TT_IP_INSN(IP, TT_OP_STALLWAIT(stall_res, wait_res), 0, 0, 0)
#define TT_STALLWAIT(stall_res, wait_res)        TT_IP_STALLWAIT(::ckernel::instrn_buffer, stall_res, wait_res)
#define TTI_STALLWAIT(stall_res, wait_res)       TT_IP_STALLWAIT(nullptr, stall_res, wait_res)

#define TT_OP_STOREIND(MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex)                                      \
    TT_OP(                                                                                                                                         \
        0x66,                                                                                                                                      \
        (((MemHierSel) << 23) + ((SizeSel) << 22) + ((RegSizeSel) << 21) + ((OffsetIndex) << 14) + ((AutoIncSpec) << 12) + ((DataRegIndex) << 6) + \
         ((AddrRegIndex) << 0)))
#define TT_STOREIND_VALID(MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex)                                   \
    (ckernel::is_valid(MemHierSel, 1) && ckernel::is_valid(SizeSel, 1) && ckernel::is_valid(RegSizeSel, 1) && ckernel::is_valid(OffsetIndex, 7) && \
     ckernel::is_valid(AutoIncSpec, 2) && ckernel::is_valid(DataRegIndex, 6) && ckernel::is_valid(AddrRegIndex, 6))
#define TT_IP_STOREIND(IP, MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex) \
    TT_IP_INSN(IP, TT_OP_STOREIND(MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex), 0, 0, 0)
#define TT_STOREIND(MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex) \
    TT_IP_STOREIND(::ckernel::instrn_buffer, MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex)
#define TTI_STOREIND(MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex) \
    TT_IP_STOREIND(nullptr, MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex)

#define TT_OP_STOREREG(TdmaDataRegIndex, RegAddr)     TT_OP(0x67, (((TdmaDataRegIndex) << 18) + ((RegAddr) << 0)))
#define TT_STOREREG_VALID(TdmaDataRegIndex, RegAddr)  (ckernel::is_valid(TdmaDataRegIndex, 6) && ckernel::is_valid(RegAddr, 18))
#define TT_IP_STOREREG(IP, TdmaDataRegIndex, RegAddr) TT_IP_INSN(IP, TT_OP_STOREREG(TdmaDataRegIndex, RegAddr), 0, 0, 0)
#define TT_STOREREG(TdmaDataRegIndex, RegAddr)        TT_IP_STOREREG(::ckernel::instrn_buffer, TdmaDataRegIndex, RegAddr)
#define TTI_STOREREG(TdmaDataRegIndex, RegAddr)       TT_IP_STOREREG(nullptr, TdmaDataRegIndex, RegAddr)

#define TT_OP_STREAMWAIT(stall_res, target_value, target_sel, wait_stream_sel) \
    TT_OP(0xa7, (((stall_res) << 15) + ((target_value) << 4) + ((target_sel) << 3) + ((wait_stream_sel) << 0)))
#define TT_STREAMWAIT_VALID(stall_res, target_value, target_sel, wait_stream_sel) \
    (ckernel::is_valid(stall_res, 9) && ckernel::is_valid(target_value, 11) && ckernel::is_valid(target_sel, 1) && ckernel::is_valid(wait_stream_sel, 3))
#define TT_IP_STREAMWAIT(IP, stall_res, target_value, target_sel, wait_stream_sel) \
    TT_IP_INSN(IP, TT_OP_STREAMWAIT(stall_res, target_value, target_sel, wait_stream_sel), 0, 0, 0)
#define TT_STREAMWAIT(stall_res, target_value, target_sel, wait_stream_sel) \
    TT_IP_STREAMWAIT(::ckernel::instrn_buffer, stall_res, target_value, target_sel, wait_stream_sel)
#define TTI_STREAMWAIT(stall_res, target_value, target_sel, wait_stream_sel) TT_IP_STREAMWAIT(nullptr, stall_res, target_value, target_sel, wait_stream_sel)

#define TT_OP_STREAMWRCFG(stream_id_sel, StreamRegAddr, CfgReg) TT_OP(0xb7, (((stream_id_sel) << 21) + ((StreamRegAddr) << 11) + ((CfgReg) << 0)))
#define TT_STREAMWRCFG_VALID(stream_id_sel, StreamRegAddr, CfgReg) \
    (ckernel::is_valid(stream_id_sel, 3) && ckernel::is_valid(StreamRegAddr, 10) && ckernel::is_valid(CfgReg, 11))
#define TT_IP_STREAMWRCFG(IP, stream_id_sel, StreamRegAddr, CfgReg) TT_IP_INSN(IP, TT_OP_STREAMWRCFG(stream_id_sel, StreamRegAddr, CfgReg), 0, 0, 0)
#define TT_STREAMWRCFG(stream_id_sel, StreamRegAddr, CfgReg)        TT_IP_STREAMWRCFG(::ckernel::instrn_buffer, stream_id_sel, StreamRegAddr, CfgReg)
#define TTI_STREAMWRCFG(stream_id_sel, StreamRegAddr, CfgReg)       TT_IP_STREAMWRCFG(nullptr, stream_id_sel, StreamRegAddr, CfgReg)

#define TT_OP_SUBDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_OP(0x59, (((OpBisConst) << 23) + ((ResultRegIndex) << 12) + ((OpBRegIndex) << 6) + ((OpARegIndex) << 0)))
#define TT_SUBDMAREG_VALID(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    (ckernel::is_valid(OpBisConst, 1) && ckernel::is_valid(ResultRegIndex, 11) && ckernel::is_valid(OpBRegIndex, 6) && ckernel::is_valid(OpARegIndex, 6))
#define TT_IP_SUBDMAREG(IP, OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_INSN(IP, TT_OP_SUBDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex), 0, 0, 0)
#define TT_SUBDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_IP_SUBDMAREG(::ckernel::instrn_buffer, OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex)
#define TTI_SUBDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) TT_IP_SUBDMAREG(nullptr, OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex)

#define TT_OP_TBUFCMD     TT_OP(0x4b, 0)
#define TT_IP_TBUFCMD(IP) TT_IP_INSN(IP, TT_OP_TBUFCMD, 0, 0, 0)
#define TTI_TBUFCMD       TT_IP_TBUFCMD(nullptr)

#define TT_OP_TRNSPSRCA     TT_OP(0x14, 0)
#define TT_IP_TRNSPSRCA(IP) TT_IP_INSN(IP, TT_OP_TRNSPSRCA, 0, 0, 0)
#define TTI_TRNSPSRCA       TT_IP_TRNSPSRCA(nullptr)

#define TT_OP_TRNSPSRCB     TT_OP(0x16, 0)
#define TT_IP_TRNSPSRCB(IP) TT_IP_INSN(IP, TT_OP_TRNSPSRCB, 0, 0, 0)
#define TTI_TRNSPSRCB       TT_IP_TRNSPSRCB(nullptr)

#define TT_OP_UNPACR(                                                                                                                                \
    Unpack_block_selection,                                                                                                                          \
    AddrMode,                                                                                                                                        \
    CfgContextCntInc,                                                                                                                                \
    CfgContextId,                                                                                                                                    \
    AddrCntContextId,                                                                                                                                \
    OvrdThreadId,                                                                                                                                    \
    SetDatValid,                                                                                                                                     \
    srcb_bcast,                                                                                                                                      \
    ZeroWrite2,                                                                                                                                      \
    AutoIncContextID,                                                                                                                                \
    RowSearch,                                                                                                                                       \
    SearchCacheFlush,                                                                                                                                \
    Last)                                                                                                                                            \
    TT_OP(                                                                                                                                           \
        0x42,                                                                                                                                        \
        (((Unpack_block_selection) << 23) + ((AddrMode) << 15) + ((CfgContextCntInc) << 13) + ((CfgContextId) << 10) + ((AddrCntContextId) << 8) +   \
         ((OvrdThreadId) << 7) + ((SetDatValid) << 6) + ((srcb_bcast) << 5) + ((ZeroWrite2) << 4) + ((AutoIncContextID) << 3) + ((RowSearch) << 2) + \
         ((SearchCacheFlush) << 1) + ((Last) << 0)))
#define TT_UNPACR_VALID(                                                                                                                                    \
    Unpack_block_selection,                                                                                                                                 \
    AddrMode,                                                                                                                                               \
    CfgContextCntInc,                                                                                                                                       \
    CfgContextId,                                                                                                                                           \
    AddrCntContextId,                                                                                                                                       \
    OvrdThreadId,                                                                                                                                           \
    SetDatValid,                                                                                                                                            \
    srcb_bcast,                                                                                                                                             \
    ZeroWrite2,                                                                                                                                             \
    AutoIncContextID,                                                                                                                                       \
    RowSearch,                                                                                                                                              \
    SearchCacheFlush,                                                                                                                                       \
    Last)                                                                                                                                                   \
    (ckernel::is_valid(Unpack_block_selection, 1) && ckernel::is_valid(AddrMode, 8) && ckernel::is_valid(CfgContextCntInc, 2) &&                            \
     ckernel::is_valid(CfgContextId, 3) && ckernel::is_valid(AddrCntContextId, 2) && ckernel::is_valid(OvrdThreadId, 1) &&                                  \
     ckernel::is_valid(SetDatValid, 1) && ckernel::is_valid(srcb_bcast, 1) && ckernel::is_valid(ZeroWrite2, 1) && ckernel::is_valid(AutoIncContextID, 1) && \
     ckernel::is_valid(RowSearch, 1) && ckernel::is_valid(SearchCacheFlush, 1) && ckernel::is_valid(Last, 1))
#define TT_IP_UNPACR(               \
    IP,                             \
    Unpack_block_selection,         \
    AddrMode,                       \
    CfgContextCntInc,               \
    CfgContextId,                   \
    AddrCntContextId,               \
    OvrdThreadId,                   \
    SetDatValid,                    \
    srcb_bcast,                     \
    ZeroWrite2,                     \
    AutoIncContextID,               \
    RowSearch,                      \
    SearchCacheFlush,               \
    Last)                           \
    TT_IP_INSN(                     \
        IP,                         \
        TT_OP_UNPACR(               \
            Unpack_block_selection, \
            AddrMode,               \
            CfgContextCntInc,       \
            CfgContextId,           \
            AddrCntContextId,       \
            OvrdThreadId,           \
            SetDatValid,            \
            srcb_bcast,             \
            ZeroWrite2,             \
            AutoIncContextID,       \
            RowSearch,              \
            SearchCacheFlush,       \
            Last),                  \
        0,                          \
        0,                          \
        0)
#define TT_UNPACR(                \
    Unpack_block_selection,       \
    AddrMode,                     \
    CfgContextCntInc,             \
    CfgContextId,                 \
    AddrCntContextId,             \
    OvrdThreadId,                 \
    SetDatValid,                  \
    srcb_bcast,                   \
    ZeroWrite2,                   \
    AutoIncContextID,             \
    RowSearch,                    \
    SearchCacheFlush,             \
    Last)                         \
    TT_IP_UNPACR(                 \
        ::ckernel::instrn_buffer, \
        Unpack_block_selection,   \
        AddrMode,                 \
        CfgContextCntInc,         \
        CfgContextId,             \
        AddrCntContextId,         \
        OvrdThreadId,             \
        SetDatValid,              \
        srcb_bcast,               \
        ZeroWrite2,               \
        AutoIncContextID,         \
        RowSearch,                \
        SearchCacheFlush,         \
        Last)
#define TTI_UNPACR(             \
    Unpack_block_selection,     \
    AddrMode,                   \
    CfgContextCntInc,           \
    CfgContextId,               \
    AddrCntContextId,           \
    OvrdThreadId,               \
    SetDatValid,                \
    srcb_bcast,                 \
    ZeroWrite2,                 \
    AutoIncContextID,           \
    RowSearch,                  \
    SearchCacheFlush,           \
    Last)                       \
    TT_IP_UNPACR(               \
        nullptr,                \
        Unpack_block_selection, \
        AddrMode,               \
        CfgContextCntInc,       \
        CfgContextId,           \
        AddrCntContextId,       \
        OvrdThreadId,           \
        SetDatValid,            \
        srcb_bcast,             \
        ZeroWrite2,             \
        AutoIncContextID,       \
        RowSearch,              \
        SearchCacheFlush,       \
        Last)

#define TT_OP_UNPACR_NOP(Unpacker_Select, Stream_Id, Msg_Clr_Cnt, Set_Dvalid, Clr_to1_fmt_Ctrl, Stall_Clr_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Unpack_Pop) \
    TT_OP(                                                                                                                                                   \
        0x43,                                                                                                                                                \
        (((Unpacker_Select) << 23) + ((Stream_Id) << 16) + ((Msg_Clr_Cnt) << 12) + ((Set_Dvalid) << 8) + ((Clr_to1_fmt_Ctrl) << 6) +                         \
         ((Stall_Clr_Cntrl) << 5) + ((Bank_Clr_Ctrl) << 4) + ((Src_ClrVal_Ctrl) << 2) + ((Unpack_Pop) << 0)))
#define TT_UNPACR_NOP_VALID(                                                                                                                              \
    Unpacker_Select, Stream_Id, Msg_Clr_Cnt, Set_Dvalid, Clr_to1_fmt_Ctrl, Stall_Clr_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Unpack_Pop)                   \
    (ckernel::is_valid(Unpacker_Select, 1) && ckernel::is_valid(Stream_Id, 7) && ckernel::is_valid(Msg_Clr_Cnt, 4) && ckernel::is_valid(Set_Dvalid, 4) && \
     ckernel::is_valid(Clr_to1_fmt_Ctrl, 2) && ckernel::is_valid(Stall_Clr_Cntrl, 1) && ckernel::is_valid(Bank_Clr_Ctrl, 1) &&                            \
     ckernel::is_valid(Src_ClrVal_Ctrl, 2) && ckernel::is_valid(Unpack_Pop, 2))
#define TT_IP_UNPACR_NOP(                                                                                                                                     \
    IP, Unpacker_Select, Stream_Id, Msg_Clr_Cnt, Set_Dvalid, Clr_to1_fmt_Ctrl, Stall_Clr_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Unpack_Pop)                   \
    TT_IP_INSN(                                                                                                                                               \
        IP,                                                                                                                                                   \
        TT_OP_UNPACR_NOP(Unpacker_Select, Stream_Id, Msg_Clr_Cnt, Set_Dvalid, Clr_to1_fmt_Ctrl, Stall_Clr_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Unpack_Pop), \
        0,                                                                                                                                                    \
        0,                                                                                                                                                    \
        0)
#define TT_UNPACR_NOP(Unpacker_Select, Stream_Id, Msg_Clr_Cnt, Set_Dvalid, Clr_to1_fmt_Ctrl, Stall_Clr_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Unpack_Pop) \
    TT_IP_UNPACR_NOP(                                                                                                                                     \
        ::ckernel::instrn_buffer,                                                                                                                         \
        Unpacker_Select,                                                                                                                                  \
        Stream_Id,                                                                                                                                        \
        Msg_Clr_Cnt,                                                                                                                                      \
        Set_Dvalid,                                                                                                                                       \
        Clr_to1_fmt_Ctrl,                                                                                                                                 \
        Stall_Clr_Cntrl,                                                                                                                                  \
        Bank_Clr_Ctrl,                                                                                                                                    \
        Src_ClrVal_Ctrl,                                                                                                                                  \
        Unpack_Pop)
#define TTI_UNPACR_NOP(Unpacker_Select, Stream_Id, Msg_Clr_Cnt, Set_Dvalid, Clr_to1_fmt_Ctrl, Stall_Clr_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Unpack_Pop) \
    TT_IP_UNPACR_NOP(                                                                                                                                      \
        nullptr, Unpacker_Select, Stream_Id, Msg_Clr_Cnt, Set_Dvalid, Clr_to1_fmt_Ctrl, Stall_Clr_Cntrl, Bank_Clr_Ctrl, Src_ClrVal_Ctrl, Unpack_Pop)

#define TT_OP_WRCFG(GprAddress, wr128b, CfgReg)     TT_OP(0xb0, (((GprAddress) << 16) + ((wr128b) << 15) + ((CfgReg) << 0)))
#define TT_WRCFG_VALID(GprAddress, wr128b, CfgReg)  (ckernel::is_valid(GprAddress, 8) && ckernel::is_valid(wr128b, 1) && ckernel::is_valid(CfgReg, 15))
#define TT_IP_WRCFG(IP, GprAddress, wr128b, CfgReg) TT_IP_INSN(IP, TT_OP_WRCFG(GprAddress, wr128b, CfgReg), 0, 0, 0)
#define TT_WRCFG(GprAddress, wr128b, CfgReg)        TT_IP_WRCFG(::ckernel::instrn_buffer, GprAddress, wr128b, CfgReg)
#define TTI_WRCFG(GprAddress, wr128b, CfgReg)       TT_IP_WRCFG(nullptr, GprAddress, wr128b, CfgReg)

#define TT_OP_XMOV(Mov_block_selection, Last)     TT_OP(0x40, (((Mov_block_selection) << 23) + ((Last) << 0)))
#define TT_XMOV_VALID(Mov_block_selection, Last)  (ckernel::is_valid(Mov_block_selection, 1) && ckernel::is_valid(Last, 23))
#define TT_IP_XMOV(IP, Mov_block_selection, Last) TT_IP_INSN(IP, TT_OP_XMOV(Mov_block_selection, Last), 0, 0, 0)
#define TT_XMOV(Mov_block_selection, Last)        TT_IP_XMOV(::ckernel::instrn_buffer, Mov_block_selection, Last)
#define TTI_XMOV(Mov_block_selection, Last)       TT_IP_XMOV(nullptr, Mov_block_selection, Last)

#define TT_OP_ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where) \
    TT_OP(0x10, (((clear_mode) << 19) + ((use_32_bit_mode) << 18) + ((clear_zero_flags) << 17) + ((addr_mode) << 14) + ((where) << 0)))
#define TT_ZEROACC_VALID(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where)                                                                      \
    (ckernel::is_valid(clear_mode, 5) && ckernel::is_valid(use_32_bit_mode, 1) && ckernel::is_valid(clear_zero_flags, 1) && ckernel::is_valid(addr_mode, 3) && \
     ckernel::is_valid(where, 14))
#define TT_IP_ZEROACC(IP, clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where) \
    TT_IP_INSN(IP, TT_OP_ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where), 0, 0, 0)
#define TT_ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where) \
    TT_IP_ZEROACC(::ckernel::instrn_buffer, clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where)
#define TTI_ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where) \
    TT_IP_ZEROACC(nullptr, clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where)

#define TT_OP_ZEROSRC(zero_val, write_mode, bank_mask, src_mask) TT_OP(0x11, (((zero_val) << 4) + ((write_mode) << 3) + ((bank_mask) << 2) + ((src_mask) << 0)))
#define TT_ZEROSRC_VALID(zero_val, write_mode, bank_mask, src_mask) \
    (ckernel::is_valid(zero_val, 20) && ckernel::is_valid(write_mode, 1) && ckernel::is_valid(bank_mask, 1) && ckernel::is_valid(src_mask, 2))
#define TT_IP_ZEROSRC(IP, zero_val, write_mode, bank_mask, src_mask) TT_IP_INSN(IP, TT_OP_ZEROSRC(zero_val, write_mode, bank_mask, src_mask), 0, 0, 0)
#define TT_ZEROSRC(zero_val, write_mode, bank_mask, src_mask)        TT_IP_ZEROSRC(::ckernel::instrn_buffer, zero_val, write_mode, bank_mask, src_mask)
#define TTI_ZEROSRC(zero_val, write_mode, bank_mask, src_mask)       TT_IP_ZEROSRC(nullptr, zero_val, write_mode, bank_mask, src_mask)
