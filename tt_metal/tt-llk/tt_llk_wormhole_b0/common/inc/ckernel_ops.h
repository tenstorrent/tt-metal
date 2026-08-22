// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Auto-generated file, do not modify -- except for the intrinsic redirect
// block guarded by #if defined(__riscv_xtttensixwh) near the end.  That block is
// maintained by hand and appended after the generated content.
// Regenerating this file drops it; keep it, or the .ttinsn asm comes back
// and every Tensix instruction is a barrier to pass_rvtt_config again.
//

#pragma once

#define TT_OP(opcode, params) ((opcode << 24) + params)
#define INSTRUCTION_WORD(x)   __asm__ __volatile__(".ttinsn %0" : : "n"((x))) // Swizzle 32 bits into the instruction stream.

#define TT_OP_ADDDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_OP(0x58, (((OpBisConst) << 23) + ((ResultRegIndex) << 12) + ((OpBRegIndex) << 6) + ((OpARegIndex) << 0)))
#define TT_ADDDMAREG_VALID(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    (ckernel::is_valid(OpBisConst, 1) && ckernel::is_valid(ResultRegIndex, 6) && ckernel::is_valid(OpBRegIndex, 6) && ckernel::is_valid(OpARegIndex, 6))
#define TT_ADDDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    ckernel::instrn_buffer[0] = TT_OP_ADDDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex)
#define TTI_ADDDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    INSTRUCTION_WORD(TT_OP_ADDDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex))

#define TT_OP_ADDRCRXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) \
    TT_OP(0x53, (((CntSetMask) << 21) + ((Ch1_Y) << 15) + ((Ch1_X) << 12) + ((Ch0_Y) << 9) + ((Ch0_X) << 6) + ((BitMask) << 0)))
#define TT_ADDRCRXY_VALID(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)                                                            \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(Ch1_Y, 6) && ckernel::is_valid(Ch1_X, 3) && ckernel::is_valid(Ch0_Y, 3) && \
     ckernel::is_valid(Ch0_X, 3) && ckernel::is_valid(BitMask, 6))
#define TT_ADDRCRXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)  ckernel::instrn_buffer[0] = TT_OP_ADDRCRXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)
#define TTI_ADDRCRXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) INSTRUCTION_WORD(TT_OP_ADDRCRXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask))

#define TT_OP_ADDRCRZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) \
    TT_OP(0x56, (((CntSetMask) << 21) + ((Ch1_Y) << 15) + ((Ch1_X) << 12) + ((Ch0_Y) << 9) + ((Ch0_X) << 6) + ((BitMask) << 0)))
#define TT_ADDRCRZW_VALID(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)                                                            \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(Ch1_Y, 6) && ckernel::is_valid(Ch1_X, 3) && ckernel::is_valid(Ch0_Y, 3) && \
     ckernel::is_valid(Ch0_X, 3) && ckernel::is_valid(BitMask, 6))
#define TT_ADDRCRZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)  ckernel::instrn_buffer[0] = TT_OP_ADDRCRZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)
#define TTI_ADDRCRZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) INSTRUCTION_WORD(TT_OP_ADDRCRZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask))

#define TT_OP_APOOL3S1(clear_dvalid, addr_mode, index_en, dst) TT_OP(0x25, (((clear_dvalid) << 22) + ((addr_mode) << 15) + ((index_en) << 14) + ((dst) << 0)))
#define TT_APOOL3S1_VALID(clear_dvalid, addr_mode, index_en, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(addr_mode, 7) && ckernel::is_valid(index_en, 1) && ckernel::is_valid(dst, 14))
#define TT_APOOL3S1(clear_dvalid, addr_mode, index_en, dst)  ckernel::instrn_buffer[0] = TT_OP_APOOL3S1(clear_dvalid, addr_mode, index_en, dst)
#define TTI_APOOL3S1(clear_dvalid, addr_mode, index_en, dst) INSTRUCTION_WORD(TT_OP_APOOL3S1(clear_dvalid, addr_mode, index_en, dst))

#define TT_OP_APOOL3S2(clear_dvalid, addr_mode, index_en, dst) TT_OP(0x32, (((clear_dvalid) << 22) + ((addr_mode) << 15) + ((index_en) << 14) + ((dst) << 0)))
#define TT_APOOL3S2_VALID(clear_dvalid, addr_mode, index_en, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(addr_mode, 7) && ckernel::is_valid(index_en, 1) && ckernel::is_valid(dst, 14))
#define TT_APOOL3S2(clear_dvalid, addr_mode, index_en, dst)  ckernel::instrn_buffer[0] = TT_OP_APOOL3S2(clear_dvalid, addr_mode, index_en, dst)
#define TTI_APOOL3S2(clear_dvalid, addr_mode, index_en, dst) INSTRUCTION_WORD(TT_OP_APOOL3S2(clear_dvalid, addr_mode, index_en, dst))

#define TT_OP_ATCAS(MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_OP(0x64, (((MemHierSel) << 23) + ((SwapVal) << 18) + ((CmpVal) << 14) + ((Sel32b) << 12) + ((DataRegIndex) << 6) + ((AddrRegIndex) << 0)))
#define TT_ATCAS_VALID(MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex)                                                   \
    (ckernel::is_valid(MemHierSel, 1) && ckernel::is_valid(SwapVal, 4) && ckernel::is_valid(CmpVal, 4) && ckernel::is_valid(Sel32b, 2) && \
     ckernel::is_valid(DataRegIndex, 6) && ckernel::is_valid(AddrRegIndex, 6))
#define TT_ATCAS(MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex) \
    ckernel::instrn_buffer[0] = TT_OP_ATCAS(MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex)
#define TTI_ATCAS(MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex) \
    INSTRUCTION_WORD(TT_OP_ATCAS(MemHierSel, SwapVal, CmpVal, Sel32b, DataRegIndex, AddrRegIndex))

#define TT_OP_ATGETM(mutex_index)    TT_OP(0xa0, (((mutex_index) << 0)))
#define TT_ATGETM_VALID(mutex_index) (ckernel::is_valid(mutex_index, 24))
#define TT_ATGETM(mutex_index)       ckernel::instrn_buffer[0] = TT_OP_ATGETM(mutex_index)
#define TTI_ATGETM(mutex_index)      INSTRUCTION_WORD(TT_OP_ATGETM(mutex_index))

#define TT_OP_ATINCGET(MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_OP(0x61, (((MemHierSel) << 23) + ((WrapVal) << 14) + ((Sel32b) << 12) + ((DataRegIndex) << 6) + ((AddrRegIndex) << 0)))
#define TT_ATINCGET_VALID(MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex)                                                              \
    (ckernel::is_valid(MemHierSel, 1) && ckernel::is_valid(WrapVal, 9) && ckernel::is_valid(Sel32b, 2) && ckernel::is_valid(DataRegIndex, 6) && \
     ckernel::is_valid(AddrRegIndex, 6))
#define TT_ATINCGET(MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    ckernel::instrn_buffer[0] = TT_OP_ATINCGET(MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex)
#define TTI_ATINCGET(MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    INSTRUCTION_WORD(TT_OP_ATINCGET(MemHierSel, WrapVal, Sel32b, DataRegIndex, AddrRegIndex))

#define TT_OP_ATINCGETPTR(MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    TT_OP(                                                                                          \
        0x62,                                                                                       \
        (((MemHierSel) << 23) + ((NoIncr) << 22) + ((IncrVal) << 18) + ((WrapVal) << 14) + ((Sel32b) << 12) + ((DataRegIndex) << 6) + ((AddrRegIndex) << 0)))
#define TT_ATINCGETPTR_VALID(MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex)                                     \
    (ckernel::is_valid(MemHierSel, 1) && ckernel::is_valid(NoIncr, 1) && ckernel::is_valid(IncrVal, 4) && ckernel::is_valid(WrapVal, 4) && \
     ckernel::is_valid(Sel32b, 2) && ckernel::is_valid(DataRegIndex, 6) && ckernel::is_valid(AddrRegIndex, 6))
#define TT_ATINCGETPTR(MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    ckernel::instrn_buffer[0] = TT_OP_ATINCGETPTR(MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex)
#define TTI_ATINCGETPTR(MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex) \
    INSTRUCTION_WORD(TT_OP_ATINCGETPTR(MemHierSel, NoIncr, IncrVal, WrapVal, Sel32b, DataRegIndex, AddrRegIndex))

#define TT_OP_ATRELM(mutex_index)    TT_OP(0xa1, (((mutex_index) << 0)))
#define TT_ATRELM_VALID(mutex_index) (ckernel::is_valid(mutex_index, 24))
#define TT_ATRELM(mutex_index)       ckernel::instrn_buffer[0] = TT_OP_ATRELM(mutex_index)
#define TTI_ATRELM(mutex_index)      INSTRUCTION_WORD(TT_OP_ATRELM(mutex_index))

#define TT_OP_ATSWAP(MemHierSel, SwapMask, DataRegIndex, AddrRegIndex) \
    TT_OP(0x63, (((MemHierSel) << 23) + ((SwapMask) << 14) + ((DataRegIndex) << 6) + ((AddrRegIndex) << 0)))
#define TT_ATSWAP_VALID(MemHierSel, SwapMask, DataRegIndex, AddrRegIndex) \
    (ckernel::is_valid(MemHierSel, 1) && ckernel::is_valid(SwapMask, 9) && ckernel::is_valid(DataRegIndex, 6) && ckernel::is_valid(AddrRegIndex, 6))
#define TT_ATSWAP(MemHierSel, SwapMask, DataRegIndex, AddrRegIndex)  ckernel::instrn_buffer[0] = TT_OP_ATSWAP(MemHierSel, SwapMask, DataRegIndex, AddrRegIndex)
#define TTI_ATSWAP(MemHierSel, SwapMask, DataRegIndex, AddrRegIndex) INSTRUCTION_WORD(TT_OP_ATSWAP(MemHierSel, SwapMask, DataRegIndex, AddrRegIndex))

#define TT_OP_BITWOPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_OP(0x5b, (((OpBisConst) << 23) + ((OpSel) << 18) + ((ResultRegIndex) << 12) + ((OpBRegIndex) << 6) + ((OpARegIndex) << 0)))
#define TT_BITWOPDMAREG_VALID(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)                                                           \
    (ckernel::is_valid(OpBisConst, 1) && ckernel::is_valid(OpSel, 5) && ckernel::is_valid(ResultRegIndex, 6) && ckernel::is_valid(OpBRegIndex, 6) && \
     ckernel::is_valid(OpARegIndex, 6))
#define TT_BITWOPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    ckernel::instrn_buffer[0] = TT_OP_BITWOPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)
#define TTI_BITWOPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    INSTRUCTION_WORD(TT_OP_BITWOPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex))

#define TT_OP_CLEARDVALID(cleardvalid, reset)    TT_OP(0x36, (((cleardvalid) << 22) + ((reset) << 0)))
#define TT_CLEARDVALID_VALID(cleardvalid, reset) (ckernel::is_valid(cleardvalid, 2) && ckernel::is_valid(reset, 22))
#define TT_CLEARDVALID(cleardvalid, reset)       ckernel::instrn_buffer[0] = TT_OP_CLEARDVALID(cleardvalid, reset)
#define TTI_CLEARDVALID(cleardvalid, reset)      INSTRUCTION_WORD(TT_OP_CLEARDVALID(cleardvalid, reset))

#define TT_OP_CLREXPHIST TT_OP(0x21, 0)
#define TTI_CLREXPHIST   INSTRUCTION_WORD(TT_OP_CLREXPHIST)

#define TT_OP_CMPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_OP(0x5d, (((OpBisConst) << 23) + ((OpSel) << 18) + ((ResultRegIndex) << 12) + ((OpBRegIndex) << 6) + ((OpARegIndex) << 0)))
#define TT_CMPDMAREG_VALID(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)                                                              \
    (ckernel::is_valid(OpBisConst, 1) && ckernel::is_valid(OpSel, 5) && ckernel::is_valid(ResultRegIndex, 6) && ckernel::is_valid(OpBRegIndex, 6) && \
     ckernel::is_valid(OpARegIndex, 6))
#define TT_CMPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    ckernel::instrn_buffer[0] = TT_OP_CMPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)
#define TTI_CMPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    INSTRUCTION_WORD(TT_OP_CMPDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex))

#define TT_OP_CONV3S1(clear_dvalid, rotate_weights, addr_mode, dst) \
    TT_OP(0x22, (((clear_dvalid) << 22) + ((rotate_weights) << 17) + ((addr_mode) << 15) + ((dst) << 0)))
#define TT_CONV3S1_VALID(clear_dvalid, rotate_weights, addr_mode, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(rotate_weights, 5) && ckernel::is_valid(addr_mode, 2) && ckernel::is_valid(dst, 15))
#define TT_CONV3S1(clear_dvalid, rotate_weights, addr_mode, dst)  ckernel::instrn_buffer[0] = TT_OP_CONV3S1(clear_dvalid, rotate_weights, addr_mode, dst)
#define TTI_CONV3S1(clear_dvalid, rotate_weights, addr_mode, dst) INSTRUCTION_WORD(TT_OP_CONV3S1(clear_dvalid, rotate_weights, addr_mode, dst))

#define TT_OP_CONV3S2(clear_dvalid, rotate_weights, addr_mode, dst) \
    TT_OP(0x23, (((clear_dvalid) << 22) + ((rotate_weights) << 17) + ((addr_mode) << 15) + ((dst) << 0)))
#define TT_CONV3S2_VALID(clear_dvalid, rotate_weights, addr_mode, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(rotate_weights, 5) && ckernel::is_valid(addr_mode, 2) && ckernel::is_valid(dst, 15))
#define TT_CONV3S2(clear_dvalid, rotate_weights, addr_mode, dst)  ckernel::instrn_buffer[0] = TT_OP_CONV3S2(clear_dvalid, rotate_weights, addr_mode, dst)
#define TTI_CONV3S2(clear_dvalid, rotate_weights, addr_mode, dst) INSTRUCTION_WORD(TT_OP_CONV3S2(clear_dvalid, rotate_weights, addr_mode, dst))

#define TT_OP_DMANOP TT_OP(0x60, 0)
#define TTI_DMANOP   INSTRUCTION_WORD(TT_OP_DMANOP)

#define TT_OP_DOTPV(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_OP(0x29, (((clear_dvalid) << 22) + ((dest_accum_en) << 21) + ((instr_mod19) << 19) + ((addr_mode) << 15) + ((dst) << 0)))
#define TT_DOTPV_VALID(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)                                                                          \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(dest_accum_en, 1) && ckernel::is_valid(instr_mod19, 2) && ckernel::is_valid(addr_mode, 4) && \
     ckernel::is_valid(dst, 15))
#define TT_DOTPV(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    ckernel::instrn_buffer[0] = TT_OP_DOTPV(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)
#define TTI_DOTPV(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    INSTRUCTION_WORD(TT_OP_DOTPV(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst))

#define TT_OP_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_OP(0x28, (((clear_dvalid) << 22) + ((dest_accum_en) << 21) + ((instr_mod19) << 19) + ((addr_mode) << 15) + ((dst) << 0)))
#define TT_ELWADD_VALID(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)                                                                         \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(dest_accum_en, 1) && ckernel::is_valid(instr_mod19, 2) && ckernel::is_valid(addr_mode, 4) && \
     ckernel::is_valid(dst, 15))
#define TT_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    ckernel::instrn_buffer[0] = TT_OP_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)
#define TTI_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    INSTRUCTION_WORD(TT_OP_ELWADD(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst))

#define TT_OP_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_OP(0x27, (((clear_dvalid) << 22) + ((dest_accum_en) << 21) + ((instr_mod19) << 19) + ((addr_mode) << 15) + ((dst) << 0)))
#define TT_ELWMUL_VALID(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)                                                                         \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(dest_accum_en, 1) && ckernel::is_valid(instr_mod19, 2) && ckernel::is_valid(addr_mode, 4) && \
     ckernel::is_valid(dst, 15))
#define TT_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    ckernel::instrn_buffer[0] = TT_OP_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)
#define TTI_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    INSTRUCTION_WORD(TT_OP_ELWMUL(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst))

#define TT_OP_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    TT_OP(0x30, (((clear_dvalid) << 22) + ((dest_accum_en) << 21) + ((instr_mod19) << 19) + ((addr_mode) << 15) + ((dst) << 0)))
#define TT_ELWSUB_VALID(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)                                                                         \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(dest_accum_en, 1) && ckernel::is_valid(instr_mod19, 2) && ckernel::is_valid(addr_mode, 4) && \
     ckernel::is_valid(dst, 15))
#define TT_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    ckernel::instrn_buffer[0] = TT_OP_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst)
#define TTI_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst) \
    INSTRUCTION_WORD(TT_OP_ELWSUB(clear_dvalid, dest_accum_en, instr_mod19, addr_mode, dst))

#define TT_OP_FLUSHDMA(FlushSpec)    TT_OP(0x46, (((FlushSpec) << 0)))
#define TT_FLUSHDMA_VALID(FlushSpec) (ckernel::is_valid(FlushSpec, 24))
#define TT_FLUSHDMA(FlushSpec)       ckernel::instrn_buffer[0] = TT_OP_FLUSHDMA(FlushSpec)
#define TTI_FLUSHDMA(FlushSpec)      INSTRUCTION_WORD(TT_OP_FLUSHDMA(FlushSpec))

#define TT_OP_GAPOOL(clear_dvalid, instr_mod19, addr_mode, max_pool_index_en, dst) \
    TT_OP(0x34, (((clear_dvalid) << 22) + ((instr_mod19) << 19) + ((addr_mode) << 15) + ((max_pool_index_en) << 14) + ((dst) << 0)))
#define TT_GAPOOL_VALID(clear_dvalid, instr_mod19, addr_mode, max_pool_index_en, dst)                                                                         \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(instr_mod19, 3) && ckernel::is_valid(addr_mode, 4) && ckernel::is_valid(max_pool_index_en, 1) && \
     ckernel::is_valid(dst, 14))
#define TT_GAPOOL(clear_dvalid, instr_mod19, addr_mode, max_pool_index_en, dst) \
    ckernel::instrn_buffer[0] = TT_OP_GAPOOL(clear_dvalid, instr_mod19, addr_mode, max_pool_index_en, dst)
#define TTI_GAPOOL(clear_dvalid, instr_mod19, addr_mode, max_pool_index_en, dst) \
    INSTRUCTION_WORD(TT_OP_GAPOOL(clear_dvalid, instr_mod19, addr_mode, max_pool_index_en, dst))

#define TT_OP_GATESRCRST(reset_srcb_gate_control, reset_srca_gate_control) TT_OP(0x35, (((reset_srcb_gate_control) << 1) + ((reset_srca_gate_control) << 0)))
#define TT_GATESRCRST_VALID(reset_srcb_gate_control, reset_srca_gate_control) \
    (ckernel::is_valid(reset_srcb_gate_control, 23) && ckernel::is_valid(reset_srca_gate_control, 1))
#define TT_GATESRCRST(reset_srcb_gate_control, reset_srca_gate_control) \
    ckernel::instrn_buffer[0] = TT_OP_GATESRCRST(reset_srcb_gate_control, reset_srca_gate_control)
#define TTI_GATESRCRST(reset_srcb_gate_control, reset_srca_gate_control) INSTRUCTION_WORD(TT_OP_GATESRCRST(reset_srcb_gate_control, reset_srca_gate_control))

#define TT_OP_GMPOOL(clear_dvalid, instr_mod19, addr_mode, max_pool_index_en, dst) \
    TT_OP(0x33, (((clear_dvalid) << 22) + ((instr_mod19) << 19) + ((addr_mode) << 15) + ((max_pool_index_en) << 14) + ((dst) << 0)))
#define TT_GMPOOL_VALID(clear_dvalid, instr_mod19, addr_mode, max_pool_index_en, dst)                                                                         \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(instr_mod19, 3) && ckernel::is_valid(addr_mode, 4) && ckernel::is_valid(max_pool_index_en, 1) && \
     ckernel::is_valid(dst, 14))
#define TT_GMPOOL(clear_dvalid, instr_mod19, addr_mode, max_pool_index_en, dst) \
    ckernel::instrn_buffer[0] = TT_OP_GMPOOL(clear_dvalid, instr_mod19, addr_mode, max_pool_index_en, dst)
#define TTI_GMPOOL(clear_dvalid, instr_mod19, addr_mode, max_pool_index_en, dst) \
    INSTRUCTION_WORD(TT_OP_GMPOOL(clear_dvalid, instr_mod19, addr_mode, max_pool_index_en, dst))

#define TT_OP_INCADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X) \
    TT_OP(0x52, (((CntSetMask) << 21) + ((Ch1_Y) << 15) + ((Ch1_X) << 12) + ((Ch0_Y) << 9) + ((Ch0_X) << 6)))
#define TT_INCADCXY_VALID(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)                                                                     \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(Ch1_Y, 6) && ckernel::is_valid(Ch1_X, 3) && ckernel::is_valid(Ch0_Y, 3) && \
     ckernel::is_valid(Ch0_X, 3))
#define TT_INCADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)  ckernel::instrn_buffer[0] = TT_OP_INCADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)
#define TTI_INCADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X) INSTRUCTION_WORD(TT_OP_INCADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X))

#define TT_OP_INCADCZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X) \
    TT_OP(0x55, (((CntSetMask) << 21) + ((Ch1_Y) << 15) + ((Ch1_X) << 12) + ((Ch0_Y) << 9) + ((Ch0_X) << 6)))
#define TT_INCADCZW_VALID(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)                                                                     \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(Ch1_Y, 6) && ckernel::is_valid(Ch1_X, 3) && ckernel::is_valid(Ch0_Y, 3) && \
     ckernel::is_valid(Ch0_X, 3))
#define TT_INCADCZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)  ckernel::instrn_buffer[0] = TT_OP_INCADCZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X)
#define TTI_INCADCZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X) INSTRUCTION_WORD(TT_OP_INCADCZW(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X))

#define TT_OP_INCRWC(rwc_cr, rwc_d, rwc_b, rwc_a) TT_OP(0x38, (((rwc_cr) << 18) + ((rwc_d) << 14) + ((rwc_b) << 10) + ((rwc_a) << 6)))
#define TT_INCRWC_VALID(rwc_cr, rwc_d, rwc_b, rwc_a) \
    (ckernel::is_valid(rwc_cr, 6) && ckernel::is_valid(rwc_d, 4) && ckernel::is_valid(rwc_b, 4) && ckernel::is_valid(rwc_a, 4))
#define TT_INCRWC(rwc_cr, rwc_d, rwc_b, rwc_a)  ckernel::instrn_buffer[0] = TT_OP_INCRWC(rwc_cr, rwc_d, rwc_b, rwc_a)
#define TTI_INCRWC(rwc_cr, rwc_d, rwc_b, rwc_a) INSTRUCTION_WORD(TT_OP_INCRWC(rwc_cr, rwc_d, rwc_b, rwc_a))

#define TT_OP_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex) \
    TT_OP(0x49, (((SizeSel) << 22) + ((OffsetIndex) << 14) + ((AutoIncSpec) << 12) + ((DataRegIndex) << 6) + ((AddrRegIndex) << 0)))
#define TT_LOADIND_VALID(SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex)                                                               \
    (ckernel::is_valid(SizeSel, 2) && ckernel::is_valid(OffsetIndex, 8) && ckernel::is_valid(AutoIncSpec, 2) && ckernel::is_valid(DataRegIndex, 6) && \
     ckernel::is_valid(AddrRegIndex, 6))
#define TT_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex) \
    ckernel::instrn_buffer[0] = TT_OP_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex)
#define TTI_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex) \
    INSTRUCTION_WORD(TT_OP_LOADIND(SizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex))

#define TT_OP_LOADREG(TdmaDataRegIndex, RegAddr)    TT_OP(0x68, (((TdmaDataRegIndex) << 18) + ((RegAddr) << 0)))
#define TT_LOADREG_VALID(TdmaDataRegIndex, RegAddr) (ckernel::is_valid(TdmaDataRegIndex, 6) && ckernel::is_valid(RegAddr, 18))
#define TT_LOADREG(TdmaDataRegIndex, RegAddr)       ckernel::instrn_buffer[0] = TT_OP_LOADREG(TdmaDataRegIndex, RegAddr)
#define TTI_LOADREG(TdmaDataRegIndex, RegAddr)      INSTRUCTION_WORD(TT_OP_LOADREG(TdmaDataRegIndex, RegAddr))

#define TT_OP_MFCONV3S1(clear_dvalid, rotate_weights, addr_mode, dst) \
    TT_OP(0x3a, (((clear_dvalid) << 22) + ((rotate_weights) << 17) + ((addr_mode) << 15) + ((dst) << 0)))
#define TT_MFCONV3S1_VALID(clear_dvalid, rotate_weights, addr_mode, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(rotate_weights, 5) && ckernel::is_valid(addr_mode, 2) && ckernel::is_valid(dst, 15))
#define TT_MFCONV3S1(clear_dvalid, rotate_weights, addr_mode, dst)  ckernel::instrn_buffer[0] = TT_OP_MFCONV3S1(clear_dvalid, rotate_weights, addr_mode, dst)
#define TTI_MFCONV3S1(clear_dvalid, rotate_weights, addr_mode, dst) INSTRUCTION_WORD(TT_OP_MFCONV3S1(clear_dvalid, rotate_weights, addr_mode, dst))

#define TT_OP_MOP(mop_type, loop_count, zmask_lo16)    TT_OP(0x01, (((mop_type) << 23) + ((loop_count) << 16) + ((zmask_lo16) << 0)))
#define TT_MOP_VALID(mop_type, loop_count, zmask_lo16) (ckernel::is_valid(mop_type, 1) && ckernel::is_valid(loop_count, 7) && ckernel::is_valid(zmask_lo16, 16))
#define TT_MOP(mop_type, loop_count, zmask_lo16)       ckernel::instrn_buffer[0] = TT_OP_MOP(mop_type, loop_count, zmask_lo16)
#define TTI_MOP(mop_type, loop_count, zmask_lo16)      INSTRUCTION_WORD(TT_OP_MOP(mop_type, loop_count, zmask_lo16))

#define TT_OP_MOP_CFG(zmask_hi16)    TT_OP(0x03, (((zmask_hi16) << 0)))
#define TT_MOP_CFG_VALID(zmask_hi16) (ckernel::is_valid(zmask_hi16, 24))
#define TT_MOP_CFG(zmask_hi16)       ckernel::instrn_buffer[0] = TT_OP_MOP_CFG(zmask_hi16)
#define TTI_MOP_CFG(zmask_hi16)      INSTRUCTION_WORD(TT_OP_MOP_CFG(zmask_hi16))

#define TT_OP_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst) \
    TT_OP(0x12, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 15) + ((instr_mod) << 12) + ((dst) << 0)))
#define TT_MOVA2D_VALID(dest_32b_lo, src, addr_mode, instr_mod, dst)                                                                         \
    (ckernel::is_valid(dest_32b_lo, 1) && ckernel::is_valid(src, 6) && ckernel::is_valid(addr_mode, 2) && ckernel::is_valid(instr_mod, 2) && \
     ckernel::is_valid(dst, 12))
#define TT_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst)  ckernel::instrn_buffer[0] = TT_OP_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst)
#define TTI_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst) INSTRUCTION_WORD(TT_OP_MOVA2D(dest_32b_lo, src, addr_mode, instr_mod, dst))

#define TT_OP_MOVB2A(srca, addr_mode, instr_mod, srcb) TT_OP(0x0b, (((srca) << 17) + ((addr_mode) << 15) + ((instr_mod) << 12) + ((srcb) << 0)))
#define TT_MOVB2A_VALID(srca, addr_mode, instr_mod, srcb) \
    (ckernel::is_valid(srca, 7) && ckernel::is_valid(addr_mode, 2) && ckernel::is_valid(instr_mod, 2) && ckernel::is_valid(srcb, 12))
#define TT_MOVB2A(srca, addr_mode, instr_mod, srcb)  ckernel::instrn_buffer[0] = TT_OP_MOVB2A(srca, addr_mode, instr_mod, srcb)
#define TTI_MOVB2A(srca, addr_mode, instr_mod, srcb) INSTRUCTION_WORD(TT_OP_MOVB2A(srca, addr_mode, instr_mod, srcb))

#define TT_OP_MOVB2D(dest_32b_lo, src, addr_mode, instr_mod, dst) \
    TT_OP(0x13, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 15) + ((instr_mod) << 12) + ((dst) << 0)))
#define TT_MOVB2D_VALID(dest_32b_lo, src, addr_mode, instr_mod, dst)                                                                         \
    (ckernel::is_valid(dest_32b_lo, 1) && ckernel::is_valid(src, 6) && ckernel::is_valid(addr_mode, 2) && ckernel::is_valid(instr_mod, 3) && \
     ckernel::is_valid(dst, 12))
#define TT_MOVB2D(dest_32b_lo, src, addr_mode, instr_mod, dst)  ckernel::instrn_buffer[0] = TT_OP_MOVB2D(dest_32b_lo, src, addr_mode, instr_mod, dst)
#define TTI_MOVB2D(dest_32b_lo, src, addr_mode, instr_mod, dst) INSTRUCTION_WORD(TT_OP_MOVB2D(dest_32b_lo, src, addr_mode, instr_mod, dst))

#define TT_OP_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst) \
    TT_OP(0x08, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 15) + ((instr_mod) << 12) + ((dst) << 0)))
#define TT_MOVD2A_VALID(dest_32b_lo, src, addr_mode, instr_mod, dst)                                                                         \
    (ckernel::is_valid(dest_32b_lo, 1) && ckernel::is_valid(src, 6) && ckernel::is_valid(addr_mode, 2) && ckernel::is_valid(instr_mod, 2) && \
     ckernel::is_valid(dst, 12))
#define TT_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst)  ckernel::instrn_buffer[0] = TT_OP_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst)
#define TTI_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst) INSTRUCTION_WORD(TT_OP_MOVD2A(dest_32b_lo, src, addr_mode, instr_mod, dst))

#define TT_OP_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, dst) \
    TT_OP(0x0a, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 15) + ((instr_mod) << 12) + ((dst) << 0)))
#define TT_MOVD2B_VALID(dest_32b_lo, src, addr_mode, instr_mod, dst)                                                                         \
    (ckernel::is_valid(dest_32b_lo, 1) && ckernel::is_valid(src, 6) && ckernel::is_valid(addr_mode, 2) && ckernel::is_valid(instr_mod, 2) && \
     ckernel::is_valid(dst, 12))
#define TT_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, dst)  ckernel::instrn_buffer[0] = TT_OP_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, dst)
#define TTI_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, dst) INSTRUCTION_WORD(TT_OP_MOVD2B(dest_32b_lo, src, addr_mode, instr_mod, dst))

#define TT_OP_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst) \
    TT_OP(0x09, (((dest_32b_lo) << 23) + ((src) << 17) + ((addr_mode) << 15) + ((instr_mod) << 12) + ((dst) << 0)))
#define TT_MOVDBGA2D_VALID(dest_32b_lo, src, addr_mode, instr_mod, dst)                                                                      \
    (ckernel::is_valid(dest_32b_lo, 1) && ckernel::is_valid(src, 6) && ckernel::is_valid(addr_mode, 2) && ckernel::is_valid(instr_mod, 3) && \
     ckernel::is_valid(dst, 12))
#define TT_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst)  ckernel::instrn_buffer[0] = TT_OP_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst)
#define TTI_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst) INSTRUCTION_WORD(TT_OP_MOVDBGA2D(dest_32b_lo, src, addr_mode, instr_mod, dst))

#define TT_OP_MPOOL3S1(clear_dvalid, addr_mode, index_en, dst) TT_OP(0x24, (((clear_dvalid) << 22) + ((addr_mode) << 15) + ((index_en) << 14) + ((dst) << 0)))
#define TT_MPOOL3S1_VALID(clear_dvalid, addr_mode, index_en, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(addr_mode, 7) && ckernel::is_valid(index_en, 1) && ckernel::is_valid(dst, 14))
#define TT_MPOOL3S1(clear_dvalid, addr_mode, index_en, dst)  ckernel::instrn_buffer[0] = TT_OP_MPOOL3S1(clear_dvalid, addr_mode, index_en, dst)
#define TTI_MPOOL3S1(clear_dvalid, addr_mode, index_en, dst) INSTRUCTION_WORD(TT_OP_MPOOL3S1(clear_dvalid, addr_mode, index_en, dst))

#define TT_OP_MPOOL3S2(clear_dvalid, addr_mode, index_en, dst) TT_OP(0x31, (((clear_dvalid) << 22) + ((addr_mode) << 15) + ((index_en) << 14) + ((dst) << 0)))
#define TT_MPOOL3S2_VALID(clear_dvalid, addr_mode, index_en, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(addr_mode, 7) && ckernel::is_valid(index_en, 1) && ckernel::is_valid(dst, 14))
#define TT_MPOOL3S2(clear_dvalid, addr_mode, index_en, dst)  ckernel::instrn_buffer[0] = TT_OP_MPOOL3S2(clear_dvalid, addr_mode, index_en, dst)
#define TTI_MPOOL3S2(clear_dvalid, addr_mode, index_en, dst) INSTRUCTION_WORD(TT_OP_MPOOL3S2(clear_dvalid, addr_mode, index_en, dst))

#define TT_OP_MULDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_OP(0x5a, (((OpBisConst) << 23) + ((ResultRegIndex) << 12) + ((OpBRegIndex) << 6) + ((OpARegIndex) << 0)))
#define TT_MULDMAREG_VALID(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    (ckernel::is_valid(OpBisConst, 1) && ckernel::is_valid(ResultRegIndex, 6) && ckernel::is_valid(OpBRegIndex, 6) && ckernel::is_valid(OpARegIndex, 6))
#define TT_MULDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    ckernel::instrn_buffer[0] = TT_OP_MULDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex)
#define TTI_MULDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    INSTRUCTION_WORD(TT_OP_MULDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex))

#define TT_OP_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst) \
    TT_OP(0x26, (((clear_dvalid) << 22) + ((instr_mod19) << 19) + ((addr_mode) << 15) + ((dst) << 0)))
#define TT_MVMUL_VALID(clear_dvalid, instr_mod19, addr_mode, dst) \
    (ckernel::is_valid(clear_dvalid, 2) && ckernel::is_valid(instr_mod19, 3) && ckernel::is_valid(addr_mode, 4) && ckernel::is_valid(dst, 15))
#define TT_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst)  ckernel::instrn_buffer[0] = TT_OP_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst)
#define TTI_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst) INSTRUCTION_WORD(TT_OP_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst))

#define TT_OP_NOP TT_OP(0x02, 0)
#define TTI_NOP   INSTRUCTION_WORD(TT_OP_NOP)

#define TT_OP_PACR(AddrMode, ZeroWrite, PackSel, OvrdThreadId, Concat, Flush, Last) \
    TT_OP(0x41, (((AddrMode) << 15) + ((ZeroWrite) << 12) + ((PackSel) << 8) + ((OvrdThreadId) << 7) + ((Concat) << 4) + ((Flush) << 1) + ((Last) << 0)))
#define TT_PACR_VALID(AddrMode, ZeroWrite, PackSel, OvrdThreadId, Concat, Flush, Last)                                                           \
    (ckernel::is_valid(AddrMode, 9) && ckernel::is_valid(ZeroWrite, 3) && ckernel::is_valid(PackSel, 4) && ckernel::is_valid(OvrdThreadId, 1) && \
     ckernel::is_valid(Concat, 3) && ckernel::is_valid(Flush, 3) && ckernel::is_valid(Last, 1))
#define TT_PACR(AddrMode, ZeroWrite, PackSel, OvrdThreadId, Concat, Flush, Last) \
    ckernel::instrn_buffer[0] = TT_OP_PACR(AddrMode, ZeroWrite, PackSel, OvrdThreadId, Concat, Flush, Last)
#define TTI_PACR(AddrMode, ZeroWrite, PackSel, OvrdThreadId, Concat, Flush, Last) \
    INSTRUCTION_WORD(TT_OP_PACR(AddrMode, ZeroWrite, PackSel, OvrdThreadId, Concat, Flush, Last))

#define TT_OP_PACR_SETREG(Push, AddrSel, WrData, PackSel, StreamId, Flush, Last) \
    TT_OP(0x4a, (((Push) << 23) + ((AddrSel) << 22) + ((WrData) << 12) + ((PackSel) << 8) + ((StreamId) << 2) + ((Flush) << 1) + ((Last) << 0)))
#define TT_PACR_SETREG_VALID(Push, AddrSel, WrData, PackSel, StreamId, Flush, Last)                                                   \
    (ckernel::is_valid(Push, 1) && ckernel::is_valid(AddrSel, 1) && ckernel::is_valid(WrData, 10) && ckernel::is_valid(PackSel, 4) && \
     ckernel::is_valid(StreamId, 6) && ckernel::is_valid(Flush, 1) && ckernel::is_valid(Last, 1))
#define TT_PACR_SETREG(Push, AddrSel, WrData, PackSel, StreamId, Flush, Last) \
    ckernel::instrn_buffer[0] = TT_OP_PACR_SETREG(Push, AddrSel, WrData, PackSel, StreamId, Flush, Last)
#define TTI_PACR_SETREG(Push, AddrSel, WrData, PackSel, StreamId, Flush, Last) \
    INSTRUCTION_WORD(TT_OP_PACR_SETREG(Push, AddrSel, WrData, PackSel, StreamId, Flush, Last))

#define TT_OP_RAREB TT_OP(0x15, 0)
#define TTI_RAREB   INSTRUCTION_WORD(TT_OP_RAREB)

#define TT_OP_RDCFG(GprAddress, CfgReg)    TT_OP(0xb1, (((GprAddress) << 16) + ((CfgReg) << 0)))
#define TT_RDCFG_VALID(GprAddress, CfgReg) (ckernel::is_valid(GprAddress, 8) && ckernel::is_valid(CfgReg, 16))
#define TT_RDCFG(GprAddress, CfgReg)       ckernel::instrn_buffer[0] = TT_OP_RDCFG(GprAddress, CfgReg)
#define TTI_RDCFG(GprAddress, CfgReg)      INSTRUCTION_WORD(TT_OP_RDCFG(GprAddress, CfgReg))

#define TT_OP_REG2FLOP(SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex) \
    TT_OP(0x48, (((SizeSel) << 22) + ((TargetSel) << 20) + ((ByteOffset) << 18) + ((ContextId_2) << 16) + ((FlopIndex) << 6) + ((RegIndex) << 0)))
#define TT_REG2FLOP_VALID(SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex)                                                       \
    (ckernel::is_valid(SizeSel, 2) && ckernel::is_valid(TargetSel, 2) && ckernel::is_valid(ByteOffset, 2) && ckernel::is_valid(ContextId_2, 2) && \
     ckernel::is_valid(FlopIndex, 10) && ckernel::is_valid(RegIndex, 6))
#define TT_REG2FLOP(SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex) \
    ckernel::instrn_buffer[0] = TT_OP_REG2FLOP(SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex)
#define TTI_REG2FLOP(SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex) \
    INSTRUCTION_WORD(TT_OP_REG2FLOP(SizeSel, TargetSel, ByteOffset, ContextId_2, FlopIndex, RegIndex))

#define TT_OP_REPLAY(start_idx, len, execute_while_loading, load_mode) \
    TT_OP(0x04, (((start_idx) << 14) + ((len) << 4) + ((execute_while_loading) << 1) + ((load_mode) << 0)))
#define TT_REPLAY_VALID(start_idx, len, execute_while_loading, load_mode) \
    (ckernel::is_valid(start_idx, 10) && ckernel::is_valid(len, 10) && ckernel::is_valid(execute_while_loading, 3) && ckernel::is_valid(load_mode, 1))
#define TT_REPLAY(start_idx, len, execute_while_loading, load_mode)  ckernel::instrn_buffer[0] = TT_OP_REPLAY(start_idx, len, execute_while_loading, load_mode)
#define TTI_REPLAY(start_idx, len, execute_while_loading, load_mode) INSTRUCTION_WORD(TT_OP_REPLAY(start_idx, len, execute_while_loading, load_mode))

#define TT_OP_RMWCIB0(Mask, Data, CfgRegAddr)    TT_OP(0xb3, (((Mask) << 16) + ((Data) << 8) + ((CfgRegAddr) << 0)))
#define TT_RMWCIB0_VALID(Mask, Data, CfgRegAddr) (ckernel::is_valid(Mask, 8) && ckernel::is_valid(Data, 8) && ckernel::is_valid(CfgRegAddr, 8))
#define TT_RMWCIB0(Mask, Data, CfgRegAddr)       ckernel::instrn_buffer[0] = TT_OP_RMWCIB0(Mask, Data, CfgRegAddr)
#define TTI_RMWCIB0(Mask, Data, CfgRegAddr)      INSTRUCTION_WORD(TT_OP_RMWCIB0(Mask, Data, CfgRegAddr))

#define TT_OP_RMWCIB1(Mask, Data, CfgRegAddr)    TT_OP(0xb4, (((Mask) << 16) + ((Data) << 8) + ((CfgRegAddr) << 0)))
#define TT_RMWCIB1_VALID(Mask, Data, CfgRegAddr) (ckernel::is_valid(Mask, 8) && ckernel::is_valid(Data, 8) && ckernel::is_valid(CfgRegAddr, 8))
#define TT_RMWCIB1(Mask, Data, CfgRegAddr)       ckernel::instrn_buffer[0] = TT_OP_RMWCIB1(Mask, Data, CfgRegAddr)
#define TTI_RMWCIB1(Mask, Data, CfgRegAddr)      INSTRUCTION_WORD(TT_OP_RMWCIB1(Mask, Data, CfgRegAddr))

#define TT_OP_RMWCIB2(Mask, Data, CfgRegAddr)    TT_OP(0xb5, (((Mask) << 16) + ((Data) << 8) + ((CfgRegAddr) << 0)))
#define TT_RMWCIB2_VALID(Mask, Data, CfgRegAddr) (ckernel::is_valid(Mask, 8) && ckernel::is_valid(Data, 8) && ckernel::is_valid(CfgRegAddr, 8))
#define TT_RMWCIB2(Mask, Data, CfgRegAddr)       ckernel::instrn_buffer[0] = TT_OP_RMWCIB2(Mask, Data, CfgRegAddr)
#define TTI_RMWCIB2(Mask, Data, CfgRegAddr)      INSTRUCTION_WORD(TT_OP_RMWCIB2(Mask, Data, CfgRegAddr))

#define TT_OP_RMWCIB3(Mask, Data, CfgRegAddr)    TT_OP(0xb6, (((Mask) << 16) + ((Data) << 8) + ((CfgRegAddr) << 0)))
#define TT_RMWCIB3_VALID(Mask, Data, CfgRegAddr) (ckernel::is_valid(Mask, 8) && ckernel::is_valid(Data, 8) && ckernel::is_valid(CfgRegAddr, 8))
#define TT_RMWCIB3(Mask, Data, CfgRegAddr)       ckernel::instrn_buffer[0] = TT_OP_RMWCIB3(Mask, Data, CfgRegAddr)
#define TTI_RMWCIB3(Mask, Data, CfgRegAddr)      INSTRUCTION_WORD(TT_OP_RMWCIB3(Mask, Data, CfgRegAddr))

#define TT_OP_RSTDMA TT_OP(0x44, 0)
#define TTI_RSTDMA   INSTRUCTION_WORD(TT_OP_RSTDMA)

#define TT_OP_SEMGET(sem_sel)    TT_OP(0xa5, (((sem_sel) << 2)))
#define TT_SEMGET_VALID(sem_sel) (ckernel::is_valid(sem_sel, 22))
#define TT_SEMGET(sem_sel)       ckernel::instrn_buffer[0] = TT_OP_SEMGET(sem_sel)
#define TTI_SEMGET(sem_sel)      INSTRUCTION_WORD(TT_OP_SEMGET(sem_sel))

#define TT_OP_SEMINIT(max_value, init_value, sem_sel)    TT_OP(0xa3, (((max_value) << 20) + ((init_value) << 16) + ((sem_sel) << 2)))
#define TT_SEMINIT_VALID(max_value, init_value, sem_sel) (ckernel::is_valid(max_value, 4) && ckernel::is_valid(init_value, 4) && ckernel::is_valid(sem_sel, 14))
#define TT_SEMINIT(max_value, init_value, sem_sel)       ckernel::instrn_buffer[0] = TT_OP_SEMINIT(max_value, init_value, sem_sel)
#define TTI_SEMINIT(max_value, init_value, sem_sel)      INSTRUCTION_WORD(TT_OP_SEMINIT(max_value, init_value, sem_sel))

#define TT_OP_SEMPOST(sem_sel)    TT_OP(0xa4, (((sem_sel) << 2)))
#define TT_SEMPOST_VALID(sem_sel) (ckernel::is_valid(sem_sel, 22))
#define TT_SEMPOST(sem_sel)       ckernel::instrn_buffer[0] = TT_OP_SEMPOST(sem_sel)
#define TTI_SEMPOST(sem_sel)      INSTRUCTION_WORD(TT_OP_SEMPOST(sem_sel))

#define TT_OP_SEMWAIT(stall_res, sem_sel, wait_sem_cond) TT_OP(0xa6, (((stall_res) << 15) + ((sem_sel) << 2) + ((wait_sem_cond) << 0)))
#define TT_SEMWAIT_VALID(stall_res, sem_sel, wait_sem_cond) \
    (ckernel::is_valid(stall_res, 9) && ckernel::is_valid(sem_sel, 13) && ckernel::is_valid(wait_sem_cond, 2))
#define TT_SEMWAIT(stall_res, sem_sel, wait_sem_cond)  ckernel::instrn_buffer[0] = TT_OP_SEMWAIT(stall_res, sem_sel, wait_sem_cond)
#define TTI_SEMWAIT(stall_res, sem_sel, wait_sem_cond) INSTRUCTION_WORD(TT_OP_SEMWAIT(stall_res, sem_sel, wait_sem_cond))

#define TT_OP_SETADC(CntSetMask, ChannelIndex, DimensionIndex, Value) \
    TT_OP(0x50, (((CntSetMask) << 21) + ((ChannelIndex) << 20) + ((DimensionIndex) << 18) + ((Value) << 0)))
#define TT_SETADC_VALID(CntSetMask, ChannelIndex, DimensionIndex, Value) \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(ChannelIndex, 1) && ckernel::is_valid(DimensionIndex, 2) && ckernel::is_valid(Value, 18))
#define TT_SETADC(CntSetMask, ChannelIndex, DimensionIndex, Value)  ckernel::instrn_buffer[0] = TT_OP_SETADC(CntSetMask, ChannelIndex, DimensionIndex, Value)
#define TTI_SETADC(CntSetMask, ChannelIndex, DimensionIndex, Value) INSTRUCTION_WORD(TT_OP_SETADC(CntSetMask, ChannelIndex, DimensionIndex, Value))

#define TT_OP_SETADCXX(CntSetMask, x_end2, x_start)    TT_OP(0x5e, (((CntSetMask) << 21) + ((x_end2) << 10) + ((x_start) << 0)))
#define TT_SETADCXX_VALID(CntSetMask, x_end2, x_start) (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(x_end2, 11) && ckernel::is_valid(x_start, 10))
#define TT_SETADCXX(CntSetMask, x_end2, x_start)       ckernel::instrn_buffer[0] = TT_OP_SETADCXX(CntSetMask, x_end2, x_start)
#define TTI_SETADCXX(CntSetMask, x_end2, x_start)      INSTRUCTION_WORD(TT_OP_SETADCXX(CntSetMask, x_end2, x_start))

#define TT_OP_SETADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) \
    TT_OP(0x51, (((CntSetMask) << 21) + ((Ch1_Y) << 15) + ((Ch1_X) << 12) + ((Ch0_Y) << 9) + ((Ch0_X) << 6) + ((BitMask) << 0)))
#define TT_SETADCXY_VALID(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)                                                            \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(Ch1_Y, 6) && ckernel::is_valid(Ch1_X, 3) && ckernel::is_valid(Ch0_Y, 3) && \
     ckernel::is_valid(Ch0_X, 3) && ckernel::is_valid(BitMask, 6))
#define TT_SETADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)  ckernel::instrn_buffer[0] = TT_OP_SETADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask)
#define TTI_SETADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) INSTRUCTION_WORD(TT_OP_SETADCXY(CntSetMask, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask))

#define TT_OP_SETADCZW(CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask) \
    TT_OP(0x54, (((CntSetMask) << 21) + ((Ch1_W) << 15) + ((Ch1_Z) << 12) + ((Ch0_W) << 9) + ((Ch0_Z) << 6) + ((BitMask) << 0)))
#define TT_SETADCZW_VALID(CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask)                                                            \
    (ckernel::is_valid(CntSetMask, 3) && ckernel::is_valid(Ch1_W, 6) && ckernel::is_valid(Ch1_Z, 3) && ckernel::is_valid(Ch0_W, 3) && \
     ckernel::is_valid(Ch0_Z, 3) && ckernel::is_valid(BitMask, 6))
#define TT_SETADCZW(CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask)  ckernel::instrn_buffer[0] = TT_OP_SETADCZW(CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask)
#define TTI_SETADCZW(CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask) INSTRUCTION_WORD(TT_OP_SETADCZW(CntSetMask, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask))

#define TT_OP_SETASHRMH(reg_mask, halo_mask)    TT_OP(0x1e, (((reg_mask) << 1) + ((halo_mask) << 0)))
#define TT_SETASHRMH_VALID(reg_mask, halo_mask) (ckernel::is_valid(reg_mask, 23) && ckernel::is_valid(halo_mask, 1))
#define TT_SETASHRMH(reg_mask, halo_mask)       ckernel::instrn_buffer[0] = TT_OP_SETASHRMH(reg_mask, halo_mask)
#define TTI_SETASHRMH(reg_mask, halo_mask)      INSTRUCTION_WORD(TT_OP_SETASHRMH(reg_mask, halo_mask))

#define TT_OP_SETASHRMH0(reg_mask, halo_mask)    TT_OP(0x1a, (((reg_mask) << 1) + ((halo_mask) << 0)))
#define TT_SETASHRMH0_VALID(reg_mask, halo_mask) (ckernel::is_valid(reg_mask, 23) && ckernel::is_valid(halo_mask, 1))
#define TT_SETASHRMH0(reg_mask, halo_mask)       ckernel::instrn_buffer[0] = TT_OP_SETASHRMH0(reg_mask, halo_mask)
#define TTI_SETASHRMH0(reg_mask, halo_mask)      INSTRUCTION_WORD(TT_OP_SETASHRMH0(reg_mask, halo_mask))

#define TT_OP_SETASHRMH1(reg_mask, halo_mask)    TT_OP(0x1b, (((reg_mask) << 1) + ((halo_mask) << 0)))
#define TT_SETASHRMH1_VALID(reg_mask, halo_mask) (ckernel::is_valid(reg_mask, 23) && ckernel::is_valid(halo_mask, 1))
#define TT_SETASHRMH1(reg_mask, halo_mask)       ckernel::instrn_buffer[0] = TT_OP_SETASHRMH1(reg_mask, halo_mask)
#define TTI_SETASHRMH1(reg_mask, halo_mask)      INSTRUCTION_WORD(TT_OP_SETASHRMH1(reg_mask, halo_mask))

#define TT_OP_SETASHRMV(reg_mask2)    TT_OP(0x1c, (((reg_mask2) << 0)))
#define TT_SETASHRMV_VALID(reg_mask2) (ckernel::is_valid(reg_mask2, 24))
#define TT_SETASHRMV(reg_mask2)       ckernel::instrn_buffer[0] = TT_OP_SETASHRMV(reg_mask2)
#define TTI_SETASHRMV(reg_mask2)      INSTRUCTION_WORD(TT_OP_SETASHRMV(reg_mask2))

#define TT_OP_SETC16(setc16_reg, setc16_value)    TT_OP(0xb2, (((setc16_reg) << 16) + ((setc16_value) << 0)))
#define TT_SETC16_VALID(setc16_reg, setc16_value) (ckernel::is_valid(setc16_reg, 8) && ckernel::is_valid(setc16_value, 16))
#define TT_SETC16(setc16_reg, setc16_value)       ckernel::instrn_buffer[0] = TT_OP_SETC16(setc16_reg, setc16_value)
#define TTI_SETC16(setc16_reg, setc16_value)      INSTRUCTION_WORD(TT_OP_SETC16(setc16_reg, setc16_value))

#define TT_OP_SETDMAREG(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b) \
    TT_OP(0x45, (((Payload_SigSelSize) << 22) + ((Payload_SigSel) << 8) + ((SetSignalsMode) << 7) + ((RegIndex16b) << 0)))
#define TT_SETDMAREG_VALID(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b)                                       \
    (ckernel::is_valid(Payload_SigSelSize, 2) && ckernel::is_valid(Payload_SigSel, 16) && ckernel::is_valid(SetSignalsMode, 1) && \
     ckernel::is_valid(RegIndex16b, 7))
#define TT_SETDMAREG(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b) \
    ckernel::instrn_buffer[0] = TT_OP_SETDMAREG(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b)
#define TTI_SETDMAREG(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b) \
    INSTRUCTION_WORD(TT_OP_SETDMAREG(Payload_SigSelSize, Payload_SigSel, SetSignalsMode, RegIndex16b))

#define TT_OP_SETDVALID(setvalid)    TT_OP(0x57, (((setvalid) << 0)))
#define TT_SETDVALID_VALID(setvalid) (ckernel::is_valid(setvalid, 16))
#define TT_SETDVALID(setvalid)       ckernel::instrn_buffer[0] = TT_OP_SETDVALID(setvalid)
#define TTI_SETDVALID(setvalid)      INSTRUCTION_WORD(TT_OP_SETDVALID(setvalid))

#define TT_OP_SETIBRWC(rwc_cr, rwc_bias, set_inc_ctrl) TT_OP(0x39, (((rwc_cr) << 18) + ((rwc_bias) << 6) + ((set_inc_ctrl) << 0)))
#define TT_SETIBRWC_VALID(rwc_cr, rwc_bias, set_inc_ctrl) \
    (ckernel::is_valid(rwc_cr, 4) && ckernel::is_valid(rwc_bias, 12) && ckernel::is_valid(set_inc_ctrl, 6))
#define TT_SETIBRWC(rwc_cr, rwc_bias, set_inc_ctrl)  ckernel::instrn_buffer[0] = TT_OP_SETIBRWC(rwc_cr, rwc_bias, set_inc_ctrl)
#define TTI_SETIBRWC(rwc_cr, rwc_bias, set_inc_ctrl) INSTRUCTION_WORD(TT_OP_SETIBRWC(rwc_cr, rwc_bias, set_inc_ctrl))

#define TT_OP_SETPKEDGOF(y_end, y_start, x_end, x_start) TT_OP(0x1d, (((y_end) << 12) + ((y_start) << 8) + ((x_end) << 4) + ((x_start) << 0)))
#define TT_SETPKEDGOF_VALID(y_end, y_start, x_end, x_start) \
    (ckernel::is_valid(y_end, 12) && ckernel::is_valid(y_start, 4) && ckernel::is_valid(x_end, 4) && ckernel::is_valid(x_start, 4))
#define TT_SETPKEDGOF(y_end, y_start, x_end, x_start)  ckernel::instrn_buffer[0] = TT_OP_SETPKEDGOF(y_end, y_start, x_end, x_start)
#define TTI_SETPKEDGOF(y_end, y_start, x_end, x_start) INSTRUCTION_WORD(TT_OP_SETPKEDGOF(y_end, y_start, x_end, x_start))

#define TT_OP_SETRWC(clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask) \
    TT_OP(0x37, (((clear_ab_vld) << 22) + ((rwc_cr) << 18) + ((rwc_d) << 14) + ((rwc_b) << 10) + ((rwc_a) << 6) + ((BitMask) << 0)))
#define TT_SETRWC_VALID(clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask)                                                              \
    (ckernel::is_valid(clear_ab_vld, 2) && ckernel::is_valid(rwc_cr, 4) && ckernel::is_valid(rwc_d, 4) && ckernel::is_valid(rwc_b, 4) && \
     ckernel::is_valid(rwc_a, 4) && ckernel::is_valid(BitMask, 6))
#define TT_SETRWC(clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask) \
    ckernel::instrn_buffer[0] = TT_OP_SETRWC(clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask)
#define TTI_SETRWC(clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask) INSTRUCTION_WORD(TT_OP_SETRWC(clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, BitMask))

#define TT_OP_SFPABS(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x7d, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPABS_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPABS(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPABS(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPABS(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPABS(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPADD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x85, (((lreg_src_a) << 16) + ((lreg_src_b) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPADD_VALID(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)                                                                  \
    (ckernel::is_valid(lreg_src_a, 8) && ckernel::is_valid(lreg_src_b, 4) && ckernel::is_valid(lreg_src_c, 4) && ckernel::is_valid(lreg_dest, 4) && \
     ckernel::is_valid(instr_mod1, 4))
#define TT_SFPADD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    ckernel::instrn_buffer[0] = TT_OP_SFPADD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPADD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    INSTRUCTION_WORD(TT_OP_SFPADD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1))

#define TT_OP_SFPADDI(imm16_math, lreg_dest, instr_mod1) TT_OP(0x75, (((imm16_math) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPADDI_VALID(imm16_math, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm16_math, 16) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPADDI(imm16_math, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPADDI(imm16_math, lreg_dest, instr_mod1)
#define TTI_SFPADDI(imm16_math, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPADDI(imm16_math, lreg_dest, instr_mod1))

#define TT_OP_SFPAND(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x7e, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPAND_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPAND(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPAND(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPAND(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPAND(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPCAST(lreg_src_c, lreg_dest, instr_mod1) TT_OP(0x90, (((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPCAST_VALID(lreg_src_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(lreg_src_c, 16) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPCAST(lreg_src_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPCAST(lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPCAST(lreg_src_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPCAST(lreg_src_c, lreg_dest, instr_mod1))

#define TT_OP_SFPCOMPC(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x8b, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPCOMPC_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPCOMPC(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPCOMPC(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPCOMPC(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPCOMPC(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPCONFIG(imm16_math, config_dest, instr_mod1) TT_OP(0x91, (((imm16_math) << 8) + ((config_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPCONFIG_VALID(imm16_math, config_dest, instr_mod1) \
    (ckernel::is_valid(imm16_math, 16) && ckernel::is_valid(config_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPCONFIG(imm16_math, config_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPCONFIG(imm16_math, config_dest, instr_mod1)
#define TTI_SFPCONFIG(imm16_math, config_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPCONFIG(imm16_math, config_dest, instr_mod1))

#define TT_OP_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x76, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPDIVP2_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPDIVP2(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPENCC(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x8a, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPENCC_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPENCC(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPENCC(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPENCC(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPENCC(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPEXEXP(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x77, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPEXEXP_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPEXEXP(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPEXEXP(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPEXEXP(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPEXEXP(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPEXMAN(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x78, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPEXMAN_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPEXMAN(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPEXMAN(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPEXMAN(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPEXMAN(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x79, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPIADD_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPIADD(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    TT_OP(0x70, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((sfpu_addr_mode) << 14) + ((dest_reg_addr) << 0)))
#define TT_SFPLOAD_VALID(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    (ckernel::is_valid(lreg_ind, 4) && ckernel::is_valid(instr_mod0, 4) && ckernel::is_valid(sfpu_addr_mode, 2) && ckernel::is_valid(dest_reg_addr, 14))
#define TT_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    ckernel::instrn_buffer[0] = TT_OP_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr)
#define TTI_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) INSTRUCTION_WORD(TT_OP_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr))

#define TT_OP_SFPLOADI(lreg_ind, instr_mod0, imm16)    TT_OP(0x71, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((imm16) << 0)))
#define TT_SFPLOADI_VALID(lreg_ind, instr_mod0, imm16) (ckernel::is_valid(lreg_ind, 4) && ckernel::is_valid(instr_mod0, 4) && ckernel::is_valid(imm16, 16))
#define TT_SFPLOADI(lreg_ind, instr_mod0, imm16)       ckernel::instrn_buffer[0] = TT_OP_SFPLOADI(lreg_ind, instr_mod0, imm16)
#define TTI_SFPLOADI(lreg_ind, instr_mod0, imm16)      INSTRUCTION_WORD(TT_OP_SFPLOADI(lreg_ind, instr_mod0, imm16))

#define TT_OP_SFPLOADMACRO(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    TT_OP(0x93, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((sfpu_addr_mode) << 14) + ((dest_reg_addr) << 0)))
#define TT_SFPLOADMACRO_VALID(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    (ckernel::is_valid(lreg_ind, 4) && ckernel::is_valid(instr_mod0, 4) && ckernel::is_valid(sfpu_addr_mode, 2) && ckernel::is_valid(dest_reg_addr, 14))
#define TT_SFPLOADMACRO(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    ckernel::instrn_buffer[0] = TT_OP_SFPLOADMACRO(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr)
#define TTI_SFPLOADMACRO(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    INSTRUCTION_WORD(TT_OP_SFPLOADMACRO(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr))

#define TT_OP_SFPLUT(lreg_ind, instr_mod0, dest_reg_addr) TT_OP(0x73, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((dest_reg_addr) << 0)))
#define TT_SFPLUT_VALID(lreg_ind, instr_mod0, dest_reg_addr) \
    (ckernel::is_valid(lreg_ind, 4) && ckernel::is_valid(instr_mod0, 4) && ckernel::is_valid(dest_reg_addr, 16))
#define TT_SFPLUT(lreg_ind, instr_mod0, dest_reg_addr)  ckernel::instrn_buffer[0] = TT_OP_SFPLUT(lreg_ind, instr_mod0, dest_reg_addr)
#define TTI_SFPLUT(lreg_ind, instr_mod0, dest_reg_addr) INSTRUCTION_WORD(TT_OP_SFPLUT(lreg_ind, instr_mod0, dest_reg_addr))

#define TT_OP_SFPLUTFP32(lreg_dest, instr_mod1)    TT_OP(0x95, (((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPLUTFP32_VALID(lreg_dest, instr_mod1) (ckernel::is_valid(lreg_dest, 20) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPLUTFP32(lreg_dest, instr_mod1)       ckernel::instrn_buffer[0] = TT_OP_SFPLUTFP32(lreg_dest, instr_mod1)
#define TTI_SFPLUTFP32(lreg_dest, instr_mod1)      INSTRUCTION_WORD(TT_OP_SFPLUTFP32(lreg_dest, instr_mod1))

#define TT_OP_SFPLZ(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x81, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPLZ_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPLZ(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPLZ(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPLZ(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPLZ(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPMAD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x84, (((lreg_src_a) << 16) + ((lreg_src_b) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPMAD_VALID(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)                                                                  \
    (ckernel::is_valid(lreg_src_a, 8) && ckernel::is_valid(lreg_src_b, 4) && ckernel::is_valid(lreg_src_c, 4) && ckernel::is_valid(lreg_dest, 4) && \
     ckernel::is_valid(instr_mod1, 4))
#define TT_SFPMAD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    ckernel::instrn_buffer[0] = TT_OP_SFPMAD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPMAD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    INSTRUCTION_WORD(TT_OP_SFPMAD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1))

#define TT_OP_SFPMOV(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x7c, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPMOV_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPMOV(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPMOV(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPMOV(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPMOV(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPMUL(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x86, (((lreg_src_a) << 16) + ((lreg_src_b) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPMUL_VALID(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)                                                                  \
    (ckernel::is_valid(lreg_src_a, 8) && ckernel::is_valid(lreg_src_b, 4) && ckernel::is_valid(lreg_src_c, 4) && ckernel::is_valid(lreg_dest, 4) && \
     ckernel::is_valid(instr_mod1, 4))
#define TT_SFPMUL(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    ckernel::instrn_buffer[0] = TT_OP_SFPMUL(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPMUL(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    INSTRUCTION_WORD(TT_OP_SFPMUL(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1))

#define TT_OP_SFPMULI(imm16_math, lreg_dest, instr_mod1) TT_OP(0x74, (((imm16_math) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPMULI_VALID(imm16_math, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm16_math, 16) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPMULI(imm16_math, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPMULI(imm16_math, lreg_dest, instr_mod1)
#define TTI_SFPMULI(imm16_math, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPMULI(imm16_math, lreg_dest, instr_mod1))

#define TT_OP_SFPNOP TT_OP(0x8f, 0)
#define TTI_SFPNOP   INSTRUCTION_WORD(TT_OP_SFPNOP)

#define TT_OP_SFPNOT(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x80, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPNOT_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPNOT(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPNOT(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPNOT(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPNOT(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPOR(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x7f, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPOR_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPOR(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPOR(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPOR(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPOR(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPPOPC(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x88, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPPOPC_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPPOPC(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPPOPC(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPPOPC(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPPOPC(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPPUSHC(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x87, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPPUSHC_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPPUSHC(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPPUSHC(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPPUSHC(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPPUSHC(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPSETCC(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x7b, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSETCC_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPSETCC(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPSETCC(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSETCC(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPSETCC(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x82, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSETEXP_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPSETEXP(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x83, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSETMAN_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPSETMAN(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x89, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSETSGN_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPSETSGN(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x7a, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSHFT_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPSHFT(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPSHFT2(imm12_math, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x94, (((imm12_math) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSHFT2_VALID(imm12_math, lreg_src_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_src_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPSHFT2(imm12_math, lreg_src_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPSHFT2(imm12_math, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPSHFT2(imm12_math, lreg_src_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPSHFT2(imm12_math, lreg_src_c, lreg_dest, instr_mod1))

#define TT_OP_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    TT_OP(0x72, (((lreg_ind) << 20) + ((instr_mod0) << 16) + ((sfpu_addr_mode) << 14) + ((dest_reg_addr) << 0)))
#define TT_SFPSTORE_VALID(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    (ckernel::is_valid(lreg_ind, 4) && ckernel::is_valid(instr_mod0, 4) && ckernel::is_valid(sfpu_addr_mode, 2) && ckernel::is_valid(dest_reg_addr, 14))
#define TT_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) \
    ckernel::instrn_buffer[0] = TT_OP_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr)
#define TTI_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr) INSTRUCTION_WORD(TT_OP_SFPSTORE(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr))

#define TT_OP_SFPSWAP(imm12_math, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x92, (((imm12_math) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPSWAP_VALID(imm12_math, lreg_src_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_src_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPSWAP(imm12_math, lreg_src_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPSWAP(imm12_math, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFPSWAP(imm12_math, lreg_src_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPSWAP(imm12_math, lreg_src_c, lreg_dest, instr_mod1))

#define TT_OP_SFPTRANSP(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    TT_OP(0x8c, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPTRANSP_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPTRANSP(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPTRANSP(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPTRANSP(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPTRANSP(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFPXOR(imm12_math, lreg_c, lreg_dest, instr_mod1) TT_OP(0x8d, (((imm12_math) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFPXOR_VALID(imm12_math, lreg_c, lreg_dest, instr_mod1) \
    (ckernel::is_valid(imm12_math, 12) && ckernel::is_valid(lreg_c, 4) && ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFPXOR(imm12_math, lreg_c, lreg_dest, instr_mod1)  ckernel::instrn_buffer[0] = TT_OP_SFPXOR(imm12_math, lreg_c, lreg_dest, instr_mod1)
#define TTI_SFPXOR(imm12_math, lreg_c, lreg_dest, instr_mod1) INSTRUCTION_WORD(TT_OP_SFPXOR(imm12_math, lreg_c, lreg_dest, instr_mod1))

#define TT_OP_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_OP(0x8e, (((rnd_mode) << 21) + ((imm8_math) << 16) + ((lreg_src_b) << 12) + ((lreg_src_c) << 8) + ((lreg_dest) << 4) + ((instr_mod1) << 0)))
#define TT_SFP_STOCH_RND_VALID(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)                                                \
    (ckernel::is_valid(rnd_mode, 3) && ckernel::is_valid(imm8_math, 5) && ckernel::is_valid(lreg_src_b, 4) && ckernel::is_valid(lreg_src_c, 4) && \
     ckernel::is_valid(lreg_dest, 4) && ckernel::is_valid(instr_mod1, 4))
#define TT_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    ckernel::instrn_buffer[0] = TT_OP_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)
#define TTI_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    INSTRUCTION_WORD(TT_OP_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1))

#define TT_OP_SHIFTDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_OP(0x5c, (((OpBisConst) << 23) + ((OpSel) << 18) + ((ResultRegIndex) << 12) + ((OpBRegIndex) << 6) + ((OpARegIndex) << 0)))
#define TT_SHIFTDMAREG_VALID(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)                                                            \
    (ckernel::is_valid(OpBisConst, 1) && ckernel::is_valid(OpSel, 5) && ckernel::is_valid(ResultRegIndex, 6) && ckernel::is_valid(OpBRegIndex, 6) && \
     ckernel::is_valid(OpARegIndex, 6))
#define TT_SHIFTDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    ckernel::instrn_buffer[0] = TT_OP_SHIFTDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex)
#define TTI_SHIFTDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    INSTRUCTION_WORD(TT_OP_SHIFTDMAREG(OpBisConst, OpSel, ResultRegIndex, OpBRegIndex, OpARegIndex))

#define TT_OP_SHIFTXA(log2_amount2, shift_mode)    TT_OP(0x17, (((log2_amount2) << 2) + ((shift_mode) << 0)))
#define TT_SHIFTXA_VALID(log2_amount2, shift_mode) (ckernel::is_valid(log2_amount2, 18) && ckernel::is_valid(shift_mode, 2))
#define TT_SHIFTXA(log2_amount2, shift_mode)       ckernel::instrn_buffer[0] = TT_OP_SHIFTXA(log2_amount2, shift_mode)
#define TTI_SHIFTXA(log2_amount2, shift_mode)      INSTRUCTION_WORD(TT_OP_SHIFTXA(log2_amount2, shift_mode))

#define TT_OP_SHIFTXB(addr_mode, rot_shift, shift_row) TT_OP(0x18, (((addr_mode) << 15) + ((rot_shift) << 10) + ((shift_row) << 0)))
#define TT_SHIFTXB_VALID(addr_mode, rot_shift, shift_row) \
    (ckernel::is_valid(addr_mode, 9) && ckernel::is_valid(rot_shift, 5) && ckernel::is_valid(shift_row, 10))
#define TT_SHIFTXB(addr_mode, rot_shift, shift_row)  ckernel::instrn_buffer[0] = TT_OP_SHIFTXB(addr_mode, rot_shift, shift_row)
#define TTI_SHIFTXB(addr_mode, rot_shift, shift_row) INSTRUCTION_WORD(TT_OP_SHIFTXB(addr_mode, rot_shift, shift_row))

#define TT_OP_STALLWAIT(stall_res, wait_res)    TT_OP(0xa2, (((stall_res) << 15) + ((wait_res) << 0)))
#define TT_STALLWAIT_VALID(stall_res, wait_res) (ckernel::is_valid(stall_res, 9) && ckernel::is_valid(wait_res, 15))
#define TT_STALLWAIT(stall_res, wait_res)       ckernel::instrn_buffer[0] = TT_OP_STALLWAIT(stall_res, wait_res)
#define TTI_STALLWAIT(stall_res, wait_res)      INSTRUCTION_WORD(TT_OP_STALLWAIT(stall_res, wait_res))

#define TT_OP_STOREIND(MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex)                                      \
    TT_OP(                                                                                                                                         \
        0x66,                                                                                                                                      \
        (((MemHierSel) << 23) + ((SizeSel) << 22) + ((RegSizeSel) << 21) + ((OffsetIndex) << 14) + ((AutoIncSpec) << 12) + ((DataRegIndex) << 6) + \
         ((AddrRegIndex) << 0)))
#define TT_STOREIND_VALID(MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex)                                   \
    (ckernel::is_valid(MemHierSel, 1) && ckernel::is_valid(SizeSel, 1) && ckernel::is_valid(RegSizeSel, 1) && ckernel::is_valid(OffsetIndex, 7) && \
     ckernel::is_valid(AutoIncSpec, 2) && ckernel::is_valid(DataRegIndex, 6) && ckernel::is_valid(AddrRegIndex, 6))
#define TT_STOREIND(MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex) \
    ckernel::instrn_buffer[0] = TT_OP_STOREIND(MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex)
#define TTI_STOREIND(MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex) \
    INSTRUCTION_WORD(TT_OP_STOREIND(MemHierSel, SizeSel, RegSizeSel, OffsetIndex, AutoIncSpec, DataRegIndex, AddrRegIndex))

#define TT_OP_STOREREG(TdmaDataRegIndex, RegAddr)    TT_OP(0x67, (((TdmaDataRegIndex) << 18) + ((RegAddr) << 0)))
#define TT_STOREREG_VALID(TdmaDataRegIndex, RegAddr) (ckernel::is_valid(TdmaDataRegIndex, 6) && ckernel::is_valid(RegAddr, 18))
#define TT_STOREREG(TdmaDataRegIndex, RegAddr)       ckernel::instrn_buffer[0] = TT_OP_STOREREG(TdmaDataRegIndex, RegAddr)
#define TTI_STOREREG(TdmaDataRegIndex, RegAddr)      INSTRUCTION_WORD(TT_OP_STOREREG(TdmaDataRegIndex, RegAddr))

#define TT_OP_SUBDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    TT_OP(0x59, (((OpBisConst) << 23) + ((ResultRegIndex) << 12) + ((OpBRegIndex) << 6) + ((OpARegIndex) << 0)))
#define TT_SUBDMAREG_VALID(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    (ckernel::is_valid(OpBisConst, 1) && ckernel::is_valid(ResultRegIndex, 6) && ckernel::is_valid(OpBRegIndex, 6) && ckernel::is_valid(OpARegIndex, 6))
#define TT_SUBDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    ckernel::instrn_buffer[0] = TT_OP_SUBDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex)
#define TTI_SUBDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex) \
    INSTRUCTION_WORD(TT_OP_SUBDMAREG(OpBisConst, ResultRegIndex, OpBRegIndex, OpARegIndex))

#define TT_OP_TBUFCMD TT_OP(0x4b, 0)
#define TTI_TBUFCMD   INSTRUCTION_WORD(TT_OP_TBUFCMD)

#define TT_OP_TRNSPSRCA TT_OP(0x14, 0)
#define TTI_TRNSPSRCA   INSTRUCTION_WORD(TT_OP_TRNSPSRCA)

#define TT_OP_TRNSPSRCB TT_OP(0x16, 0)
#define TTI_TRNSPSRCB   INSTRUCTION_WORD(TT_OP_TRNSPSRCB)

#define TT_OP_UNPACR(                                                                                                                              \
    Unpack_block_selection,                                                                                                                        \
    AddrMode,                                                                                                                                      \
    CfgContextCntInc,                                                                                                                              \
    CfgContextId,                                                                                                                                  \
    AddrCntContextId,                                                                                                                              \
    OvrdThreadId,                                                                                                                                  \
    SetDatValid,                                                                                                                                   \
    rareb_en,                                                                                                                                      \
    ZeroWrite2,                                                                                                                                    \
    AutoIncContextID,                                                                                                                              \
    RowSearch,                                                                                                                                     \
    SearchCacheFlush,                                                                                                                              \
    Last)                                                                                                                                          \
    TT_OP(                                                                                                                                         \
        0x42,                                                                                                                                      \
        (((Unpack_block_selection) << 23) + ((AddrMode) << 15) + ((CfgContextCntInc) << 13) + ((CfgContextId) << 10) + ((AddrCntContextId) << 8) + \
         ((OvrdThreadId) << 7) + ((SetDatValid) << 6) + ((rareb_en) << 5) + ((ZeroWrite2) << 4) + ((AutoIncContextID) << 3) + ((RowSearch) << 2) + \
         ((SearchCacheFlush) << 1) + ((Last) << 0)))
#define TT_UNPACR_VALID(                                                                                                                                  \
    Unpack_block_selection,                                                                                                                               \
    AddrMode,                                                                                                                                             \
    CfgContextCntInc,                                                                                                                                     \
    CfgContextId,                                                                                                                                         \
    AddrCntContextId,                                                                                                                                     \
    OvrdThreadId,                                                                                                                                         \
    SetDatValid,                                                                                                                                          \
    rareb_en,                                                                                                                                             \
    ZeroWrite2,                                                                                                                                           \
    AutoIncContextID,                                                                                                                                     \
    RowSearch,                                                                                                                                            \
    SearchCacheFlush,                                                                                                                                     \
    Last)                                                                                                                                                 \
    (ckernel::is_valid(Unpack_block_selection, 1) && ckernel::is_valid(AddrMode, 8) && ckernel::is_valid(CfgContextCntInc, 1) &&                          \
     ckernel::is_valid(CfgContextId, 3) && ckernel::is_valid(AddrCntContextId, 2) && ckernel::is_valid(OvrdThreadId, 1) &&                                \
     ckernel::is_valid(SetDatValid, 1) && ckernel::is_valid(rareb_en, 1) && ckernel::is_valid(ZeroWrite2, 1) && ckernel::is_valid(AutoIncContextID, 1) && \
     ckernel::is_valid(RowSearch, 1) && ckernel::is_valid(SearchCacheFlush, 1) && ckernel::is_valid(Last, 1))
#define TT_UNPACR(                            \
    Unpack_block_selection,                   \
    AddrMode,                                 \
    CfgContextCntInc,                         \
    CfgContextId,                             \
    AddrCntContextId,                         \
    OvrdThreadId,                             \
    SetDatValid,                              \
    rareb_en,                                 \
    ZeroWrite2,                               \
    AutoIncContextID,                         \
    RowSearch,                                \
    SearchCacheFlush,                         \
    Last)                                     \
    ckernel::instrn_buffer[0] = TT_OP_UNPACR( \
        Unpack_block_selection,               \
        AddrMode,                             \
        CfgContextCntInc,                     \
        CfgContextId,                         \
        AddrCntContextId,                     \
        OvrdThreadId,                         \
        SetDatValid,                          \
        rareb_en,                             \
        ZeroWrite2,                           \
        AutoIncContextID,                     \
        RowSearch,                            \
        SearchCacheFlush,                     \
        Last)
#define TTI_UNPACR(                \
    Unpack_block_selection,        \
    AddrMode,                      \
    CfgContextCntInc,              \
    CfgContextId,                  \
    AddrCntContextId,              \
    OvrdThreadId,                  \
    SetDatValid,                   \
    rareb_en,                      \
    ZeroWrite2,                    \
    AutoIncContextID,              \
    RowSearch,                     \
    SearchCacheFlush,              \
    Last)                          \
    INSTRUCTION_WORD(TT_OP_UNPACR( \
        Unpack_block_selection,    \
        AddrMode,                  \
        CfgContextCntInc,          \
        CfgContextId,              \
        AddrCntContextId,          \
        OvrdThreadId,              \
        SetDatValid,               \
        rareb_en,                  \
        ZeroWrite2,                \
        AutoIncContextID,          \
        RowSearch,                 \
        SearchCacheFlush,          \
        Last))

#define TT_OP_UNPACR_NOP(Unpack_block_selection, NoOp)    TT_OP(0x43, (((Unpack_block_selection) << 23) + ((NoOp) << 0)))
#define TT_UNPACR_NOP_VALID(Unpack_block_selection, NoOp) (ckernel::is_valid(Unpack_block_selection, 1) && ckernel::is_valid(NoOp, 23))
#define TT_UNPACR_NOP(Unpack_block_selection, NoOp)       ckernel::instrn_buffer[0] = TT_OP_UNPACR_NOP(Unpack_block_selection, NoOp)
#define TTI_UNPACR_NOP(Unpack_block_selection, NoOp)      INSTRUCTION_WORD(TT_OP_UNPACR_NOP(Unpack_block_selection, NoOp))

#define TT_OP_WRCFG(GprAddress, wr128b, CfgReg)    TT_OP(0xb0, (((GprAddress) << 16) + ((wr128b) << 15) + ((CfgReg) << 0)))
#define TT_WRCFG_VALID(GprAddress, wr128b, CfgReg) (ckernel::is_valid(GprAddress, 8) && ckernel::is_valid(wr128b, 1) && ckernel::is_valid(CfgReg, 15))
#define TT_WRCFG(GprAddress, wr128b, CfgReg)       ckernel::instrn_buffer[0] = TT_OP_WRCFG(GprAddress, wr128b, CfgReg)
#define TTI_WRCFG(GprAddress, wr128b, CfgReg)      INSTRUCTION_WORD(TT_OP_WRCFG(GprAddress, wr128b, CfgReg))

#define TT_OP_XMOV(Mov_block_selection, Last)    TT_OP(0x40, (((Mov_block_selection) << 23) + ((Last) << 0)))
#define TT_XMOV_VALID(Mov_block_selection, Last) (ckernel::is_valid(Mov_block_selection, 1) && ckernel::is_valid(Last, 23))
#define TT_XMOV(Mov_block_selection, Last)       ckernel::instrn_buffer[0] = TT_OP_XMOV(Mov_block_selection, Last)
#define TTI_XMOV(Mov_block_selection, Last)      INSTRUCTION_WORD(TT_OP_XMOV(Mov_block_selection, Last))

#define TT_OP_ZEROACC(clear_mode, AddrMode, dst)    TT_OP(0x10, (((clear_mode) << 19) + ((AddrMode) << 15) + ((dst) << 0)))
#define TT_ZEROACC_VALID(clear_mode, AddrMode, dst) (ckernel::is_valid(clear_mode, 5) && ckernel::is_valid(AddrMode, 4) && ckernel::is_valid(dst, 15))
#define TT_ZEROACC(clear_mode, AddrMode, dst)       ckernel::instrn_buffer[0] = TT_OP_ZEROACC(clear_mode, AddrMode, dst)
#define TTI_ZEROACC(clear_mode, AddrMode, dst)      INSTRUCTION_WORD(TT_OP_ZEROACC(clear_mode, AddrMode, dst))

#define TT_OP_ZEROSRC(zero_val, write_mode, bank_mask, src_mask) TT_OP(0x11, (((zero_val) << 4) + ((write_mode) << 3) + ((bank_mask) << 2) + ((src_mask) << 0)))
#define TT_ZEROSRC_VALID(zero_val, write_mode, bank_mask, src_mask) \
    (ckernel::is_valid(zero_val, 20) && ckernel::is_valid(write_mode, 1) && ckernel::is_valid(bank_mask, 1) && ckernel::is_valid(src_mask, 2))
#define TT_ZEROSRC(zero_val, write_mode, bank_mask, src_mask)  ckernel::instrn_buffer[0] = TT_OP_ZEROSRC(zero_val, write_mode, bank_mask, src_mask)
#define TTI_ZEROSRC(zero_val, write_mode, bank_mask, src_mask) INSTRUCTION_WORD(TT_OP_ZEROSRC(zero_val, write_mode, bank_mask, src_mask))

// Only the three TRISCs are built with the Tensix extension
// (-mcpu=tt-wh-tensix); BRISC/NCRISC use plain -mcpu=tt-wh, where these
// builtins are not registered at all.  Those cores still issue the odd Tensix
// instruction (boot.h's TTI_ZEROACC/TTI_SFPCONFIG) as a raw .ttinsn word,
// which needs no codegen support, so leave the legacy macros in place there.
// Without this guard the declarations below turn every call into an ordinary
// extern function and the failure surfaces as an undefined reference at link.
#if defined(__riscv_xtttensixwh)

// The instruction-issue macros below expand to compiler builtins.  Many of
// those expansions land inside templates, where a non-dependent name must be
// declared at the point of definition, so pull in sfpi's declarations rather
// than relying on the builtin being known implicitly.  Declaring a __builtin_
// name is a redundant redeclaration of what the compiler already knows, hence
// the suppression.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include "tensix_builtins.h"
#pragma GCC diagnostic pop

// A Tensix instruction is issued by storing its word to the instruction buffer,
// which the macros below do directly.  ckernel.h declares the buffer too, but
// only after including headers that already issue instructions, so declare it
// here.  std::uint32_t, not unsigned: uint32_t is long unsigned int on this
// target and a mismatched declaration is an error.
#include <cstdint>
extern volatile std::uint32_t __instrn_buffer[];

// --------------------------------------------------------------------------
// Route instruction issue through the compiler's intrinsics.
//
// TT_<OP> (runtime issue, formerly a store to instrn_buffer) and TTI_<OP>
// (compile-time issue, formerly .ttinsn) both become the same intrinsic call.
// The compiler emits the immediate instruction when every operand is a
// constant and materialises the word otherwise, so the choice those two
// spellings used to make by hand is now the compiler's to make.  Kernels are
// unchanged: they keep writing TTI_FOO(...) / TT_FOO(...).
//
// This is what lets pass_rvtt_config see the instruction stream at all: an
// asm-issued instruction is opaque to it, so every .ttinsn was a barrier that
// stopped config folding dead.
//
// TT_OP_<OP> is deliberately NOT redirected: it yields the instruction
// *word* as a value, which MOP templates and the replay buffer store rather
// than issue.  The operands are cast because the builtins are typed and the
// callers pass scoped enums the old arithmetic macros accepted implicitly.
//
// Redirected: 124 of 128.  These have no matching intrinsic or a
// different operand shape, and keep their original definitions:
//   RAREB, SFPLUTFP32, SFPSETMAN, TRNSPSRCA
//
// RAREB and TRNSPSRCA have intrinsics but no rvtt-cfg-reads.def entry, so
// routing them through one would trade an asm barrier for a "not in read-set
// table" barrier and fold nothing.  RAREB is named for the unpacker's rarefy-B
// feature, which is config-driven, so an empty read set is a guess this file
// is not the place to make.
// --------------------------------------------------------------------------

#undef TT_ADDDMAREG
#define TT_ADDDMAREG(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_adddmareg((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_ADDDMAREG
#define TTI_ADDDMAREG(a0, a1, a2, a3) TT_ADDDMAREG(a0, a1, a2, a3)
#undef TT_ADDRCRXY
#define TT_ADDRCRXY(a0, a1, a2, a3, a4, a5) \
    __instrn_buffer[0] = __builtin_rvtt_addrcrxy((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4), (unsigned)(a5))
#undef TTI_ADDRCRXY
#define TTI_ADDRCRXY(a0, a1, a2, a3, a4, a5) TT_ADDRCRXY(a0, a1, a2, a3, a4, a5)
#undef TT_ADDRCRZW
#define TT_ADDRCRZW(a0, a1, a2, a3, a4, a5) \
    __instrn_buffer[0] = __builtin_rvtt_addrcrzw((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4), (unsigned)(a5))
#undef TTI_ADDRCRZW
#define TTI_ADDRCRZW(a0, a1, a2, a3, a4, a5) TT_ADDRCRZW(a0, a1, a2, a3, a4, a5)
#undef TT_APOOL3S1
#define TT_APOOL3S1(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_apool3s1((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_APOOL3S1
#define TTI_APOOL3S1(a0, a1, a2, a3) TT_APOOL3S1(a0, a1, a2, a3)
#undef TT_APOOL3S2
#define TT_APOOL3S2(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_apool3s2((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_APOOL3S2
#define TTI_APOOL3S2(a0, a1, a2, a3) TT_APOOL3S2(a0, a1, a2, a3)
#undef TT_ATCAS
#define TT_ATCAS(a0, a1, a2, a3, a4, a5) \
    __instrn_buffer[0] = __builtin_rvtt_atcas((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4), (unsigned)(a5))
#undef TTI_ATCAS
#define TTI_ATCAS(a0, a1, a2, a3, a4, a5) TT_ATCAS(a0, a1, a2, a3, a4, a5)
#undef TT_ATGETM
#define TT_ATGETM(a0) __instrn_buffer[0] = __builtin_rvtt_atgetm((unsigned)(a0))
#undef TTI_ATGETM
#define TTI_ATGETM(a0) TT_ATGETM(a0)
#undef TT_ATINCGET
#define TT_ATINCGET(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_atincget((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_ATINCGET
#define TTI_ATINCGET(a0, a1, a2, a3, a4) TT_ATINCGET(a0, a1, a2, a3, a4)
#undef TT_ATINCGETPTR
#define TT_ATINCGETPTR(a0, a1, a2, a3, a4, a5, a6) \
    __instrn_buffer[0] =                           \
        __builtin_rvtt_atincgetptr((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4), (unsigned)(a5), (unsigned)(a6))
#undef TTI_ATINCGETPTR
#define TTI_ATINCGETPTR(a0, a1, a2, a3, a4, a5, a6) TT_ATINCGETPTR(a0, a1, a2, a3, a4, a5, a6)
#undef TT_ATRELM
#define TT_ATRELM(a0) __instrn_buffer[0] = __builtin_rvtt_atrelm((unsigned)(a0))
#undef TTI_ATRELM
#define TTI_ATRELM(a0) TT_ATRELM(a0)
#undef TT_ATSWAP
#define TT_ATSWAP(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_atswap((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_ATSWAP
#define TTI_ATSWAP(a0, a1, a2, a3) TT_ATSWAP(a0, a1, a2, a3)
#undef TT_BITWOPDMAREG
#define TT_BITWOPDMAREG(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_bitwopdmareg((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_BITWOPDMAREG
#define TTI_BITWOPDMAREG(a0, a1, a2, a3, a4) TT_BITWOPDMAREG(a0, a1, a2, a3, a4)
#undef TT_CLEARDVALID
#define TT_CLEARDVALID(a0, a1) __instrn_buffer[0] = __builtin_rvtt_cleardvalid((unsigned)(a0), (unsigned)(a1))
#undef TTI_CLEARDVALID
#define TTI_CLEARDVALID(a0, a1) TT_CLEARDVALID(a0, a1)
#undef TT_CMPDMAREG
#define TT_CMPDMAREG(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_cmpdmareg((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_CMPDMAREG
#define TTI_CMPDMAREG(a0, a1, a2, a3, a4) TT_CMPDMAREG(a0, a1, a2, a3, a4)
#undef TT_CONV3S1
#define TT_CONV3S1(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_conv3s1((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_CONV3S1
#define TTI_CONV3S1(a0, a1, a2, a3) TT_CONV3S1(a0, a1, a2, a3)
#undef TT_CONV3S2
#define TT_CONV3S2(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_conv3s2((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_CONV3S2
#define TTI_CONV3S2(a0, a1, a2, a3) TT_CONV3S2(a0, a1, a2, a3)
#undef TT_DOTPV
#define TT_DOTPV(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_wh_dotpv((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_DOTPV
#define TTI_DOTPV(a0, a1, a2, a3, a4) TT_DOTPV(a0, a1, a2, a3, a4)
#undef TT_ELWADD
#define TT_ELWADD(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_wh_elwadd((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_ELWADD
#define TTI_ELWADD(a0, a1, a2, a3, a4) TT_ELWADD(a0, a1, a2, a3, a4)
#undef TT_ELWMUL
#define TT_ELWMUL(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_wh_elwmul((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_ELWMUL
#define TTI_ELWMUL(a0, a1, a2, a3, a4) TT_ELWMUL(a0, a1, a2, a3, a4)
#undef TT_ELWSUB
#define TT_ELWSUB(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_wh_elwsub((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_ELWSUB
#define TTI_ELWSUB(a0, a1, a2, a3, a4) TT_ELWSUB(a0, a1, a2, a3, a4)
#undef TT_FLUSHDMA
#define TT_FLUSHDMA(a0) __instrn_buffer[0] = __builtin_rvtt_flushdma((unsigned)(a0))
#undef TTI_FLUSHDMA
#define TTI_FLUSHDMA(a0) TT_FLUSHDMA(a0)
#undef TT_GAPOOL
#define TT_GAPOOL(a0, a1, a2, a3, a4) __instrn_buffer[0] = __builtin_rvtt_gapool((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_GAPOOL
#define TTI_GAPOOL(a0, a1, a2, a3, a4) TT_GAPOOL(a0, a1, a2, a3, a4)
#undef TT_GATESRCRST
#define TT_GATESRCRST(a0, a1) __instrn_buffer[0] = __builtin_rvtt_gatesrcrst((unsigned)(a0), (unsigned)(a1))
#undef TTI_GATESRCRST
#define TTI_GATESRCRST(a0, a1) TT_GATESRCRST(a0, a1)
#undef TT_GMPOOL
#define TT_GMPOOL(a0, a1, a2, a3, a4) __instrn_buffer[0] = __builtin_rvtt_gmpool((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_GMPOOL
#define TTI_GMPOOL(a0, a1, a2, a3, a4) TT_GMPOOL(a0, a1, a2, a3, a4)
#undef TT_INCADCXY
#define TT_INCADCXY(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_incadcxy((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_INCADCXY
#define TTI_INCADCXY(a0, a1, a2, a3, a4) TT_INCADCXY(a0, a1, a2, a3, a4)
#undef TT_INCADCZW
#define TT_INCADCZW(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_incadczw((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_INCADCZW
#define TTI_INCADCZW(a0, a1, a2, a3, a4) TT_INCADCZW(a0, a1, a2, a3, a4)
// The intrinsic is spelled ttincrwc, not incrwc, which is why the sweep that
// produced this list did not pair them up.
#undef TT_INCRWC
#define TT_INCRWC(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_ttincrwc((int)(a0), (int)(a1), (int)(a2), (int)(a3))
#undef TTI_INCRWC
#define TTI_INCRWC(a0, a1, a2, a3) TT_INCRWC(a0, a1, a2, a3)
#undef TT_LOADIND
#define TT_LOADIND(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_loadind((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_LOADIND
#define TTI_LOADIND(a0, a1, a2, a3, a4) TT_LOADIND(a0, a1, a2, a3, a4)
#undef TT_LOADREG
#define TT_LOADREG(a0, a1) __instrn_buffer[0] = __builtin_rvtt_loadreg((unsigned)(a0), (unsigned)(a1))
#undef TTI_LOADREG
#define TTI_LOADREG(a0, a1) TT_LOADREG(a0, a1)
#undef TT_MFCONV3S1
#define TT_MFCONV3S1(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_mfconv3s1((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_MFCONV3S1
#define TTI_MFCONV3S1(a0, a1, a2, a3) TT_MFCONV3S1(a0, a1, a2, a3)
#undef TT_MOP
#define TT_MOP(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_wh_mop((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_MOP
#define TTI_MOP(a0, a1, a2) TT_MOP(a0, a1, a2)
#undef TT_MOVA2D
#define TT_MOVA2D(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_wh_mova2d((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_MOVA2D
#define TTI_MOVA2D(a0, a1, a2, a3, a4) TT_MOVA2D(a0, a1, a2, a3, a4)
#undef TT_MOVB2A
#define TT_MOVB2A(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_movb2a((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_MOVB2A
#define TTI_MOVB2A(a0, a1, a2, a3) TT_MOVB2A(a0, a1, a2, a3)
#undef TT_MOVB2D
#define TT_MOVB2D(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_wh_movb2d((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_MOVB2D
#define TTI_MOVB2D(a0, a1, a2, a3, a4) TT_MOVB2D(a0, a1, a2, a3, a4)
#undef TT_MOVD2A
#define TT_MOVD2A(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_wh_movd2a((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_MOVD2A
#define TTI_MOVD2A(a0, a1, a2, a3, a4) TT_MOVD2A(a0, a1, a2, a3, a4)
#undef TT_MOVD2B
#define TT_MOVD2B(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_wh_movd2b((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_MOVD2B
#define TTI_MOVD2B(a0, a1, a2, a3, a4) TT_MOVD2B(a0, a1, a2, a3, a4)
#undef TT_MOVDBGA2D
#define TT_MOVDBGA2D(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_wh_movdbga2d((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_MOVDBGA2D
#define TTI_MOVDBGA2D(a0, a1, a2, a3, a4) TT_MOVDBGA2D(a0, a1, a2, a3, a4)
#undef TT_MPOOL3S1
#define TT_MPOOL3S1(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_mpool3s1((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_MPOOL3S1
#define TTI_MPOOL3S1(a0, a1, a2, a3) TT_MPOOL3S1(a0, a1, a2, a3)
#undef TT_MPOOL3S2
#define TT_MPOOL3S2(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_mpool3s2((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_MPOOL3S2
#define TTI_MPOOL3S2(a0, a1, a2, a3) TT_MPOOL3S2(a0, a1, a2, a3)
#undef TT_MULDMAREG
#define TT_MULDMAREG(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_muldmareg((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_MULDMAREG
#define TTI_MULDMAREG(a0, a1, a2, a3) TT_MULDMAREG(a0, a1, a2, a3)
// TTI_NOP and TTI_DMANOP take no operands, so they are object-like macros
// rather than function-like ones, and there is no TT_ (runtime) spelling to
// redirect: a NOP has nothing to compute at runtime.  Routing them through
// intrinsics matters out of proportion to what they do: as inline asm they
// were opaque to pass_rvtt_config, and the packer pads its config sequences
// with them, so each one discarded the tracked config state mid-sequence.
#undef TTI_DMANOP
#define TTI_DMANOP __instrn_buffer[0] = __builtin_rvtt_ttdmanop()
#undef TTI_NOP
#define TTI_NOP __instrn_buffer[0] = __builtin_rvtt_ttnop()
#undef TT_MVMUL
#define TT_MVMUL(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_mvmul((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_MVMUL
#define TTI_MVMUL(a0, a1, a2, a3) TT_MVMUL(a0, a1, a2, a3)
#undef TT_PACR
#define TT_PACR(a0, a1, a2, a3, a4, a5, a6) \
    __instrn_buffer[0] = __builtin_rvtt_wh_pacr((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4), (unsigned)(a5), (unsigned)(a6))
#undef TTI_PACR
#define TTI_PACR(a0, a1, a2, a3, a4, a5, a6) TT_PACR(a0, a1, a2, a3, a4, a5, a6)
#undef TT_RDCFG
#define TT_RDCFG(a0, a1) __instrn_buffer[0] = __builtin_rvtt_rdcfg((unsigned)(a0), (unsigned)(a1))
#undef TTI_RDCFG
#define TTI_RDCFG(a0, a1) TT_RDCFG(a0, a1)
#undef TT_REG2FLOP
#define TT_REG2FLOP(a0, a1, a2, a3, a4, a5) \
    __instrn_buffer[0] = __builtin_rvtt_reg2flop((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4), (unsigned)(a5))
#undef TTI_REG2FLOP
#define TTI_REG2FLOP(a0, a1, a2, a3, a4, a5) TT_REG2FLOP(a0, a1, a2, a3, a4, a5)
#undef TT_RMWCIB0
#define TT_RMWCIB0(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_rmwciB0((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_RMWCIB0
#define TTI_RMWCIB0(a0, a1, a2) TT_RMWCIB0(a0, a1, a2)
#undef TT_RMWCIB1
#define TT_RMWCIB1(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_rmwciB1((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_RMWCIB1
#define TTI_RMWCIB1(a0, a1, a2) TT_RMWCIB1(a0, a1, a2)
#undef TT_RMWCIB2
#define TT_RMWCIB2(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_rmwciB2((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_RMWCIB2
#define TTI_RMWCIB2(a0, a1, a2) TT_RMWCIB2(a0, a1, a2)
#undef TT_RMWCIB3
#define TT_RMWCIB3(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_rmwciB3((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_RMWCIB3
#define TTI_RMWCIB3(a0, a1, a2) TT_RMWCIB3(a0, a1, a2)
#undef TT_SEMGET
#define TT_SEMGET(a0) __instrn_buffer[0] = __builtin_rvtt_wh_semget((unsigned)(a0))
#undef TTI_SEMGET
#define TTI_SEMGET(a0) TT_SEMGET(a0)
#undef TT_SEMINIT
#define TT_SEMINIT(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_seminit((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_SEMINIT
#define TTI_SEMINIT(a0, a1, a2) TT_SEMINIT(a0, a1, a2)
#undef TT_SEMPOST
#define TT_SEMPOST(a0) __instrn_buffer[0] = __builtin_rvtt_sempost((unsigned)(a0))
#undef TTI_SEMPOST
#define TTI_SEMPOST(a0) TT_SEMPOST(a0)
#undef TT_SEMWAIT
#define TT_SEMWAIT(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_semwait((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_SEMWAIT
#define TTI_SEMWAIT(a0, a1, a2) TT_SEMWAIT(a0, a1, a2)
#undef TT_SETADC
#define TT_SETADC(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_setadc((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SETADC
#define TTI_SETADC(a0, a1, a2, a3) TT_SETADC(a0, a1, a2, a3)
#undef TT_SETADCXX
#define TT_SETADCXX(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_setadcxx((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_SETADCXX
#define TTI_SETADCXX(a0, a1, a2) TT_SETADCXX(a0, a1, a2)
#undef TT_SETADCXY
#define TT_SETADCXY(a0, a1, a2, a3, a4, a5) \
    __instrn_buffer[0] = __builtin_rvtt_setadcxy((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4), (unsigned)(a5))
#undef TTI_SETADCXY
#define TTI_SETADCXY(a0, a1, a2, a3, a4, a5) TT_SETADCXY(a0, a1, a2, a3, a4, a5)
#undef TT_SETADCZW
#define TT_SETADCZW(a0, a1, a2, a3, a4, a5) \
    __instrn_buffer[0] = __builtin_rvtt_setadczw((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4), (unsigned)(a5))
#undef TTI_SETADCZW
#define TTI_SETADCZW(a0, a1, a2, a3, a4, a5) TT_SETADCZW(a0, a1, a2, a3, a4, a5)
#undef TT_SETASHRMH
#define TT_SETASHRMH(a0, a1) __instrn_buffer[0] = __builtin_rvtt_setashrmh((unsigned)(a0), (unsigned)(a1))
#undef TTI_SETASHRMH
#define TTI_SETASHRMH(a0, a1) TT_SETASHRMH(a0, a1)
#undef TT_SETASHRMH0
#define TT_SETASHRMH0(a0, a1) __instrn_buffer[0] = __builtin_rvtt_setashrmh0((unsigned)(a0), (unsigned)(a1))
#undef TTI_SETASHRMH0
#define TTI_SETASHRMH0(a0, a1) TT_SETASHRMH0(a0, a1)
#undef TT_SETASHRMH1
#define TT_SETASHRMH1(a0, a1) __instrn_buffer[0] = __builtin_rvtt_setashrmh1((unsigned)(a0), (unsigned)(a1))
#undef TTI_SETASHRMH1
#define TTI_SETASHRMH1(a0, a1) TT_SETASHRMH1(a0, a1)
#undef TT_SETASHRMV
#define TT_SETASHRMV(a0) __instrn_buffer[0] = __builtin_rvtt_setashrmv((unsigned)(a0))
#undef TTI_SETASHRMV
#define TTI_SETASHRMV(a0) TT_SETASHRMV(a0)
#undef TT_SETC16
#define TT_SETC16(a0, a1) __instrn_buffer[0] = __builtin_rvtt_wh_setc16((unsigned)(a0), (unsigned)(a1))
#undef TTI_SETC16
#define TTI_SETC16(a0, a1) TT_SETC16(a0, a1)
#undef TT_SETDMAREG
#define TT_SETDMAREG(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_setdmareg((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SETDMAREG
#define TTI_SETDMAREG(a0, a1, a2, a3) TT_SETDMAREG(a0, a1, a2, a3)
#undef TT_SETDVALID
#define TT_SETDVALID(a0) __instrn_buffer[0] = __builtin_rvtt_setdvalid((unsigned)(a0))
#undef TTI_SETDVALID
#define TTI_SETDVALID(a0) TT_SETDVALID(a0)
#undef TT_SETIBRWC
#define TT_SETIBRWC(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_setibrwc((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_SETIBRWC
#define TTI_SETIBRWC(a0, a1, a2) TT_SETIBRWC(a0, a1, a2)
#undef TT_SETPKEDGOF
#define TT_SETPKEDGOF(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_setpkedgof((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SETPKEDGOF
#define TTI_SETPKEDGOF(a0, a1, a2, a3) TT_SETPKEDGOF(a0, a1, a2, a3)
#undef TT_SETRWC
#define TT_SETRWC(a0, a1, a2, a3, a4, a5) \
    __instrn_buffer[0] = __builtin_rvtt_setrwc((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4), (unsigned)(a5))
#undef TTI_SETRWC
#define TTI_SETRWC(a0, a1, a2, a3, a4, a5) TT_SETRWC(a0, a1, a2, a3, a4, a5)
#undef TT_SFPABS
#define TT_SFPABS(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpabs((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPABS
#define TTI_SFPABS(a0, a1, a2, a3) TT_SFPABS(a0, a1, a2, a3)
#undef TT_SFPADD
#define TT_SFPADD(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_wh_sfpadd((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_SFPADD
#define TTI_SFPADD(a0, a1, a2, a3, a4) TT_SFPADD(a0, a1, a2, a3, a4)
#undef TT_SFPADDI
#define TT_SFPADDI(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_wh_sfpaddi((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_SFPADDI
#define TTI_SFPADDI(a0, a1, a2) TT_SFPADDI(a0, a1, a2)
#undef TT_SFPAND
#define TT_SFPAND(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpand((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPAND
#define TTI_SFPAND(a0, a1, a2, a3) TT_SFPAND(a0, a1, a2, a3)
#undef TT_SFPCAST
#define TT_SFPCAST(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_wh_sfpcast((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_SFPCAST
#define TTI_SFPCAST(a0, a1, a2) TT_SFPCAST(a0, a1, a2)
#undef TT_SFPCOMPC
#define TT_SFPCOMPC(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpcompc((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPCOMPC
#define TTI_SFPCOMPC(a0, a1, a2, a3) TT_SFPCOMPC(a0, a1, a2, a3)
#undef TT_SFPCONFIG
#define TT_SFPCONFIG(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_wh_sfpconfig((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_SFPCONFIG
#define TTI_SFPCONFIG(a0, a1, a2) TT_SFPCONFIG(a0, a1, a2)
#undef TT_SFPDIVP2
#define TT_SFPDIVP2(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpdivp2((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPDIVP2
#define TTI_SFPDIVP2(a0, a1, a2, a3) TT_SFPDIVP2(a0, a1, a2, a3)
#undef TT_SFPENCC
#define TT_SFPENCC(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpencc((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPENCC
#define TTI_SFPENCC(a0, a1, a2, a3) TT_SFPENCC(a0, a1, a2, a3)
#undef TT_SFPEXEXP
#define TT_SFPEXEXP(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpexexp((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPEXEXP
#define TTI_SFPEXEXP(a0, a1, a2, a3) TT_SFPEXEXP(a0, a1, a2, a3)
#undef TT_SFPEXMAN
#define TT_SFPEXMAN(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpexman((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPEXMAN
#define TTI_SFPEXMAN(a0, a1, a2, a3) TT_SFPEXMAN(a0, a1, a2, a3)
#undef TT_SFPIADD
#define TT_SFPIADD(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpiadd((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPIADD
#define TTI_SFPIADD(a0, a1, a2, a3) TT_SFPIADD(a0, a1, a2, a3)
#undef TT_SFPLOAD
#define TT_SFPLOAD(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpload((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPLOAD
#define TTI_SFPLOAD(a0, a1, a2, a3) TT_SFPLOAD(a0, a1, a2, a3)
#undef TT_SFPLOADI
#define TT_SFPLOADI(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_wh_sfploadi((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_SFPLOADI
#define TTI_SFPLOADI(a0, a1, a2) TT_SFPLOADI(a0, a1, a2)
#undef TT_SFPLOADMACRO
#define TT_SFPLOADMACRO(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfploadmacro((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPLOADMACRO
#define TTI_SFPLOADMACRO(a0, a1, a2, a3) TT_SFPLOADMACRO(a0, a1, a2, a3)
#undef TT_SFPLUT
#define TT_SFPLUT(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_wh_sfplut((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_SFPLUT
#define TTI_SFPLUT(a0, a1, a2) TT_SFPLUT(a0, a1, a2)
#undef TT_SFPLZ
#define TT_SFPLZ(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfplz((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPLZ
#define TTI_SFPLZ(a0, a1, a2, a3) TT_SFPLZ(a0, a1, a2, a3)
#undef TT_SFPMAD
#define TT_SFPMAD(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_wh_sfpmad((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_SFPMAD
#define TTI_SFPMAD(a0, a1, a2, a3, a4) TT_SFPMAD(a0, a1, a2, a3, a4)
#undef TT_SFPMOV
#define TT_SFPMOV(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpmov((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPMOV
#define TTI_SFPMOV(a0, a1, a2, a3) TT_SFPMOV(a0, a1, a2, a3)
#undef TT_SFPMUL
#define TT_SFPMUL(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_wh_sfpmul((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_SFPMUL
#define TTI_SFPMUL(a0, a1, a2, a3, a4) TT_SFPMUL(a0, a1, a2, a3, a4)
#undef TT_SFPMULI
#define TT_SFPMULI(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_wh_sfpmuli((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_SFPMULI
#define TTI_SFPMULI(a0, a1, a2) TT_SFPMULI(a0, a1, a2)
#undef TT_SFPNOT
#define TT_SFPNOT(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpnot((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPNOT
#define TTI_SFPNOT(a0, a1, a2, a3) TT_SFPNOT(a0, a1, a2, a3)
#undef TT_SFPOR
#define TT_SFPOR(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpor((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPOR
#define TTI_SFPOR(a0, a1, a2, a3) TT_SFPOR(a0, a1, a2, a3)
#undef TT_SFPPOPC
#define TT_SFPPOPC(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfppopc((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPPOPC
#define TTI_SFPPOPC(a0, a1, a2, a3) TT_SFPPOPC(a0, a1, a2, a3)
#undef TT_SFPPUSHC
#define TT_SFPPUSHC(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfppushc((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPPUSHC
#define TTI_SFPPUSHC(a0, a1, a2, a3) TT_SFPPUSHC(a0, a1, a2, a3)
#undef TT_SFPSETCC
#define TT_SFPSETCC(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpsetcc((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPSETCC
#define TTI_SFPSETCC(a0, a1, a2, a3) TT_SFPSETCC(a0, a1, a2, a3)
#undef TT_SFPSETEXP
#define TT_SFPSETEXP(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpsetexp((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPSETEXP
#define TTI_SFPSETEXP(a0, a1, a2, a3) TT_SFPSETEXP(a0, a1, a2, a3)
#undef TT_SFPSETSGN
#define TT_SFPSETSGN(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpsetsgn((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPSETSGN
#define TTI_SFPSETSGN(a0, a1, a2, a3) TT_SFPSETSGN(a0, a1, a2, a3)
#undef TT_SFPSHFT
#define TT_SFPSHFT(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpshft((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPSHFT
#define TTI_SFPSHFT(a0, a1, a2, a3) TT_SFPSHFT(a0, a1, a2, a3)
#undef TT_SFPSHFT2
#define TT_SFPSHFT2(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpshft2((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPSHFT2
#define TTI_SFPSHFT2(a0, a1, a2, a3) TT_SFPSHFT2(a0, a1, a2, a3)
#undef TT_SFPSTORE
#define TT_SFPSTORE(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpstore((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPSTORE
#define TTI_SFPSTORE(a0, a1, a2, a3) TT_SFPSTORE(a0, a1, a2, a3)
#undef TT_SFPSWAP
#define TT_SFPSWAP(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpswap((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPSWAP
#define TTI_SFPSWAP(a0, a1, a2, a3) TT_SFPSWAP(a0, a1, a2, a3)
#undef TT_SFPTRANSP
#define TT_SFPTRANSP(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfptransp((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPTRANSP
#define TTI_SFPTRANSP(a0, a1, a2, a3) TT_SFPTRANSP(a0, a1, a2, a3)
#undef TT_SFPXOR
#define TT_SFPXOR(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_sfpxor((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SFPXOR
#define TTI_SFPXOR(a0, a1, a2, a3) TT_SFPXOR(a0, a1, a2, a3)
#undef TT_SHIFTDMAREG
#define TT_SHIFTDMAREG(a0, a1, a2, a3, a4) \
    __instrn_buffer[0] = __builtin_rvtt_shiftdmareg((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4))
#undef TTI_SHIFTDMAREG
#define TTI_SHIFTDMAREG(a0, a1, a2, a3, a4) TT_SHIFTDMAREG(a0, a1, a2, a3, a4)
#undef TT_SHIFTXA
#define TT_SHIFTXA(a0, a1) __instrn_buffer[0] = __builtin_rvtt_shiftxa((unsigned)(a0), (unsigned)(a1))
#undef TTI_SHIFTXA
#define TTI_SHIFTXA(a0, a1) TT_SHIFTXA(a0, a1)
#undef TT_SHIFTXB
#define TT_SHIFTXB(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_wh_shiftxb((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_SHIFTXB
#define TTI_SHIFTXB(a0, a1, a2) TT_SHIFTXB(a0, a1, a2)
#undef TT_STALLWAIT
#define TT_STALLWAIT(a0, a1) __instrn_buffer[0] = __builtin_rvtt_stallwait((unsigned)(a0), (unsigned)(a1))
#undef TTI_STALLWAIT
#define TTI_STALLWAIT(a0, a1) TT_STALLWAIT(a0, a1)
#undef TT_STOREIND
#define TT_STOREIND(a0, a1, a2, a3, a4, a5, a6) \
    __instrn_buffer[0] = __builtin_rvtt_storeind((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3), (unsigned)(a4), (unsigned)(a5), (unsigned)(a6))
#undef TTI_STOREIND
#define TTI_STOREIND(a0, a1, a2, a3, a4, a5, a6) TT_STOREIND(a0, a1, a2, a3, a4, a5, a6)
#undef TT_STOREREG
#define TT_STOREREG(a0, a1) __instrn_buffer[0] = __builtin_rvtt_storereg((unsigned)(a0), (unsigned)(a1))
#undef TTI_STOREREG
#define TTI_STOREREG(a0, a1) TT_STOREREG(a0, a1)
#undef TT_SUBDMAREG
#define TT_SUBDMAREG(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_subdmareg((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_SUBDMAREG
#define TTI_SUBDMAREG(a0, a1, a2, a3) TT_SUBDMAREG(a0, a1, a2, a3)
#undef TT_WRCFG
#define TT_WRCFG(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_wrcfg((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_WRCFG
#define TTI_WRCFG(a0, a1, a2) TT_WRCFG(a0, a1, a2)
#undef TT_XMOV
#define TT_XMOV(a0, a1) __instrn_buffer[0] = __builtin_rvtt_xmov((unsigned)(a0), (unsigned)(a1))
#undef TTI_XMOV
#define TTI_XMOV(a0, a1) TT_XMOV(a0, a1)
#undef TT_ZEROACC
#define TT_ZEROACC(a0, a1, a2) __instrn_buffer[0] = __builtin_rvtt_wh_zeroacc((unsigned)(a0), (unsigned)(a1), (unsigned)(a2))
#undef TTI_ZEROACC
#define TTI_ZEROACC(a0, a1, a2) TT_ZEROACC(a0, a1, a2)
#undef TT_ZEROSRC
#define TT_ZEROSRC(a0, a1, a2, a3) __instrn_buffer[0] = __builtin_rvtt_wh_zerosrc((unsigned)(a0), (unsigned)(a1), (unsigned)(a2), (unsigned)(a3))
#undef TTI_ZEROSRC
#define TTI_ZEROSRC(a0, a1, a2, a3) TT_ZEROSRC(a0, a1, a2, a3)

// These have intrinsics too.  The sweep that produced the list above paired
// macros with builtins by name and operand count, which is why the operand-less
// ones and UNPACR (whose macro names its 13 fields) were left behind.
//
// It is worth the noise: left as .ttinsn each one is opaque to
// pass_rvtt_config, which then discards the whole config state at every
// occurrence.
#undef TTI_CLREXPHIST
#define TTI_CLREXPHIST __instrn_buffer[0] = __builtin_rvtt_clrexphist()
#undef TTI_RSTDMA
#define TTI_RSTDMA __instrn_buffer[0] = __builtin_rvtt_rstdma()
#undef TTI_SFPNOP
#define TTI_SFPNOP __instrn_buffer[0] = __builtin_rvtt_sfpnop()
#undef TTI_TBUFCMD
#define TTI_TBUFCMD __instrn_buffer[0] = __builtin_rvtt_tbufcmd()
#undef TTI_TRNSPSRCB
#define TTI_TRNSPSRCB __instrn_buffer[0] = __builtin_rvtt_trnspsrcb()

#undef TT_PACR_SETREG
#define TT_PACR_SETREG(Push, AddrSel, WrData, PackSel, StreamId, Flush, Last) \
    __instrn_buffer[0] = __builtin_rvtt_wh_pacrsetreg(                        \
        (unsigned)(Push), (unsigned)(AddrSel), (unsigned)(WrData), (unsigned)(PackSel), (unsigned)(StreamId), (unsigned)(Flush), (unsigned)(Last))
#undef TTI_PACR_SETREG
#define TTI_PACR_SETREG(Push, AddrSel, WrData, PackSel, StreamId, Flush, Last) TT_PACR_SETREG(Push, AddrSel, WrData, PackSel, StreamId, Flush, Last)

#undef TT_MOP_CFG
#define TT_MOP_CFG(zmask_hi16) __instrn_buffer[0] = __builtin_rvtt_mopcfg((unsigned)(zmask_hi16))
#undef TTI_MOP_CFG
#define TTI_MOP_CFG(zmask_hi16) TT_MOP_CFG(zmask_hi16)

// REPLAY must be visible to pass_rvtt_replay: issued as .ttinsn it is opaque,
// so the pass neither reserves the buffer slots this reserves nor sees that it
// is emitting inside a recording window, and it allocates on top of whatever
// the author recorded.  ckernel_sfpu_gcd.h captures 28 instructions this way.
#undef TT_REPLAY
#define TT_REPLAY(start_idx, len, execute_while_loading, load_mode) \
    __builtin_rvtt_ttreplay(ckernel::instrn_buffer, (unsigned)(len), 0, 0, (unsigned)(start_idx), (unsigned)(execute_while_loading), (unsigned)(load_mode))
#undef TTI_REPLAY
#define TTI_REPLAY(start_idx, len, execute_while_loading, load_mode) TT_REPLAY(start_idx, len, execute_while_loading, load_mode)

// Spelled wh_sfpstochrnd, not sfp_stoch_rnd, which is why the name sweep
// missed it.  The field order matches TT_OP_SFP_STOCH_RND exactly.
#undef TT_SFP_STOCH_RND
#define TT_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    __instrn_buffer[0] = __builtin_rvtt_wh_sfpstochrnd(                                      \
        (unsigned)(rnd_mode), (unsigned)(imm8_math), (unsigned)(lreg_src_b), (unsigned)(lreg_src_c), (unsigned)(lreg_dest), (unsigned)(instr_mod1))
#undef TTI_SFP_STOCH_RND
#define TTI_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1) \
    TT_SFP_STOCH_RND(rnd_mode, imm8_math, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1)

#undef TTI_UNPACR
#define TTI_UNPACR(                                \
    Unpack_block_selection,                        \
    AddrMode,                                      \
    CfgContextCntInc,                              \
    CfgContextId,                                  \
    AddrCntContextId,                              \
    OvrdThreadId,                                  \
    SetDatValid,                                   \
    rareb_en,                                      \
    ZeroWrite2,                                    \
    AutoIncContextID,                              \
    RowSearch,                                     \
    SearchCacheFlush,                              \
    Last)                                          \
    __instrn_buffer[0] = __builtin_rvtt_wh_unpacr( \
        (unsigned)(Unpack_block_selection),        \
        (unsigned)(AddrMode),                      \
        (unsigned)(CfgContextCntInc),              \
        (unsigned)(CfgContextId),                  \
        (unsigned)(AddrCntContextId),              \
        (unsigned)(OvrdThreadId),                  \
        (unsigned)(SetDatValid),                   \
        (unsigned)(rareb_en),                      \
        (unsigned)(ZeroWrite2),                    \
        (unsigned)(AutoIncContextID),              \
        (unsigned)(RowSearch),                     \
        (unsigned)(SearchCacheFlush),              \
        (unsigned)(Last))
#undef TT_UNPACR_NOP
#define TT_UNPACR_NOP(a0, a1) __instrn_buffer[0] = __builtin_rvtt_wh_unpacr_nop((unsigned)(a0), (unsigned)(a1))
#undef TTI_UNPACR_NOP
#define TTI_UNPACR_NOP(a0, a1) TT_UNPACR_NOP(a0, a1)

#endif // __riscv_xtttensixwh
