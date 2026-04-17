// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// MT: This should be dissolved and moved to the appropriate place
#include <cstdint>

#include "tensix.h"

// Hand-coded parameter encoding for various common instructions
namespace ckernel
{

struct p_setrwc
{
    constexpr static std::uint32_t CLR_A    = 0x1;
    constexpr static std::uint32_t CLR_B    = 0x2;
    constexpr static std::uint32_t CLR_AB   = 0x3;
    constexpr static std::uint32_t CLR_NONE = 0x0;

    constexpr static std::uint32_t SET_A     = 0x1;
    constexpr static std::uint32_t SET_B     = 0x2;
    constexpr static std::uint32_t SET_AB    = 0x3;
    constexpr static std::uint32_t SET_D     = 0x4;
    constexpr static std::uint32_t SET_AD    = 0x5;
    constexpr static std::uint32_t SET_BD    = 0x6;
    constexpr static std::uint32_t SET_ABD   = 0x7;
    constexpr static std::uint32_t SET_F     = 0x8;
    constexpr static std::uint32_t SET_A_F   = 0x9;
    constexpr static std::uint32_t SET_B_F   = 0xa;
    constexpr static std::uint32_t SET_AB_F  = 0xb;
    constexpr static std::uint32_t SET_D_F   = 0xc;
    constexpr static std::uint32_t SET_AD_F  = 0xd;
    constexpr static std::uint32_t SET_BD_F  = 0xe;
    constexpr static std::uint32_t SET_ABD_F = 0xf;

    constexpr static std::uint32_t CR_A         = 0x1;
    constexpr static std::uint32_t CR_B         = 0x2;
    constexpr static std::uint32_t CR_AB        = 0x3;
    constexpr static std::uint32_t CR_D         = 0x4;
    constexpr static std::uint32_t CR_AD        = 0x5;
    constexpr static std::uint32_t CR_BD        = 0x6;
    constexpr static std::uint32_t CR_ABD       = 0x7;
    constexpr static std::uint32_t C_TO_CR_MODE = 0x8;
};

struct p_setibrwc
{
    constexpr static std::uint32_t SET_BIAS = 0x0;
    constexpr static std::uint32_t INC_BIAS = 0x1;
    constexpr static std::uint32_t CR_NONE  = 0x0;
    constexpr static std::uint32_t CR_BIAS  = 0x1;
};

struct p_unpacr
{
    constexpr static std::uint32_t RAREFYB_DISABLE = 0x0;
    constexpr static std::uint32_t RAREFYB_ENABLE  = 0x1;

    constexpr static std::uint32_t TILE0_ADDRCNT_CONTEXT = (0); // Address counter context for tile 0
    constexpr static std::uint32_t TILE1_ADDRCNT_CONTEXT = (0); // Address counter context for tile 1
    constexpr static std::uint32_t TILE2_ADDRCNT_CONTEXT = (1); // Address counter context for tile 2
    constexpr static std::uint32_t TILE3_ADDRCNT_CONTEXT = (1); // Address counter context for tile 3
    constexpr static std::uint32_t TILE0_CFG_CONTEXT     = (0); // Config context for tile 0
    constexpr static std::uint32_t TILE1_CFG_CONTEXT     = (0); // Config context for tile 1
    constexpr static std::uint32_t TILE2_CFG_CONTEXT     = (0); // Config context for tile 2
    constexpr static std::uint32_t TILE3_CFG_CONTEXT     = (0); // Config context for tile 3
    constexpr static std::uint32_t AUTO_INC_CONTEXT      = (1); // Auto increment config context (max value set through unpacker config command)
};

struct p_unpacr_nop
{
    constexpr static std::uint32_t UNP_POP = 0b000;
    constexpr static std::uint32_t UNP_NOP = 0b010;

    constexpr static std::uint32_t UNP_ZEROSRC   = 0b001;
    constexpr static std::uint32_t UNP_NEGINFSRC = 0b101;

    constexpr static std::uint32_t UNP_SET_DVALID = 0b111;

    constexpr static std::uint32_t UNP_ZEROSRC_RESET_ALL_BANKS    = 0b1001; // default is clear current bank
    constexpr static std::uint32_t UNP_ZEROSRC_STALL_RESET_WR_RDY = 0b10001;
    constexpr static std::uint32_t UNP_ZEROSRC_SET_DVALID         = 0b1000001;

    constexpr static std::uint32_t UNP0 = 0x0;
    constexpr static std::uint32_t UNP1 = 0x1;
};

struct p_srcb
{
    constexpr static std::uint32_t FORWARD_PASS  = 0x0;
    constexpr static std::uint32_t BACKWARD_PASS = 0x1;
};

constexpr static std::uint32_t SETADC_CH0(std::uint32_t cnt)
{
    return cnt;
}

constexpr static std::uint32_t SETADC_CH1(std::uint32_t cnt)
{
    return cnt << 2;
}

constexpr static std::uint32_t SETADC_CH01(std::uint32_t cnt)
{
    return cnt << 2 | cnt;
}

struct p_setadc
{
    constexpr static std::uint32_t UNP0   = 0b001;
    constexpr static std::uint32_t UNP1   = 0b010;
    constexpr static std::uint32_t UNP_A  = 0b001;
    constexpr static std::uint32_t UNP_B  = 0b010;
    constexpr static std::uint32_t UNP_AB = 0b011;
    constexpr static std::uint32_t PAC    = 0b100;

    constexpr static std::uint32_t SET_X = 0;
    constexpr static std::uint32_t SET_Y = 1;
    constexpr static std::uint32_t SET_Z = 2;
    constexpr static std::uint32_t SET_W = 3;

    constexpr static std::uint32_t X  = 1;
    constexpr static std::uint32_t Y  = 2;
    constexpr static std::uint32_t XY = 3;
    constexpr static std::uint32_t Z  = 1;
    constexpr static std::uint32_t W  = 2;
    constexpr static std::uint32_t ZW = 3;

    constexpr static std::uint32_t CH_0 = 0;
    constexpr static std::uint32_t CH_1 = 1;
};

struct p_pacr
{
    constexpr static std::uint32_t P_ZERO_OUTPUT_DISABLED = 0x0;
    constexpr static std::uint32_t P_ZERO_OUTPUT_ENABLED  = 0x1;
};

struct p_ind
{
    constexpr static std::uint32_t HIER_REGFILE = 0x0;
    constexpr static std::uint32_t HIER_L1      = 0x1;

    constexpr static std::uint32_t INC_NONE = 0x0;
    constexpr static std::uint32_t INC_2B   = 0x1;
    constexpr static std::uint32_t INC_4B   = 0x2;
    constexpr static std::uint32_t INC_16B  = 0x3;

    constexpr static std::uint32_t LD_16B   = 0;
    constexpr static std::uint32_t LD_32bit = 1;
    constexpr static std::uint32_t LD_16bit = 2;
    constexpr static std::uint32_t LD_8bit  = 3;
};

struct p_mov
{
    constexpr static std::uint32_t DEST_NORM    = 0;
    constexpr static std::uint32_t DEST_32B_LOW = 1;
};

struct p_mova2d
{
    constexpr static std::uint32_t MATH_HALO_ROWS = 0x0;
    constexpr static std::uint32_t MOV_1_ROW      = 0x0;
    constexpr static std::uint32_t MOV_8_ROWS     = 0x2;
};

struct p_movd2a
{
    constexpr static std::uint32_t MOV_1_ROW  = 0x0;
    constexpr static std::uint32_t MOV_4_ROWS = 0x2;
};

struct p_movb2d
{
    constexpr static std::uint32_t SRC_ZERO_OFFSET          = 0x0;
    constexpr static std::uint32_t SRC_ROW16_OFFSET         = 0x10;
    constexpr static std::uint32_t MOV_1_ROW                = 0x0;
    constexpr static std::uint32_t MOV_1_ROW_D0_BRCST       = 0x1;
    constexpr static std::uint32_t MOV_8_ROW_BRCST          = 0x2;
    constexpr static std::uint32_t MOV_8_ROW_BRCST_D0_BRCST = 0x3;
    constexpr static std::uint32_t MOV_4_ROWS               = 0x4;
    constexpr static std::uint32_t MOV_4_ROWS_D0_BRCST      = 0x5;
};

struct p_movd2b
{
    constexpr static std::uint32_t SRC_ZERO_OFFSET  = 0x0;
    constexpr static std::uint32_t SRC_ROW16_OFFSET = 0x10;
    constexpr static std::uint32_t MOV_1_ROW        = 0x0;
    constexpr static std::uint32_t MOV_4_ROWS       = 0x2;
};

struct p_movb2a
{
    constexpr static std::uint32_t SRCA_ZERO_OFFSET  = 0x0;
    constexpr static std::uint32_t SRCB_ZERO_OFFSET  = 0x0;
    constexpr static std::uint32_t SRCB_ROW16_OFFSET = 0x10;
    constexpr static std::uint32_t MOV_1_ROW         = 0x0;
    constexpr static std::uint32_t MOV_4_ROWS        = 0x2;
};

struct p_stall
{
    // What to stall on
    constexpr static std::uint32_t NONE    = 0x0;
    constexpr static std::uint32_t THCON   = 0x1;
    constexpr static std::uint32_t UNPACK0 = 0x2;
    constexpr static std::uint32_t UNPACK1 = 0x4;
    constexpr static std::uint32_t UNPACK  = UNPACK0 | UNPACK1;
    constexpr static std::uint32_t PACK0   = 0x8;
    constexpr static std::uint32_t PACK1   = 0x10;
    constexpr static std::uint32_t PACK2   = 0x20;
    constexpr static std::uint32_t PACK3   = 0x40;
    constexpr static std::uint32_t PACK    = PACK0 | PACK1 | PACK2 | PACK3;
    constexpr static std::uint32_t MATH    = 0x80;
    // constexpr static uint SEM_ZERO    = 0x20;
    // constexpr static uint SEM_MAX     = 0x40;
    constexpr static std::uint32_t SRCA_CLR       = 0x100;
    constexpr static std::uint32_t SRCB_CLR       = 0x200;
    constexpr static std::uint32_t SRCA_VLD       = 0x400;
    constexpr static std::uint32_t SRCB_VLD       = 0x800;
    constexpr static std::uint32_t XMOV           = 0x1000;
    constexpr static std::uint32_t TRISC_CFG      = 0x2000;
    constexpr static std::uint32_t SFPU1          = 0x4000;
    constexpr static std::uint32_t WAIT_SFPU      = 0x4000;
    constexpr static std::uint32_t ALL_THREAD_RES = THCON | UNPACK | PACK | MATH | XMOV;

    // What to stall
    constexpr static std::uint32_t STALL_TDMA   = 0x1;
    constexpr static std::uint32_t STALL_SYNC   = 0x2;
    constexpr static std::uint32_t STALL_PACK   = 0x4;
    constexpr static std::uint32_t STALL_UNPACK = 0x8;
    //    constexpr static uint STALL_XSEARCH = 0x10;
    constexpr static std::uint32_t STALL_XMOV   = 0x10;
    constexpr static std::uint32_t STALL_THCON  = 0x20;
    constexpr static std::uint32_t STALL_MATH   = 0x40;
    constexpr static std::uint32_t STALL_CFG    = 0x80;
    constexpr static std::uint32_t STALL_SFPU   = 0x100;
    constexpr static std::uint32_t STALL_THREAD = 0x1ff;

    constexpr static std::uint32_t STALL_ON_ZERO = 0x1;
    constexpr static std::uint32_t STALL_ON_MAX  = 0x2;

    constexpr static std::uint32_t SEMAPHORE_0    = 0x1;
    constexpr static std::uint32_t SEMAPHORE_1    = 0x2;
    constexpr static std::uint32_t SEMAPHORE_2    = 0x4;
    constexpr static std::uint32_t SEMAPHORE_3    = 0x8;
    constexpr static std::uint32_t SEMAPHORE_4    = 0x10;
    constexpr static std::uint32_t SEMAPHORE_5    = 0x20;
    constexpr static std::uint32_t SEMAPHORE_6    = 0x40;
    constexpr static std::uint32_t SEMAPHORE_7    = 0x80;
    constexpr static std::uint32_t SEMAPHORE_BIAS = SEMAPHORE_4;
};

struct p_zeroacc
{
    constexpr static std::uint32_t CLR_SPECIFIC = 0b000;
    constexpr static std::uint32_t CLR_16       = 0b001;
    constexpr static std::uint32_t CLR_HALF     = 0b010;
    constexpr static std::uint32_t CLR_ALL      = 0b011;
    constexpr static std::uint32_t CLR_HALF_32B = 0b110;
    constexpr static std::uint32_t CLR_ALL_32B  = 0b111;
};

struct p_zerosrc
{
    constexpr static std::uint32_t CLR_A  = 0x1;
    constexpr static std::uint32_t CLR_B  = 0x2;
    constexpr static std::uint32_t CLR_AB = 0x3;
};

struct p_shiftx
{
    constexpr static std::uint32_t SHIFT_1 = 0x0;
    constexpr static std::uint32_t SHIFT_2 = 0x1;
    constexpr static std::uint32_t SHIFT_4 = 0x2;
    constexpr static std::uint32_t SHIFT_8 = 0x3;

    constexpr static std::uint32_t RESERVED0    = 0x0;
    constexpr static std::uint32_t RESERVED1    = 0x1;
    constexpr static std::uint32_t RIGHT_AWAY0  = 0x2;
    constexpr static std::uint32_t LEFT_TOWARD0 = 0x3;
};

struct p_cfg
{
    constexpr static std::uint32_t WRCFG_128b = 0x1;
    constexpr static std::uint32_t WRCFG_32b  = 0x0;
};

struct p_alu
{
    constexpr static std::uint32_t AND = 0x0;
    constexpr static std::uint32_t OR  = 0x1;
    constexpr static std::uint32_t XOR = 0x2;
};

struct p_gpool
{
    constexpr static std::uint32_t DIM_1X16  = 0x0;
    constexpr static std::uint32_t DIM_16X16 = 0x1;
    constexpr static std::uint32_t INDEX_DIS = 0x0;
    constexpr static std::uint32_t INDEX_EN  = 0x1;
};

struct p_elwise
{
    constexpr static std::uint32_t SRCB_NO_BCAST  = 0x0;
    constexpr static std::uint32_t DEST_ACCUM_EN  = 0x1;
    constexpr static std::uint32_t DEST_ACCUM_DIS = 0x0;
    constexpr static std::uint32_t SRCB_BCAST_COL = 0x1;
    constexpr static std::uint32_t SRCB_BCAST_ROW = 0x2;
    constexpr static std::uint32_t SRCB_BCAST_ALL = 0x3;

    constexpr static std::uint32_t CLR_A  = 0x1;
    constexpr static std::uint32_t CLR_B  = 0x2;
    constexpr static std::uint32_t CLR_AB = 0x3;
};

struct p_sfpu
{
    // SFPU registers
    constexpr static std::uint32_t LREG0 = 0;
    constexpr static std::uint32_t LREG1 = 1;
    constexpr static std::uint32_t LREG2 = 2;
    constexpr static std::uint32_t LREG3 = 3;
    constexpr static std::uint32_t LREG4 = 4;
    constexpr static std::uint32_t LREG5 = 5;
    constexpr static std::uint32_t LREG6 = 6;
    constexpr static std::uint32_t LREG7 = 7;

    // HW provided constants
    constexpr static std::uint32_t LCONST_0_8373 = 8;
    constexpr static std::uint32_t LCONST_0      = 9;
    constexpr static std::uint32_t LCONST_1      = 10;

    // Programmable constants
    constexpr static std::uint32_t LREG11      = 11;
    constexpr static std::uint32_t LREG12      = 12;
    constexpr static std::uint32_t LREG13      = 13;
    constexpr static std::uint32_t LREG14      = 14;
    constexpr static std::uint32_t LCONST_neg1 = 11;

    constexpr static std::uint32_t LTILEID = 15;

    constexpr static std::uint32_t kCONST_1_FP16B  = 0x3F80;
    constexpr static std::uint32_t kCONST_1_FP16A  = 0x3C00;
    constexpr static std::uint32_t kCONST_0        = 0x0000;
    constexpr static std::uint32_t kCONST_Exp_8Bit = 0;
    constexpr static std::uint32_t kCONST_Exp_5Bit = 1;
};

struct p_sfpswap
{
    // SFPSWAP instruction modes
    constexpr static std::uint32_t UNCONDITIONALLY = 0;
    constexpr static std::uint32_t ALL_ROWS_MAX    = 1;
    constexpr static std::uint32_t ROWS_01_MAX     = 2;
    constexpr static std::uint32_t ROWS_02_MAX     = 3;
    constexpr static std::uint32_t ROWS_03_MAX     = 4;
    constexpr static std::uint32_t ROW_0_MAX       = 5;
    constexpr static std::uint32_t ROW_1_MAX       = 6;
    constexpr static std::uint32_t ROW_2_MAX       = 5;
    constexpr static std::uint32_t ROW_3_MAX       = 6;
};

struct p_exp
{
    constexpr static std::uint32_t FRAC_BITS = 3;
    constexpr static std::uint32_t C23_73    = 0x4340; // Based on FRAC_BITS
    // ADJ_EXP = -0x4300 + 0x003F
    //  0x4300 : 0100 0011 0000 0000
    //  0x003F : 0000 0000 0011 1111
    // -0x4300 : 1011 1101 0000 0000
    // ADJ_EXP : 1011 1101 0011 1111 (-0x4300 + 0x003F = 0xBD3F)
    constexpr static std::uint32_t ADJ_EXP = 0xBD3F;
};

struct p_setdmareg
{
    constexpr static std::uint32_t PAYLOAD_IMMEDIATE   = 0;
    constexpr static std::uint32_t PAYLOAD_16BIT       = 0;
    constexpr static std::uint32_t PAYLOAD_32BIT       = 1;
    constexpr static std::uint32_t PAYLOAD_128BIT      = 2;
    constexpr static std::uint32_t PAYLOAD_TILE_HEADER = 3;

    constexpr static std::uint32_t MODE_IMMEDIATE = 0;
    constexpr static std::uint32_t MODE_SIGNAL    = 1;
};

struct p_mop
{
    constexpr static std::uint32_t MASK_LOOP   = 0;
    constexpr static std::uint32_t DOUBLE_LOOP = 1;
};

struct p_adddmareg
{
    constexpr static std::uint32_t REG_PLUS_REG = 0;
    constexpr static std::uint32_t REG_PLUS_IMM = 1;
};

constexpr static std::uint32_t REG2FLOP_FLOP_INDEX(std::uint32_t addr)
{
    return addr - THCON_CFGREG_BASE_ADDR32;
}

struct p_reg2flop
{
    constexpr static std::uint32_t WRITE_16B = 0;
    constexpr static std::uint32_t WRITE_4B  = 1;
    constexpr static std::uint32_t WRITE_2B  = 2;
    constexpr static std::uint32_t WRITE_1B  = 3;
};

} // namespace ckernel
