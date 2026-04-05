// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Hand-coded parameter encoding for various common instructions
#include <cstdint>

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
    constexpr static std::uint32_t SET_ABD_F = 0xf;
    constexpr static std::uint32_t SET_AB_F  = 0xb;

    constexpr static std::uint32_t CR_A         = 0x1;
    constexpr static std::uint32_t CR_B         = 0x2;
    constexpr static std::uint32_t CR_D         = 0x4;
    constexpr static std::uint32_t C_TO_CR_MODE = 0x8;
    constexpr static std::uint32_t CR_ABD       = 0x7;
};

struct p_setibrwc
{
    constexpr static std::uint32_t SET_BIAS = 0x0;
    constexpr static std::uint32_t INC_BIAS = 0x0;
    constexpr static std::uint32_t CR_NONE  = 0x0;
    constexpr static std::uint32_t CR_BIAS  = 0x1;
};

struct p_unpacr
{
    constexpr static std::uint32_t UNP_A    = 0b000;
    constexpr static std::uint32_t UNP_B    = 0b001;
    constexpr static std::uint32_t UNP_S    = 0b010;
    constexpr static std::uint32_t UNP_DEST = 0b011;

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

    constexpr static std::uint32_t UNP_POP           = 0x0;
    constexpr static std::uint32_t UNP_CLRSRC        = 0x1;
    constexpr static std::uint32_t UNP_NOP           = 0x2;
    constexpr static std::uint32_t UNP_POP_STREAM    = 0x3;
    constexpr static std::uint32_t UNP_CLRSRC_ZERO   = 0x0;
    constexpr static std::uint32_t UNP_CLRSRC_NEGINF = 0x1;
    constexpr static std::uint32_t UNP_CLRSRC_ONE    = 0x2;
    constexpr static std::uint32_t UNP_CLRSRC_IMM    = 0x3;

    constexpr static std::uint32_t UNP_STALL_FPU_RD = 0x0;
    constexpr static std::uint32_t UNP_STALL_UNP_WR = 0x1;

    constexpr static std::uint32_t UNP_CLRSRC_RESET_ALL_BANKS = 0x1;
    constexpr static std::uint32_t UNP_CLRSRC_ONE_FP16A       = 0x0;
    constexpr static std::uint32_t UNP_CLRSRC_ONE_FP16B       = 0x1;
    constexpr static std::uint32_t UNP_CLRSRC_ONE_TF32        = 0x1;
    constexpr static std::uint32_t UNP_CLRSRC_ONE_INT8        = 0x2;
};

struct p_set_inc_sel
{
    constexpr static std::uint32_t TILE_SEL = 0b000;
    constexpr static std::uint32_t FACE_SEL = 0b001;
    constexpr static std::uint32_t ROW_SEL  = 0b010;
};

struct p_srcb
{
    constexpr static std::uint32_t FORWARD_PASS  = 0x0;
    constexpr static std::uint32_t BACKWARD_PASS = 0x1;
};

struct p_setadc
{
    // RT: Is this outdated?
    constexpr static std::uint32_t UNP0 = 0b001;
    constexpr static std::uint32_t UNP1 = 0b010;
    constexpr static std::uint32_t PAC  = 0b100;

    constexpr static std::uint32_t SET_X = 0;
    constexpr static std::uint32_t SET_Y = 1;
    constexpr static std::uint32_t SET_Z = 2;
    constexpr static std::uint32_t SET_W = 3;

    constexpr static std::uint32_t CH_0 = 0;
    constexpr static std::uint32_t CH_1 = 1;
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

struct p_movd2a
{
    constexpr static std::uint32_t MOV_1_ROW  = 0x0;
    constexpr static std::uint32_t MOV_4_ROWS = 0x1;
    constexpr static std::uint32_t MOV_8_ROWS = 0x2;
};

struct p_movd2b
{
    constexpr static std::uint32_t MOV_1_ROW  = 0x0;
    constexpr static std::uint32_t MOV_4_ROWS = 0x1;
    constexpr static std::uint32_t MOV_8_ROWS = 0x2;

    constexpr static std::uint32_t SRC_ROW16_OFFSET = 0x10;
    constexpr static std::uint32_t SRC_ROW32_OFFSET = 0x20;
};

struct p_movb2a
{
    constexpr static std::uint32_t MOV_1_ROW   = 0x0;
    constexpr static std::uint32_t MOV_4_ROWS  = 0x1;
    constexpr static std::uint32_t MOV_8_ROWS  = 0x2;
    constexpr static std::uint32_t MOV_16_ROWS = 0x3;

    constexpr static std::uint32_t SRCA_ZERO_OFFSET  = 0x0;
    constexpr static std::uint32_t SRCB_ROW16_OFFSET = 0x10;
    constexpr static std::uint32_t SRCB_ROW32_OFFSET = 0x20;
};

struct p_mov_src_to_dest
{
    constexpr static std::uint32_t MOV_1_ROW   = 0x0;
    constexpr static std::uint32_t MOV_4_ROWS  = 0x1;
    constexpr static std::uint32_t MOV_8_ROWS  = 0x2;
    constexpr static std::uint32_t MOV_16_ROWS = 0x3;

    constexpr static std::uint32_t SRC_ROW16_OFFSET = 0x10;
    constexpr static std::uint32_t SRC_ROW32_OFFSET = 0x20;
};

struct p_stall
{
    // What to stall on
    constexpr static std::uint32_t NOTHING         = 0;
    constexpr static std::uint32_t THCON           = 1;
    constexpr static std::uint32_t UNPACK0         = 2;
    constexpr static std::uint32_t UNPACK0_DONE_RD = 3;
    constexpr static std::uint32_t UNPACK1         = 4;
    constexpr static std::uint32_t UNPACK1_DONE_RD = 5;
    constexpr static std::uint32_t UNPACK2         = 6;
    constexpr static std::uint32_t UNPACK2_DONE_RD = 7;
    constexpr static std::uint32_t PACK0           = 8;
    constexpr static std::uint32_t PACK0_DONE_WR   = 9;
    constexpr static std::uint32_t PACK1           = 10;
    constexpr static std::uint32_t PACK1_DONE_WR   = 11;
    constexpr static std::uint32_t MATH            = 12;
    constexpr static std::uint32_t SRCA_CLR        = 13;
    constexpr static std::uint32_t SRCB_CLR        = 14;
    constexpr static std::uint32_t UNPACK_SRCS_RDY = 15;
    constexpr static std::uint32_t PACK_SRCS_RDY   = 16;
    constexpr static std::uint32_t SRCA_VLD        = 17;
    constexpr static std::uint32_t SRCB_VLD        = 18;
    constexpr static std::uint32_t SFPU_SRCS_RDY   = 19;
    constexpr static std::uint32_t XMOV            = 20;
    constexpr static std::uint32_t TRISC_CFG       = 21;
    constexpr static std::uint32_t SFPU1           = 22; // lol name collisions
    constexpr static std::uint32_t CFGEXU          = 23;

    constexpr static std::uint32_t WAIT_SFPU = SFPU1;
    constexpr static std::uint32_t PACK      = PACK0;

    constexpr static struct
    {
        int dont_use_this;
    } ALL_THREAD_RES = {0}; // Use non-integer so that compiler throws an error if you try to use this

    // What to stall
    constexpr static std::uint32_t STALL_TDMA   = 0x1;
    constexpr static std::uint32_t STALL_SYNC   = 0x2;
    constexpr static std::uint32_t STALL_PACK   = 0x4;
    constexpr static std::uint32_t STALL_UNPACK = 0x8;
    // constexpr static uint STALL_XSEARCH     = 0x10;
    constexpr static std::uint32_t STALL_XMOV   = 0x10;
    constexpr static std::uint32_t STALL_THCON  = 0x20;
    constexpr static std::uint32_t STALL_MATH   = 0x40;
    constexpr static std::uint32_t STALL_CFG    = 0x80;
    constexpr static std::uint32_t STALL_SFPU   = 0x100;
    constexpr static std::uint32_t STALL_THREAD = 0x1ff;

    constexpr static std::uint32_t STALL_ON_ZERO = 0x1;
    constexpr static std::uint32_t STALL_ON_MAX  = 0x2;

    constexpr static std::uint32_t SEMAPHORE_0 = 0x1;
    constexpr static std::uint32_t SEMAPHORE_1 = 0x2;
    constexpr static std::uint32_t SEMAPHORE_2 = 0x4;
    constexpr static std::uint32_t SEMAPHORE_3 = 0x8;
    constexpr static std::uint32_t SEMAPHORE_4 = 0x10;
    constexpr static std::uint32_t SEMAPHORE_5 = 0x20;
    constexpr static std::uint32_t SEMAPHORE_6 = 0x40;
    constexpr static std::uint32_t SEMAPHORE_7 = 0x80;
};

struct p_zeroacc
{
    constexpr static std::uint32_t CLR_SPECIFIC       = 0x0;
    constexpr static std::uint32_t CLR_16             = 0x1;
    constexpr static std::uint32_t CLR_HALF           = 0x2;
    constexpr static std::uint32_t CLR_ALL            = 0x3;
    constexpr static std::uint32_t CLR_STRIPED_Z_FACE = 0x4;
};

struct p_zerosrc
{
    constexpr static std::uint32_t CLR_A  = 0x1;
    constexpr static std::uint32_t CLR_B  = 0x2;
    constexpr static std::uint32_t CLR_AB = 0x3;

    constexpr static std::uint32_t CURR_BANK = 0x0;
    constexpr static std::uint32_t ALL_BANKS = 0x1;

    constexpr static std::uint32_t READ_BANK  = 0x1;
    constexpr static std::uint32_t WRITE_BANK = 0x0;
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

    constexpr static std::uint32_t CLR_NONE      = 0x0;
    constexpr static std::uint32_t CLR_SRCA_VLD  = 0x1;
    constexpr static std::uint32_t CLR_SRCB_VLD  = 0x2;
    constexpr static std::uint32_t CLR_SRCAB_VLD = 0x3;
};

struct p_elwise
{
    constexpr static std::uint32_t SRCB_NO_BCAST  = 0x0;
    constexpr static std::uint32_t SRCB_BCAST_COL = 0x1;
    constexpr static std::uint32_t SRCB_BCAST_ROW = 0x2;
    constexpr static std::uint32_t SRCB_BCAST_ALL = 0x3;

    constexpr static std::uint32_t DISABLE_ACCUM = 0x0;
    constexpr static std::uint32_t ENABLE_ACCUM  = 0x1;

    constexpr static std::uint32_t CLR_NONE      = 0x0;
    constexpr static std::uint32_t CLR_SRCA_VLD  = 0x1;
    constexpr static std::uint32_t CLR_SRCB_VLD  = 0x2;
    constexpr static std::uint32_t CLR_SRCAB_VLD = 0x3;
};

struct p_sfpu
{
    constexpr static std::uint32_t LREG0       = 0;
    constexpr static std::uint32_t LREG1       = 1;
    constexpr static std::uint32_t LREG2       = 2;
    constexpr static std::uint32_t LREG3       = 3;
    constexpr static std::uint32_t LREG4       = 4;
    constexpr static std::uint32_t LREG5       = 5;
    constexpr static std::uint32_t LREG6       = 6;
    constexpr static std::uint32_t LREG7       = 7;
    constexpr static std::uint32_t LCONST_0    = 9;
    constexpr static std::uint32_t LCONST_1    = 10;
    constexpr static std::uint32_t LCONST_neg1 = 11;

    struct sfpmem
    {
        constexpr static std::uint32_t DEFAULT =
            0b0000; // format is determined by combination of SrcB exponent width of ALU_FORMAT_SPEC_REG and also ACC_CTRL_SFPU_Fp32
        constexpr static std::uint32_t FP16A  = 0b0001; // stored data will be interpreted as fp16 (fp16_a) format
        constexpr static std::uint32_t FP16B  = 0b0010; // stored data will be interpreted as bfloat (fp16_b) format
        constexpr static std::uint32_t FP32   = 0b0011; // stored data will be interpreted as fp32 format
        constexpr static std::uint32_t INT32  = 0b0100; // stored data will be interpreted as int32 (sign + magnitude) format
        constexpr static std::uint32_t UINT8  = 0b0101; // stored data will be interpreted as unsigned int8 format
        constexpr static std::uint32_t UINT16 = 0b0110; // stored data will be interpreted as unsigned int16 format
                                                        // TODO - Luka: add the other formats
    };

    struct mad_mode
    {
        constexpr static std::uint32_t INDEX_ADDR_D  = 0x4;
        constexpr static std::uint32_t INDEX_ADDR_A  = 0x8;
        constexpr static std::uint32_t INDEX_ADDR_AD = 0xC;
    };

    struct col_offset
    {
        constexpr static std::uint32_t EVEN_COL = 0x0;
        constexpr static std::uint32_t ODD_COL  = 0x2;
    };

    struct cc
    {
        constexpr static std::uint32_t SET_CC    = 0x2;
        constexpr static std::uint32_t CLR_CC    = 0x1;
        constexpr static std::uint32_t SET_CC_EN = 0x1;
        constexpr static std::uint32_t CLR_CC_EN = 0x0;
    };

    struct sfp_stochrnd_mod
    {
        constexpr static std::uint32_t FP32_TO_FP16A  = 0x0;
        constexpr static std::uint32_t FP32_TO_FP16B  = 0x1;
        constexpr static std::uint32_t FP32_TO_UINT8  = 0x2;
        constexpr static std::uint32_t FP32_TO_INT8   = 0x3;
        constexpr static std::uint32_t INT32_TO_UINT8 = 0x4;
        constexpr static std::uint32_t INT32_TO_INT8  = 0x5;
        constexpr static std::uint32_t FP32_TO_UINT16 = 0x6; // TODO: does Uint16 even exist?
        constexpr static std::uint32_t FP32_TO_INT16  = 0x7;
    };

    struct sfp_stochrnd_rnd_mod
    {
        constexpr static std::uint32_t NearEven   = 0x0;
        constexpr static std::uint32_t Stochastic = 0x1;
        constexpr static std::uint32_t RoundZero  = 0x2;
    };
};

struct p_cleardvalid
{
    constexpr static std::uint32_t UNPACK_TO_DEST = 0b0001;
    constexpr static std::uint32_t FPU            = 0b0010;
    constexpr static std::uint32_t SFPU           = 0b0100;
    constexpr static std::uint32_t PACK           = 0b1000;

    constexpr static std::uint32_t CLR_NONE      = 0b00;
    constexpr static std::uint32_t CLR_SRCA_VLD  = 0b01;
    constexpr static std::uint32_t CLR_SRCB_VLD  = 0b10;
    constexpr static std::uint32_t CLR_SRCAB_VLD = 0b11;
};

struct p_pacr
{
    constexpr static std::uint32_t PACK0 = 0b011;
    constexpr static std::uint32_t PACK1 = 0b100;

    constexpr static std::uint32_t DST_ACCESS_NORMAL_MODE  = 0b0;
    constexpr static std::uint32_t DST_ACCESS_STRIDED_MODE = 0b1;

    constexpr static std::uint32_t NO_ROW_PAD_ZERO                          = 0b000;
    constexpr static std::uint32_t ROW_PAD_ZERO_ALL_PACR                    = 0b001;
    constexpr static std::uint32_t ROW_PAD_ZERO_ALL_PACR_16DATUM_ALGN       = 0b101;
    constexpr static std::uint32_t ROW_PAD_ZERO_NO_CONCAT_PACR              = 0b010;
    constexpr static std::uint32_t ROW_PAD_ZERO_NO_CONCAT_PACR_16DATUM_ALGN = 0b110;
    constexpr static std::uint32_t ROW_PAD_ZERO_LAST_PACR                   = 0b011;
    constexpr static std::uint32_t ROW_PAD_ZERO_LAST_PACR_16DATUM_ALGN      = 0b111;

    constexpr static std::uint32_t CFG_CTXT_0 = 0b00;
    constexpr static std::uint32_t CFG_CTXT_1 = 0b01;
    constexpr static std::uint32_t CFG_CTXT_2 = 0b10;
    constexpr static std::uint32_t CFG_CTXT_3 = 0b11;

    constexpr static std::uint32_t ADDR_CNT_CTXT_0 = 0b00;
    constexpr static std::uint32_t ADDR_CNT_CTXT_1 = 0b01;
    constexpr static std::uint32_t ADDR_CNT_CTXT_2 = 0b10;

    constexpr static std::uint32_t ZERO_WRITE    = 0b1;
    constexpr static std::uint32_t NO_ZERO_WRITE = 0b0;

    constexpr static std::uint32_t NO_CTXT_CTRL               = 0b00;
    constexpr static std::uint32_t RTL_FLOPS_CTXT_SEL         = 0b01;
    constexpr static std::uint32_t RTL_FLOPS_CTXT_RST_AND_NOP = 0b10;
    constexpr static std::uint32_t RTL_FLOPS_CTXT_SEL_NO_RST  = 0b11;
};

struct p_ttsync
{
    // Resource bits.
    // These must line up with the definition of ttsync_instrn_rsrc_t in tt_tensix_pkg.sv .
    constexpr static std::uint32_t CFG_RD = 1 << 0;
    constexpr static std::uint32_t CFG_WR = 1 << 1;
    constexpr static std::uint32_t GPR_RD = 1 << 2;
    constexpr static std::uint32_t GPR_WR = 1 << 3;

    // Class numbers
    //<ttsync_class_numbers> //Please do not remove this line, it is used by gen_tt_instrn_resources_used.lua
    constexpr static std::uint32_t DEFAULT_CLASS = 0; // Everything else
    constexpr static std::uint32_t ATOMICS_CLASS = 1; // ATCAS, ATGETM, ATINCGET, ATINCGETPTR, ATRELM, ATSWAP
    constexpr static std::uint32_t CFG_CLASS =
        2; // CFGSHIFTMASK, CFGSHIFTMASK_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE, RMWCIB0,
           // RMWCIB0_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE, RMWCIB1, RMWCIB1_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE, RMWCIB2,
           // RMWCIB2_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE, RMWCIB3_BUT_ALIAS_BIT_8_OF_CFG_REG_ADDR_WITH_LSB_OF_OPCODE
    constexpr static std::uint32_t GPR_CLASS     = 3; // ADDGPR, BITWOPGPR, CMPGPR, MULGPR, SETGPR, SHIFTGPR, SUBGPR
    constexpr static std::uint32_t LDSTIND_CLASS = 4; // LOADIND, STOREIND
    constexpr static std::uint32_t LOADREG_CLASS = 5; // LOADREG
    constexpr static std::uint32_t MOP_CLASS     = 6; // MOP
    constexpr static std::uint32_t PACK_CLASS    = 7; // PACR0_FACE, PACR0_FACE_INC, PACR0_ROW, PACR0_ROW_INC, PACR0_TILE, PACR0_TILE_INC, PACR1_FACE,
                                                   // PACR1_FACE_INC, PACR1_ROW, PACR1_ROW_INC, PACR1_TILE, PACR1_TILE_INC, PACR_STRIDE, PACR_UNTILIZE, RV_PACR
    constexpr static std::uint32_t RDCFG_CLASS    = 8;  // RDCFG
    constexpr static std::uint32_t REPLAY_CLASS   = 9;  // REPLAY
    constexpr static std::uint32_t RV_WRCFG_CLASS = 10; // RV_WRCFG
    constexpr static std::uint32_t SHADOW_CLASS   = 11; // COMMIT_SHADOW
    constexpr static std::uint32_t STOREREG_CLASS = 12; // STOREREG
    constexpr static std::uint32_t UNPACK_CLASS =
        13; // RV_UNPACR, UNPACR0_FACE, UNPACR0_FACE_INC, UNPACR0_ROW, UNPACR0_ROW_INC, UNPACR0_STRIDE, UNPACR0_TILE, UNPACR0_TILE_INC, UNPACR1_FACE,
            // UNPACR1_FACE_INC, UNPACR1_ROW, UNPACR1_ROW_INC, UNPACR1_STRIDE, UNPACR1_TILE, UNPACR1_TILE_INC, UNPACR2_FACE, UNPACR2_FACE_INC, UNPACR2_ROW,
            // UNPACR2_ROW_INC, UNPACR2_STRIDE, UNPACR2_TILE, UNPACR2_TILE_INC, UNPACR_DEST_FACE, UNPACR_DEST_FACE_INC, UNPACR_DEST_ROW, UNPACR_DEST_ROW_INC,
            // UNPACR_DEST_STRIDE, UNPACR_DEST_TILE, UNPACR_DEST_TILE_INC, UNPACR_NOP, UNPACR_TILE_MISC, UNPACR_TILIZE
    constexpr static std::uint32_t WRCFG_CLASS = 14; // WRCFG
                                                     //</ttsync_class_numbers> //Please do not remove this line, it is used by gen_tt_instrn_resources_used.lua
};

struct p_sfpnonlinear
{
    constexpr static std::uint32_t RECIP_MODE = 0x0;
    constexpr static std::uint32_t RELU_MODE  = 0x2;
    constexpr static std::uint32_t SQRT_MODE  = 0x3;
    constexpr static std::uint32_t EXP_MODE   = 0x4;
    constexpr static std::uint32_t TANH_MODE  = 0x5;
};

} // namespace ckernel
