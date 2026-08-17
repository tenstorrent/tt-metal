// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 ckernel proj params header file for project quasar_n4

 This file was auto-generated from proj_config.yml using src/meta/reg_flow/create_ckernel_proj_params.py

*/
#pragma once

#define TRISC_COUNT             0x00000004 // = 4 in decimal
#define NEO_COUNT               0x00000004 // = 4 in decimal
#define L1_SIZE_IN_BYTES        0x00400000 // = 4194304 in decimal
#define L1_CFG_ID               0x00000001 // = 1 in decimal
#define ENABLE_PARITY           0x00000000 // = 0 in decimal
#define ENABLE_INT8_PACKING     0x00000000 // = 0 in decimal
#define HALF_FP_BW              0x00000000 // = 0 in decimal
#define INT_32_1_ENABLED        0x00000000 // = 0 in decimal
#define INT_32_1_SIGN_MAGNITUDE 0x00000000 // = 0 in decimal
#define ENABLE_FP4_PACKING      0x00000001 // = 1 in decimal
#define MATH_ROWS               0x00000008 // = 8 in decimal
#define L1_CLIENT_DISC          0x00000000 // = 0 in decimal
#define FPU_SELF_CHECK_ENABLED  0x00000000 // = 0 in decimal
#define SFPU_SELF_CHECK_ENABLED 0x00000000 // = 0 in decimal
#define TDMA_SELF_CHECK_ENABLED 0x00000000 // = 0 in decimal
#define TRISC_PORT_CNT          0x00000001 // = 1 in decimal
#define TRISC_SUB_PORT_CNT      0x00000005 // = 5 in decimal
#define UNPACK_RD_PORT_CNT      0x00000005 // = 5 in decimal
#define PACK_WR_PORT_CNT        0x00000003 // = 3 in decimal
#define NOC_RD_PORT_CNT         0x00000004 // = 4 in decimal
#define NOC_WR_PORT_CNT         0x00000004 // = 4 in decimal
#define OVRLY_RW_PORT_CNT       0x00000001 // = 1 in decimal
#define OVRLY_RW_SUB_PORT_CNT   0x00000006 // = 6 in decimal
#define OVRLY_RD_PORT_CNT       0x00000002 // = 2 in decimal
#define OVRLY_WR_PORT_CNT       0x00000002 // = 2 in decimal
