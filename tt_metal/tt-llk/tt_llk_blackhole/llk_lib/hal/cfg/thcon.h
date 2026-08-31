// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// offsets source: tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h
#pragma once

#include "field.h"

namespace hal
{
namespace cfg
{
// ============================================================
// THCON
// ============================================================

struct ThconReg0Selector
{
};

struct ThconReg1Selector
{
};

struct ThconReg2Selector
{
};

struct ThconReg3Selector
{
};

struct ThconReg4Selector
{
};

struct ThconReg5Selector
{
};

struct ThconReg6Selector
{
};

struct ThconReg7Selector
{
};

struct ThconReg8Selector
{
};

struct ThconReg9Selector
{
};

struct ThconReg10Selector
{
};

struct ThconReg11Selector
{
};

inline constexpr ThconReg0Selector Reg0 {};
inline constexpr ThconReg1Selector Reg1 {};
inline constexpr ThconReg2Selector Reg2 {};
inline constexpr ThconReg3Selector Reg3 {};
inline constexpr ThconReg4Selector Reg4 {};
inline constexpr ThconReg5Selector Reg5 {};
inline constexpr ThconReg6Selector Reg6 {};
inline constexpr ThconReg7Selector Reg7 {};
inline constexpr ThconReg8Selector Reg8 {};
inline constexpr ThconReg9Selector Reg9 {};
inline constexpr ThconReg10Selector Reg10 {};
inline constexpr ThconReg11Selector Reg11 {};

struct ThconTileDescriptorFields
{
    static constexpr Field Raw {RegisterFile::State, 32, 64, 0, 0, 128, 2, 1536};           // Complete four-word descriptor
    static constexpr Field InDataFormat {RegisterFile::State, 32, 64, 0, 0, 4, 2, 1536};    // Unpacker input data format (4b)
    static constexpr Field Uncompressed {RegisterFile::State, 32, 64, 0, 4, 1, 2, 1536};    // Input tile is uncompressed (1b)
    static constexpr Field BlobsPerXyPlane {RegisterFile::State, 32, 64, 0, 8, 4, 2, 1536}; // Number of blobs per XY plane (4b)
    static constexpr Field XDim {RegisterFile::State, 32, 64, 0, 16, 16, 2, 1536};          // Tile X dimension (16b)
    static constexpr Field YDim {RegisterFile::State, 32, 64, 1, 0, 16, 2, 1536};           // Tile Y dimension (16b)
    static constexpr Field ZDim {RegisterFile::State, 32, 64, 1, 16, 16, 2, 1536};          // Tile Z dimension (16b)
    static constexpr Field WDim {RegisterFile::State, 32, 64, 2, 0, 16, 2, 1536};           // Tile W dimension (16b)
    static constexpr Field BlobsYStartLo {RegisterFile::State, 32, 64, 2, 16, 16, 2, 1536}; // Low 16 bits of blob Y-start values (16b)
    static constexpr Field BlobsYStartHi {RegisterFile::State, 32, 64, 3, 0, 16, 2, 1536};  // High 16 bits of blob Y-start values (16b)
    static constexpr Field DigestType {RegisterFile::State, 32, 64, 3, 16, 8, 2, 1536};     // Digest type; unused (8b)
    static constexpr Field DigestSize {RegisterFile::State, 32, 64, 3, 24, 8, 2, 1536};     // Digest size; unused (8b)
};

struct ThconReg0Fields
{
    // cfg_defines.h exposes only the 128-bit anchor. Keep the physical layout
    // here so callers can name logical fields without masks, shifts, or word offsets.
    static constexpr ThconTileDescriptorFields TileDescriptor {};
};

struct ThconReg1Fields
{
    static constexpr Field Row_start_section_size {
        RegisterFile::State, 32, 64, 4, 0, 16, 2, 1536}; // Packer row-start section size (aligned to 16B word) - regs sets 0/2 (16b)
    static constexpr Field Exp_section_size {
        RegisterFile::State, 32, 64, 4, 16, 16, 2, 1536}; // Packer exp section size (aligned to 16B words) - regs sets 0/2 (16b)
    static constexpr Field L1_Dest_addr {
        RegisterFile::State, 32, 64, 5, 0, 32, 2, 1536}; // Packer destination address for packer (aligned to 16B word) - regs sets 0/2 (32b)
    static constexpr Field Disable_zero_compress {RegisterFile::State, 32, 64, 6, 0, 1, 2, 1536}; // Packer disable zero compression  - regs sets 0/2 (1b)
    static constexpr Field Add_l1_dest_addr_offset {
        RegisterFile::State, 32, 64, 6, 1, 1, 2, 1536}; // Packer add accumulated tile size to L1 destination address  - regs sets 0/2 (1b)
    static constexpr Field Disable_pack_zero_flags {
        RegisterFile::State, 32, 64, 6, 2, 1, 2, 1536}; // Disable setting zero flags in tile header during packing (1b)
    static constexpr Field ovrd_default_throttle_mode {
        RegisterFile::State, 32, 64, 6, 3, 1, 2, 1536}; // Override Unpacker default throttle module with value in cfg reg (1b)
    static constexpr Field Out_data_format {RegisterFile::State, 32, 64, 6, 4, 4, 2, 1536}; // Packer output data format - regs sets 0/2 (4b)
    static constexpr Field In_data_format {RegisterFile::State, 32, 64, 6, 8, 4, 2, 1536};  // Packer input data format- regs sets 0/2 (4b)
    static constexpr Field Dis_shared_exp_assembler {
        RegisterFile::State, 32, 64, 6, 12, 1, 2, 1536}; // Disable packer shared exponent assembler for BFP formats - regs sets 0/2 (1b)
    static constexpr Field Auto_set_last_pacr_intf_sel {
        RegisterFile::State, 32, 64, 6, 13, 1, 2, 1536}; // Auto set PACR[ReadIntfSel] for last PACR per plane using Packer tile counters (1b)
    static constexpr Field Enable_out_fifo {RegisterFile::State, 32, 64, 6, 14, 1, 2, 1536}; // Store tile size and zero mask into FIFO (1b)
    static constexpr Field Sub_l1_tile_header_size {
        RegisterFile::State, 32, 64, 6, 15, 1, 2, 1536}; // Subtract size of the tile header from packer L1 dest address (debug only) by default packer adds 16B
                                                         // to programmed L1 dest address to auto include tile header size (1b)
    static constexpr Field Source_interface_selection {
        RegisterFile::State, 32, 64, 6, 16, 1, 2, 1536}; // Packer source: 0 - Dest register file, 1 - L1  ;  regs sets 0/2 (1b)
    static constexpr Field pack_start_intf_pos {RegisterFile::State, 32, 64, 6, 17, 4, 2, 1536}; // 2-bit interface position for first PACR in a tile  - Section
                                                                                                 // 0 for contexts 0/1, Section 1 for contexts 2/3 (4b)
    static constexpr Field All_pack_disable_zero_compress_ovrd {RegisterFile::State, 32, 64, 6, 21, 1, 2, 1536}; // UNUSED - REVISIT Rakesh Cleanup (1b)
    static constexpr Field Add_tile_header_size {
        RegisterFile::State, 32, 64, 6, 22, 1, 2, 1536}; // Add size of the tile header to the tile size when tile size accumulation is enabled (1b)
    static constexpr Field pack_dis_y_pos_start_offset {
        RegisterFile::State, 32, 64, 6, 23, 1, 2, 1536}; // Disable y start position offset for edge-masking selection (1b)
    static constexpr Field L1_source_addr {
        RegisterFile::State, 32, 64, 6, 24, 8, 2, 1536}; // Packer L1 source address bit 20:13 when source is L1 - regs sets 0/2 (8b)
    static constexpr Field Downsample_mask {RegisterFile::State, 32, 64, 7, 0, 16, 2, 1536}; // Packer down-sample mask (0x555, 0x1111, 0x0101, 0x0001, etc., as
                                                                                             // down-sample rate requires - regs sets 0/2 (16b)
    static constexpr Field Downsample_rate {RegisterFile::State, 32, 64, 7, 16, 3, 2, 1536}; // Packer down-sample rate (0, 1, 2) - regs sets 0/2 (3b)
    static constexpr Field Pack_L1_Acc {
        RegisterFile::State, 32, 64, 7, 19, 1, 2, 1536}; // Send Acc command to L1 from Packer to perform Accumulation - regs sets 0/2 (1b)
    static constexpr Field Exp_threshold_en {
        RegisterFile::State, 32, 64, 7, 20, 1, 2, 1536}; // Packer force exp to zero if it's below the threshold specified - regs sets 0/2 (1b)
    static constexpr Field Unp_LF8_4b_exp {RegisterFile::State, 32, 64, 7, 22, 1, 2, 1536}; // Unpacker 0/1 LF8 4-bit exp (1b)
    static constexpr Field Pac_LF8_4b_exp {
        RegisterFile::State, 32, 64, 7, 23, 1, 2, 1536}; // Packer LF8 4-bit exp ; set 0 - Packer 0/1 ; set 1 - Packer 2/3 (1b)
    static constexpr Field Exp_threshold {RegisterFile::State, 32, 64, 7, 24, 8, 2, 1536}; // Packer exp threshold - regs sets 0/2 (8b)
};

struct ThconReg2Fields
{
    static constexpr Field Out_data_format {RegisterFile::State, 32, 64, 8, 0, 4, 2, 1536}; // Unpacker 0/1 out data format (4b)
    static constexpr Field Throttle_mode {
        RegisterFile::State, 32, 64, 8, 4, 2, 2, 1536}; // Unpacker 0/1 high perf mode (enable 512-bit unpacker l1 read interface (2b)
    static constexpr Field Context_count {
        RegisterFile::State, 32, 64, 8, 6, 2, 2, 1536}; // Unpacker 0/1 number of available contexts when auto increment context mode is used in unpack
                                                        // instruction. Counter will wrap around when value is equal to (1<<Contex_count)-1 (2b)
    static constexpr Field Haloize_mode {RegisterFile::State, 32, 64, 8, 8, 1, 2, 1536}; // Enable unpacker 0 data transpose (1b)
    static constexpr Field Tileize_mode {
        RegisterFile::State, 32, 64, 8, 9, 1, 2, 1536}; // Unpacker 0/1 Enable tile making mode. Unpacker applies stride to l1 read address. (1b)
    static constexpr Field Unpack_Src_Reg_Set_Upd {RegisterFile::State, 32, 64, 8, 10, 1, 2, 1536}; // Unpacker 0/1 Src Reg Set Update (1b)
    static constexpr Field Unpack_If_Sel {RegisterFile::State, 32, 64, 8, 11, 1, 2, 1536};          // Unpacker 0 interface selection. 0: src, 1: dst (1b)
    static constexpr Field Upsample_rate {
        RegisterFile::State, 32, 64, 8, 12, 2, 2, 1536}; // Upsample rate: 0 -> rate 1 (no upsample), 1 -> rate 2, 2-> rate 4 (2b)
    static constexpr Field Ovrd_data_format {
        RegisterFile::State, 32, 64, 8, 14, 1, 2, 1536}; // Use per context tile data format. If set use input/output data format programmed in
                                                         // REG7_Unpack_data_format_cntx* as tile input/output format (1b)
    static constexpr Field Upsample_and_interleave {
        RegisterFile::State, 32, 64, 8, 15, 1, 2, 1536}; // Unpacker 0/1 set byte enables when upsampling to allow interleaving of multiple tiles (1b)
    static constexpr Field Shift_amount_cntx0 {RegisterFile::State, 32, 64, 8, 16, 4, 2, 1536}; // Unpacker 0 shift data to the left by specified amount for
                                                                                                // context 0 before it's written to SRCA registers (4b)
    static constexpr Field Shift_amount_cntx1 {RegisterFile::State, 32, 64, 8, 20, 4, 2, 1536}; // Unpacker 0 shift data to the left by specified amount for
                                                                                                // context 1 before it's written to SRCA registers (4b)
    static constexpr Field Shift_amount_cntx2 {RegisterFile::State, 32, 64, 8, 24, 4, 2, 1536}; // Unpacker 0 shift data to the left by specified amount for
                                                                                                // context 2 before it's written to SRCA registers (4b)
    static constexpr Field Shift_amount_cntx3 {RegisterFile::State, 32, 64, 8, 28, 4, 2, 1536}; // Unpacker 0 shift data to the left by specified amount for
                                                                                                // context 3 before it's written to SRCA registers (4b)
    static constexpr Field Disable_zero_compress_cntx0 {
        RegisterFile::State, 32, 64, 9, 0, 1, 2, 1536}; // Unpacker 0/1 disable zero compression for context 0 (1b)
    static constexpr Field Disable_zero_compress_cntx1 {
        RegisterFile::State, 32, 64, 9, 1, 1, 2, 1536}; // Unpacker 0/1 disable zero compression for context 1 (1b)
    static constexpr Field Disable_zero_compress_cntx2 {
        RegisterFile::State, 32, 64, 9, 2, 1, 2, 1536}; // Unpacker 0 disable zero compression for context 2 (1b)
    static constexpr Field Disable_zero_compress_cntx3 {
        RegisterFile::State, 32, 64, 9, 3, 1, 2, 1536}; // Unpacker 0 disable zero compression for context 3 (1b)
    static constexpr Field Unpack_if_sel_cntx0 {
        RegisterFile::State, 32, 64, 9, 4, 1, 2, 1536}; // Unpacker 0 interface selection. 0: src, 1: dst for context 0 (1b)
    static constexpr Field Unpack_if_sel_cntx1 {
        RegisterFile::State, 32, 64, 9, 5, 1, 2, 1536}; // Unpacker 0 interface selection. 0: src, 1: dst for context 1 (1b)
    static constexpr Field Unpack_if_sel_cntx2 {
        RegisterFile::State, 32, 64, 9, 6, 1, 2, 1536}; // Unpacker 0 interface selection. 0: src, 1: dst for context 2 (1b)
    static constexpr Field Unpack_if_sel_cntx3 {
        RegisterFile::State, 32, 64, 9, 7, 1, 2, 1536}; // Unpacker 0 interface selection. 0: src, 1: dst for context 3 (1b)
    static constexpr Field Force_shared_exp {
        RegisterFile::State, 32, 64, 9, 8, 1, 2, 1536}; // Disable shared exponent fetch for BFP formats, If bit is set unpacker will disable shared exp read
                                                        // from L1 and use programmed value instead (1b)
    static constexpr Field Context_count_non_log2 {
        RegisterFile::State, 32, 64, 9, 9, 3, 2, 1536}; // Non-log2 version of Unpacker 0/1 number of available contexts when auto increment context mode is
                                                        // used in unpack instruction. Counter will wrap around when value is equal to Contex_count_non_log2
                                                        // (3b)
    static constexpr Field Context_count_non_log2_en {
        RegisterFile::State, 32, 64, 9, 12, 1, 2, 1536}; // REG2_Context_Count_non_log2 is used instead of REG2_Context_count (1b)
    static constexpr Field Disable_zero_compress_cntx4 {
        RegisterFile::State, 32, 64, 9, 16, 1, 2, 1536}; // Unpacker 0 disable zero compression for context 4 (1b)
    static constexpr Field Disable_zero_compress_cntx5 {
        RegisterFile::State, 32, 64, 9, 17, 1, 2, 1536}; // Unpacker 0 disable zero compression for context 5 (1b)
    static constexpr Field Disable_zero_compress_cntx6 {
        RegisterFile::State, 32, 64, 9, 18, 1, 2, 1536}; // Unpacker 0 disable zero compression for context 6 (1b)
    static constexpr Field Disable_zero_compress_cntx7 {
        RegisterFile::State, 32, 64, 9, 19, 1, 2, 1536}; // Unpacker 0 disable zero compression for context 7 (1b)
    static constexpr Field Unpack_if_sel_cntx4 {
        RegisterFile::State, 32, 64, 9, 20, 1, 2, 1536}; // Unpacker 0 interface selection. 0: src, 1: dst for context 4 (1b)
    static constexpr Field Unpack_if_sel_cntx5 {
        RegisterFile::State, 32, 64, 9, 21, 1, 2, 1536}; // Unpacker 0 interface selection. 0: src, 1: dst for context 5 (1b)
    static constexpr Field Unpack_if_sel_cntx6 {
        RegisterFile::State, 32, 64, 9, 22, 1, 2, 1536}; // Unpacker 0 interface selection. 0: src, 1: dst for context 6 (1b)
    static constexpr Field Unpack_if_sel_cntx7 {
        RegisterFile::State, 32, 64, 9, 23, 1, 2, 1536};                                    // Unpacker 0 interface selection. 0: src, 1: dst for context 7 (1b)
    static constexpr Field Metadata_x_end {RegisterFile::State, 32, 64, 9, 24, 8, 2, 1536}; // 8-bit x_end for Srcb Metadata Unpacking (8b)
    static constexpr Field Unpack_limit_address {
        RegisterFile::State, 32, 64, 10, 0, 17, 2, 1536}; // Unpacker 0/1 source/tile limit address used for wraparound. Wraparound address is computed based on
                                                          // the fifo size below. (17b)
    static constexpr Field Unpack_fifo_size {RegisterFile::State, 32, 64, 11, 0, 17, 2, 1536}; // Unpacker 0/1 source/tile fifo size (17b)
};

struct ThconReg3Fields
{
    static constexpr Field Base_address {
        RegisterFile::State, 32, 64, 12, 0, 32, 2, 1536}; // Unpacker 0/1 Source/Tile base/context 0 address for unpacker (aligned to 16B word) (32b)
    static constexpr Field Base_cntx1_address {
        RegisterFile::State, 32, 64, 13, 0, 32, 2, 1536}; // Unpacker 0/1 source/Tile context 1 address (aligned to 16B word) (32b)
    static constexpr Field Base_cntx2_address {
        RegisterFile::State, 32, 64, 14, 0, 32, 2, 1536}; // Unpacker 0 source/Tile context 2 address (aligned to 16B word) (32b)
    static constexpr Field Base_cntx3_address {
        RegisterFile::State, 32, 64, 15, 0, 32, 2, 1536}; // Unpacker 0 source/Tile context 3 address (aligned to 16B word) (32b)
};

struct ThconReg4Fields
{
    static constexpr Field Base_cntx4_address {
        RegisterFile::State, 32, 64, 16, 0, 32, 2, 1536}; // Unpacker 0 source/Tile context 4 address (aligned to 16B word) (32b)
    static constexpr Field Base_cntx5_address {
        RegisterFile::State, 32, 64, 17, 0, 32, 2, 1536}; // Unpacker 0 source/Tile context 5 address (aligned to 16B word) (32b)
    static constexpr Field Base_cntx6_address {
        RegisterFile::State, 32, 64, 18, 0, 32, 2, 1536}; // Unpacker 0 source/Tile context 6 address (aligned to 16B word) (32b)
    static constexpr Field Base_cntx7_address {
        RegisterFile::State, 32, 64, 19, 0, 32, 2, 1536}; // Unpacker 0 source/Tile context 7 address (aligned to 16B word) (32b)
};

struct ThconReg5Fields
{
    static constexpr Field Dest_cntx0_address {RegisterFile::State, 32, 64, 20, 0, 16, 2, 1536};  // Unpacker 0 dest srca context 0 address (16b)
    static constexpr Field Dest_cntx1_address {RegisterFile::State, 32, 64, 20, 16, 16, 2, 1536}; // Unpacker 0 dest srca context 1 address (16b)
    static constexpr Field Dest_cntx2_address {RegisterFile::State, 32, 64, 21, 0, 16, 2, 1536};  // Unpacker 0 dest srca context 2 address (16b)
    static constexpr Field Dest_cntx3_address {RegisterFile::State, 32, 64, 21, 16, 16, 2, 1536}; // Unpacker 0 dest srca context 3 address (16b)
    static constexpr Field Tile_x_dim_cntx0 {RegisterFile::State, 32, 64, 22, 0, 16, 2, 1536};    // Unpacker 0 tile context 0 x dim (16b)
    static constexpr Field Tile_x_dim_cntx1 {RegisterFile::State, 32, 64, 22, 16, 16, 2, 1536};   // Unpacker 0 tile context 1 x dim (16b)
    static constexpr Field Tile_x_dim_cntx2 {RegisterFile::State, 32, 64, 23, 0, 16, 2, 1536};    // Unpacker 0 tile context 2 x dim (16b)
    static constexpr Field Tile_x_dim_cntx3 {RegisterFile::State, 32, 64, 23, 16, 16, 2, 1536};   // Unpacker 0 tile context 3 x dim (16b)
};

struct ThconReg6Fields
{
    static constexpr Field Source_address {RegisterFile::State, 32, 64, 24, 0, 32, 2, 1536};      // Mover 0 Source address, aligned to 16B (32b)
    static constexpr Field Destination_address {RegisterFile::State, 32, 64, 25, 0, 32, 2, 1536}; // Mover 0 Destination address, aligned to 16B (32b)
    static constexpr Field Buffer_size {RegisterFile::State, 32, 64, 26, 0, 30, 2, 1536};         // Mover 0 Buffer size in 16B chunks (max 64k) (30b)
    static constexpr Field Transfer_direction {
        RegisterFile::State, 32, 64, 26, 30, 2, 2, 1536}; // Mover 0 Transfer direction (3: L1->L1, only legal one now) (2b)
    static constexpr Field Metadata_misc {
        RegisterFile::State, 32, 64, 27, 0, 32, 2, 1536}; // {1-bit disable metadata z-start, 1-bit enable tile header offset addition, 16-bit Dst addr, 10-bit
                                                          // tile x_dim, 1-bit disable compression, 1-bit disable metadata reg override} (32b)
};

struct ThconReg7Fields
{
    static constexpr Field Offset_address {RegisterFile::State, 32, 64, 28, 0, 16, 2, 1536};               // Unpacker 0/1 offset context 0 address (16b)
    static constexpr Field Unpack_data_format_cntx0 {RegisterFile::State, 32, 64, 28, 16, 4, 2, 1536};     // Unpacker 0/1 input data format for context 0 (4b)
    static constexpr Field Unpack_out_data_format_cntx0 {RegisterFile::State, 32, 64, 28, 20, 4, 2, 1536}; // Unpacker 0/1 output data format for context 0 (4b)
    static constexpr Field Unpack_data_format_cntx4 {RegisterFile::State, 32, 64, 28, 24, 4, 2, 1536};     // Unpacker 0 input data format for context 4 (4b)
    static constexpr Field Unpack_out_data_format_cntx4 {RegisterFile::State, 32, 64, 28, 28, 4, 2, 1536}; // Unpacker 0 output data format for context 4 (4b)
    static constexpr Field Offset_cntx1_address {RegisterFile::State, 32, 64, 29, 0, 16, 2, 1536};         // Unpacker 0/1 offset context 1 address (16b)
    static constexpr Field Unpack_data_format_cntx1 {RegisterFile::State, 32, 64, 29, 16, 4, 2, 1536};     // Unpacker 0/1 input data format for context 1 (4b)
    static constexpr Field Unpack_out_data_format_cntx1 {RegisterFile::State, 32, 64, 29, 20, 4, 2, 1536}; // Unpacker 0/1 output data format for context 1 (4b)
    static constexpr Field Unpack_data_format_cntx5 {RegisterFile::State, 32, 64, 29, 24, 4, 2, 1536};     // Unpacker 0 input data format for context 5 (4b)
    static constexpr Field Unpack_out_data_format_cntx5 {RegisterFile::State, 32, 64, 29, 28, 4, 2, 1536}; // Unpacker 0 output data format for context 5 (4b)
    static constexpr Field Offset_cntx2_address {RegisterFile::State, 32, 64, 30, 0, 16, 2, 1536};         // Unpacker 0 offset context 2 address (16b)
    static constexpr Field Unpack_data_format_cntx2 {RegisterFile::State, 32, 64, 30, 16, 4, 2, 1536};     // Unpacker 0 input data format for context 2 (4b)
    static constexpr Field Unpack_out_data_format_cntx2 {RegisterFile::State, 32, 64, 30, 20, 4, 2, 1536}; // Unpacker 0 output data format for context 2 (4b)
    static constexpr Field Unpack_data_format_cntx6 {RegisterFile::State, 32, 64, 30, 24, 4, 2, 1536};     // Unpacker 0 input data format for context 6 (4b)
    static constexpr Field Unpack_out_data_format_cntx6 {RegisterFile::State, 32, 64, 30, 28, 4, 2, 1536}; // Unpacker 0 output data format for context 6 (4b)
    static constexpr Field Offset_cntx3_address {RegisterFile::State, 32, 64, 31, 0, 16, 2, 1536};         // Unpacker 0 offset context 3 address (16b)
    static constexpr Field Unpack_data_format_cntx3 {RegisterFile::State, 32, 64, 31, 16, 4, 2, 1536};     // Unpacker 0 input data format for context 3 (4b)
    static constexpr Field Unpack_out_data_format_cntx3 {RegisterFile::State, 32, 64, 31, 20, 4, 2, 1536}; // Unpacker 0 output data format for context 3 (4b)
    static constexpr Field Unpack_data_format_cntx7 {RegisterFile::State, 32, 64, 31, 24, 4, 2, 1536};     // Unpacker 0 input data format for context 7 (4b)
    static constexpr Field Unpack_out_data_format_cntx7 {RegisterFile::State, 32, 64, 31, 28, 4, 2, 1536}; // Unpacker 0 output data format for context 7 (4b)
};

struct ThconReg8Fields
{
    static constexpr Field Row_start_section_size {
        RegisterFile::State, 32, 64, 32, 0, 16, 2, 1536}; // Packer row-start section size (aligned to 16B word) - regs sets 1/3 (16b)
    static constexpr Field Exp_section_size {
        RegisterFile::State, 32, 64, 32, 16, 16, 2, 1536}; // Packer exp section size (aligned to 16B word) - regs sets 1/3 (16b)
    static constexpr Field L1_Dest_addr {
        RegisterFile::State, 32, 64, 33, 0, 32, 2, 1536}; // Packer destination address for packer (aligned to 16B - regs sets 1/3 word) (32b)
    static constexpr Field Disable_zero_compress {RegisterFile::State, 32, 64, 34, 0, 1, 2, 1536}; // Packer disable zero compression - regs sets 1/3 (1b)
    static constexpr Field Add_l1_dest_addr_offset {
        RegisterFile::State, 32, 64, 34, 1, 1, 2, 1536}; // Packer add accumulated tile size to L1 destination - regs sets 1/3 address (1b)
    static constexpr Field Disable_pack_zero_flags {
        RegisterFile::State, 32, 64, 34, 2, 1, 2, 1536};                                     // Disable setting zero flags in tile header during packing (1b)
    static constexpr Field Unused1 {RegisterFile::State, 32, 64, 34, 3, 1, 2, 1536};         // Unused (1b)
    static constexpr Field Out_data_format {RegisterFile::State, 32, 64, 34, 4, 4, 2, 1536}; // Packer output data format - regs sets 1/3 (4b)
    static constexpr Field In_data_format {RegisterFile::State, 32, 64, 34, 8, 4, 2, 1536};  // Packer input data format - regs sets 1/3 (4b)
    static constexpr Field Dis_shared_exp_assembler {
        RegisterFile::State, 32, 64, 34, 12, 1, 2, 1536}; // Disable packer shared exponent assembler for BFP formats - regs sets 1/3 (1b)
    static constexpr Field Auto_set_last_pacr_intf_sel {
        RegisterFile::State, 32, 64, 34, 13, 1, 2, 1536}; // Auto set PACR[ReadIntfSel] for last PACR per plane using Packer tile counters (1b)
    static constexpr Field Enable_out_fifo {RegisterFile::State, 32, 64, 34, 14, 1, 2, 1536}; // Store tile size and zero mask into FIFO (1b)
    static constexpr Field Sub_l1_tile_header_size {
        RegisterFile::State, 32, 64, 34, 15, 1, 2, 1536}; // Subtract size of the tile header from packer L1 dest address (debug only) by default packer adds
                                                          // 16B to programmed L1 dest address to auto include tile header size (1b)
    static constexpr Field Source_interface_selection {
        RegisterFile::State, 32, 64, 34, 16, 1, 2, 1536}; // Packer source: 0 - Dest register file, 1 - L1;  - regs sets 1/3 (1b)
    static constexpr Field Add_tile_header_size {
        RegisterFile::State, 32, 64, 34, 17, 1, 2, 1536}; // Add size of the tile header to the tile size when tile size accumulation is enabled (1b)
    static constexpr Field pack_dis_y_pos_start_offset {
        RegisterFile::State, 32, 64, 34, 18, 1, 2, 1536}; // Disable y start position offset for edge-masking selection (1b)
    static constexpr Field unpack_tile_offset {
        RegisterFile::State, 32, 64, 34, 19, 5, 2, 1536}; // Offset to add to L1 address for next tile after unpacking current tile (5b)
    static constexpr Field L1_source_addr {
        RegisterFile::State, 32, 64, 34, 24, 8, 2, 1536}; // Packer L1 source address bit 20:13 when source is L1 - regs sets 1/3 (8b)
    static constexpr Field Downsample_mask {RegisterFile::State, 32, 64, 35, 0, 16, 2, 1536}; // Packer down-sample mask (0x555, 0x1111, 0x0101, 0x0001, etc.,
                                                                                              // as down-sample rate requires  - regs sets 1/3 (16b)
    static constexpr Field Downsample_rate {RegisterFile::State, 32, 64, 35, 16, 3, 2, 1536}; // Packer down-sample rate (0,1,2) - regs sets 1/3 (3b)
    static constexpr Field Pack_L1_Acc {
        RegisterFile::State, 32, 64, 35, 19, 1, 2, 1536}; // Send Acc command to L1 from Packer to perform Accumulation - regs sets 1/3 (1b)
    static constexpr Field Exp_threshold_en {
        RegisterFile::State, 32, 64, 35, 20, 1, 2, 1536}; // Packer force exp to zero if it's below the threshold specified  - regs sets 1/3 (1b)
    static constexpr Field Exp_threshold {RegisterFile::State, 32, 64, 35, 24, 8, 2, 1536}; // Packer exp threshold - regs sets 1/3 (8b)
};

struct ThconReg9Fields
{
    static constexpr Field Pack_0_2_limit_address {
        RegisterFile::State, 32, 64, 36, 0, 17, 2, 1536}; // Packer dest limit address used for wraparound. Wraparound address is computed based on the fifo
                                                          // size below - regs sets 0/2 (17b)
    static constexpr Field Pack_0_2_fifo_size {RegisterFile::State, 32, 64, 37, 0, 17, 2, 1536}; // Packer dest fifo size - regs sets 0/2 (17b)
    static constexpr Field Pack_1_3_limit_address {
        RegisterFile::State, 32, 64, 38, 0, 17, 2, 1536}; // Packer dest limit address used for wraparound. Wraparound address is computed based on the fifo
                                                          // size below - regs sets 1/3 (17b)
    static constexpr Field Pack_1_3_fifo_size {RegisterFile::State, 32, 64, 39, 0, 17, 2, 1536}; // Packer dest fifo size - regs sets 1/3 (17b)
};

struct ThconReg10Fields
{
    static constexpr Field Unpack_limit_address {
        RegisterFile::State, 32, 64, 40, 0, 17, 2, 1536}; // Unpacker source/tile extra limit address used by cntx1 for wraparound ;  used when
                                                          // REG10_Unpack_limit_address_en is set. Wraparound address is computed based on the fifo size below.
                                                          // (17b)
    static constexpr Field Unpack_fifo_size {RegisterFile::State, 32, 64, 41, 0, 17, 2, 1536}; // Unpacker source/tile extra fifo size used by ctnx1;  used
                                                                                               // along with REG10_Unpack_limit_address (17b)
    static constexpr Field Unpack_limit_address_en {RegisterFile::State, 32, 64, 41, 17, 1, 2, 1536}; // Unpacker enable extra limit address wraparound (1b)
    static constexpr Field Unpacker_Reg_Wr_Addr {RegisterFile::State, 32, 64, 42, 0, 24, 2, 1536}; // 4 sets of 12-bit addresses including both Sections (24b)
    static constexpr Field Packer_Reg_Wr_Addr {RegisterFile::State, 32, 64, 43, 0, 24, 2, 1536};   // 4 sets of 12-bit addresses including both Sections (24b)
};

struct ThconReg11Fields
{
    static constexpr Field Metadata_l1_addr {RegisterFile::State, 32, 64, 44, 0, 32, 2, 1536};    // Metadata Tile L1 addr (32b)
    static constexpr Field Metadata_limit_addr {RegisterFile::State, 32, 64, 45, 0, 32, 2, 1536}; // Metadata Limit addr (32b)
    static constexpr Field Metadata_fifo_size {RegisterFile::State, 32, 64, 46, 0, 32, 2, 1536};  // Metadata FIFO size (32b)
    static constexpr Field Metadata_z_cntr_rst_unpacr_count {
        RegisterFile::State, 32, 64, 47, 0, 8, 2, 1536}; // Metadata UNPACR count before resetting z-counter (8b)
    static constexpr Field Metadata_cntxt_switch_unpacr_count {
        RegisterFile::State, 32, 64, 47, 8, 8, 2, 1536}; // Metadata UNPACR count before switching context (8b)
};

struct ThconFields
{
    constexpr ThconReg0Fields operator[](ThconReg0Selector) const
    {
        return {};
    }

    constexpr ThconReg1Fields operator[](ThconReg1Selector) const
    {
        return {};
    }

    constexpr ThconReg2Fields operator[](ThconReg2Selector) const
    {
        return {};
    }

    constexpr ThconReg3Fields operator[](ThconReg3Selector) const
    {
        return {};
    }

    constexpr ThconReg4Fields operator[](ThconReg4Selector) const
    {
        return {};
    }

    constexpr ThconReg5Fields operator[](ThconReg5Selector) const
    {
        return {};
    }

    constexpr ThconReg6Fields operator[](ThconReg6Selector) const
    {
        return {};
    }

    constexpr ThconReg7Fields operator[](ThconReg7Selector) const
    {
        return {};
    }

    constexpr ThconReg8Fields operator[](ThconReg8Selector) const
    {
        return {};
    }

    constexpr ThconReg9Fields operator[](ThconReg9Selector) const
    {
        return {};
    }

    constexpr ThconReg10Fields operator[](ThconReg10Selector) const
    {
        return {};
    }

    constexpr ThconReg11Fields operator[](ThconReg11Selector) const
    {
        return {};
    }
};

/**
 * @brief Compile-time field access grouped by THCON register number.
 *
 * Select REG0 through REG11 with the typed `Reg0` through `Reg11`
 * selectors. The register prefix is omitted from the selected field name.
 * Each terminal remains a static `Field` and can be used as a `const Field&`
 * non-type template argument. Use `Sec::S0` or `Sec::S1` to select the
 * corresponding THCON section.
 *
 * @code{.cpp}
 * write<Access::TensixCfgUnit, Thcon[Reg2].Out_data_format, Sec::S0>(format);
 * write<Access::TensixCfgUnit, Thcon[Reg7].Offset_address, Sec::S1>(offset);
 * @endcode
 */
inline constexpr ThconFields Thcon {};

} // namespace cfg
} // namespace hal
