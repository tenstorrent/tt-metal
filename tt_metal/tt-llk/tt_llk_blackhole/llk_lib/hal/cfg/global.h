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
// GLOBAL
// ============================================================

class DestTargetRegCfgPack
{ // Set destination register offset for Packer. Address must be aligned to 16B
public:
    static constexpr Field Offset {RegisterFile::State, 32, 180, 0, 0, 12, 4, 32};  // Packer source/target dest register static offset - 4 reg sets (12b)
    static constexpr Field ZOffset {RegisterFile::State, 32, 180, 0, 12, 6, 4, 32}; // Packer Z-tile offset for tile position generator - 4 reg sets (6b)
};

class CgSrcPipeline
{ // Enable clock gating of the srca/srcb pipeline
public:
    static constexpr Field GateSrcAPipeEn {RegisterFile::State, 32, 184, 0, 0, 1, 1, 0}; // Gate SRCA pipeline enable (1b)
    static constexpr Field GateSrcBPipeEn {RegisterFile::State, 32, 184, 0, 1, 1, 1, 0}; // Gate SRCB pipeline enable (1b)
};

class RiscvIcInvalidate
{ // RISCV instruction cache invalidate
public:
    static constexpr Field InvalidateAll {RegisterFile::State, 32, 185, 0, 0, 5, 1, 0}; // Invalidate RISCV instruction cache. Bit 0 - main Risc core, bits 1-3
                                                                                        // - three Trisc threads, 4 - Noc control risc (5b)
};

class PrngSeed
{ // Seed master PRNG and start seeding clients
public:
    static constexpr Field Seed_Val {RegisterFile::State, 32, 186, 0, 0, 32, 1, 0}; // Seed value for master PRNG (32b)
};

class IntDescaleValues
{ // 4 8-bit words for descaling
public:
    static constexpr Field Value {RegisterFile::State, 32, 187, 0, 0, 32, 16, 32}; // Value (32b)
};

class TriscEndPc
{ // Trisc end PC value (ending location of code)
public:
    static constexpr Field PC {RegisterFile::State, 32, 203, 0, 0, 32, 3, 32}; // End PC value (32b)
};

class BriscEndPc
{ // Main risc (brisc) end PC value (ending location of code)
public:
    static constexpr Field PC {RegisterFile::State, 32, 206, 0, 0, 32, 1, 0}; // End PC value (32b)
};

class NocRiscEndPc
{ // NOC risc end PC value (ending location of code)
public:
    static constexpr Field PC {RegisterFile::State, 32, 207, 0, 0, 32, 1, 0}; // End PC value (32b)
};

class RiscPrefetchCtrl
{ // Risc prefetcher enable
public:
    static constexpr Field Enable_Trisc {RegisterFile::State, 32, 208, 0, 0, 3, 1, 0};   // Trisc prefetcher enable (3b)
    static constexpr Field Enable_Brisc {RegisterFile::State, 32, 208, 0, 3, 1, 1, 0};   // Brisc prefetcher enable (1b)
    static constexpr Field Enable_NocRisc {RegisterFile::State, 32, 208, 0, 4, 1, 1, 0}; // NOC risc prefetcher enable (1b)
    static constexpr Field Max_Req_Count {RegisterFile::State, 32, 208, 0, 5, 8, 1, 0};  // Max number of requests in stream prefetcher (8b)
};

class Scratch
{ // Just a register with no architectural effects. Used with CFGSHIFTMASK instruction
public:
    static constexpr Field val {RegisterFile::State, 32, 209, 0, 0, 32, 3, 32}; // Scratch register value. (32b)
};

class L1CacheTagSearchAccel
{ // L1 Cache Tag Search Acceleration control register
public:
    static constexpr Field Search_Enable {
        RegisterFile::State, 32, 212, 0, 0, 1, 1, 0}; // Enable the functionality (for other fields of this register, like addresses, to be captured, this bit
                                                      // has to be toggled - meaning needs to be disabled and then enabled back with new values for the other
                                                      // fields) (1b)
    static constexpr Field Start_Addr {RegisterFile::State, 32, 212, 0, 1, 17, 1, 0};     // Search Start Addr (17b)
    static constexpr Field End_Addr {RegisterFile::State, 32, 212, 1, 0, 17, 1, 0};       // Search End Addr (17b)
    static constexpr Field Tag_Value_low {RegisterFile::State, 32, 212, 2, 0, 32, 1, 0};  // Search Tag Value lower 32 bits (32b)
    static constexpr Field Tag_Value_high {RegisterFile::State, 32, 212, 3, 0, 32, 1, 0}; // Search Tag Value higher 32 bits (32b)
    static constexpr Field Tag_Width {
        RegisterFile::State, 32, 212, 4, 0, 2, 1, 0}; // Encoded Tag Width - 2'b00 : 8-bit, 2'b01 : 16-bit, 2'b10 : 32-bit, 2'b11 : 64-bit (2b)
    static constexpr Field Valid_bit_section_start_addr {RegisterFile::State, 32, 212, 4, 2, 17, 1, 0};      // Valid bit section start addr (17b)
    static constexpr Field Valid_bit_section_end_addr {RegisterFile::State, 32, 212, 5, 0, 17, 1, 0};        // Valid bit section end addr (17b)
    static constexpr Field Data_Valid_bit_section_start_addr {RegisterFile::State, 32, 212, 6, 0, 17, 1, 0}; // Data valid bit section start addr (17b)
    static constexpr Field Data_Valid_chk {RegisterFile::State, 32, 212, 6, 17, 1, 1, 0};                    // Check whether data is valid (1b)
    static constexpr Field Data_Valid_offset {RegisterFile::State, 32, 212, 7, 0, 24, 1, 0}; // Data valid offset for checking whether data is valid (24b)
    static constexpr Field Tag_inv {RegisterFile::State, 32, 212, 7, 24, 1, 1, 0};           // Invalidate Tag (1b)
    static constexpr Field Tag_inv_all {RegisterFile::State, 32, 212, 7, 25, 1, 1, 0};       // Invalidate all tags (1b)
    static constexpr Field Tag_alloc {RegisterFile::State, 32, 212, 7, 26, 1, 1, 0};         // Alloc tag after tag search if not already present (1b)
};

class DestAccessCfg
{ // There are two optional modes you can turn on in dest: - "Stride 8 swizzling in 32-bit mode" (call this A) - "Address remapping" (call this B)
public:
    static constexpr Field swizzle_32b {
        RegisterFile::State,
        32,
        220,
        0,
        0,
        1,
        1,
        0}; // Allow 32-bit datums to be swizzled across horizontal banks. See
            // https://tenstorrent.sharepoint.com/:x:/s/Specifications/EYYARiPa5RFHp7lJ8QGNdIgBj0TjQfxWP0M6Lr7T--maSw?e=c4Less
            // for details (1b)
    static constexpr Field remap_addrs {
        RegisterFile::State,
        32,
        220,
        0,
        1,
        1,
        1,
        0}; // Allow hardware to remap dest addresses behind-the-scenes. See
            // https://tenstorrent.sharepoint.com/:x:/s/Specifications/Eenm7aQbWf5FoIM6jYQoavwB5plRcQZo0mw_GcRHvklcSw?e=c2rSfC
            // for details (1b)
    static constexpr Field disable_full_write_dest_q_bypass {
        RegisterFile::State, 32, 220, 0, 2, 1, 1, 0}; // As a power optimization, if a dest write would write an entire SRAM row, it will bypass the dest_q.
                                                      // This is a chicken bit to disable that optimization in case it causes problems. This bit does nothing if
                                                      // the RTL is using the flop array for dest. (1b)
    static constexpr Field zeroacc_absolute_tile_mode {
        RegisterFile::State, 32, 220, 0, 3, 1, 1, 0}; // A thread can program its DEST_REGW_BASE register to select the half of dest it's planning to write in.
                                                      // As of Nov 18 2022, the ZEROACC CLR_16 mode (also called "tile mode") will be relative to the half of
                                                      // dest as selected by this register. However, if you want the old behaviour where the tile index is
                                                      // absolute, set this bit to 1. (1b)
};

class SrcAccessCfg
{ // Mode bits for accessing source registers
public:
    static constexpr Field math_view_srca_as_one_bank {
        RegisterFile::State,
        32,
        221,
        0,
        0,
        1,
        1,
        0}; // All math instructions will operate as if srcA is a single bank of 128 rows (instead of the default two banks of 64 rows each). Math instructions
            // will automatically wait for valid data.  Because the unpacker will still treat srcA as two banks, even if this mode is on, it is possible for a
            // math instruction to start when only half of srcA has been filled (provided that the rows it is accessing have already been filled by the
            // unpacker). However, when this mode is turned on, math instructions that  clear dvalid, will clear BOTH banks instead of just one. (1b)
    static constexpr Field math_view_srcb_as_one_bank {RegisterFile::State, 32, 221, 0, 1, 1, 1, 0}; // Same as math_view_srca_as_one_bank but for srcB (1b)
    static constexpr Field disable_contig_srca_dvalid_phase {
        RegisterFile::State, 32, 221, 0, 2, 1, 1, 0}; // TLDR: Don't set this bit to 1, it's not what you want. (1b)
    static constexpr Field disable_contig_srcb_dvalid_phase {
        RegisterFile::State, 32, 221, 0, 3, 1, 1, 0}; // Same as disable_contig_srca_dvalid_phase but for srcB (1b)
};

class ChickenBits
{ // A place to add chicken bits
public:
    static constexpr Field sfpu_scbd_disable {
        RegisterFile::State, 32, 222, 0, 0, 1, 1, 0}; // Disable any stalls from the SFPU scoreboarding logic. Not to be confused with the dest arbitration;
                                                      // this scoreboarding is the  one that lets you omit SFPNOPs between instructions that would otherwise
                                                      // have a data/control hazard. (1b)
};

} // namespace cfg
} // namespace hal
