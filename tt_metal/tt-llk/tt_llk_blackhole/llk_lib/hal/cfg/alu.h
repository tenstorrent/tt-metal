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
// ALU
// ============================================================

struct AluFormatSpecReg
{                                                                                        // Alu format spec override register
    static constexpr Field SrcA_val {RegisterFile::State, 32, 0, 0, 0, 4, 1, 0};         // override spec for format spec of srcA (4b)
    static constexpr Field SrcA_override {RegisterFile::State, 32, 0, 0, 4, 1, 1, 0};    // override bit enable bit (1b)
    static constexpr Field SrcB_val {RegisterFile::State, 32, 0, 0, 5, 4, 1, 0};         // override for format spec of srcB (4b)
    static constexpr Field SrcB_override {RegisterFile::State, 32, 0, 0, 9, 1, 1, 0};    // override bit enable bit (1b)
    static constexpr Field Dstacc_val {RegisterFile::State, 32, 0, 0, 10, 4, 1, 0};      // override for format spec of Dest (4b)
    static constexpr Field Dstacc_override {RegisterFile::State, 32, 0, 0, 14, 1, 1, 0}; // override enable bit (1b)
};

struct AluRoundingMode
{                                                                                      // Stochastic Rounding control
    static constexpr Field Fpu_srnd_en {RegisterFile::State, 32, 1, 0, 0, 1, 1, 0};    // Enable stochastic rounding in Fpu (1b)
    static constexpr Field Gasket_srnd_en {RegisterFile::State, 32, 1, 0, 1, 1, 1, 0}; // Enable stochastic rounding in packer gasket (1b)
    static constexpr Field Packer_srnd_en {RegisterFile::State, 32, 1, 0, 2, 1, 1, 0}; // Enable stochastic rounding in packer during BFP conversion (1b)
    static constexpr Field Padding {RegisterFile::State, 32, 1, 0, 3, 10, 1, 0};       // padding (10b)
    static constexpr Field GS_LF {RegisterFile::State, 32, 1, 0, 13, 1, 1, 0};         // Set operand widths to match GS LF (1.4 for src a and srcb) (1b)
    static constexpr Field Bfp8_HF {
        RegisterFile::State, 32, 1, 0, 14, 1, 1, 0}; // Compute BFP8 operands in high-fidelity in single phase (must chose LF kernel) (1b)
};

struct AluFormatSpecReg0
{                                                                                     // SrcA spec register - matches unpacker1 dst format
    static constexpr Field SrcAUnsigned {RegisterFile::State, 32, 1, 0, 15, 1, 1, 0}; // srcA unsigned mode (valid only for int8 format) When set unpacker
                                                                                      // output will be treated as unsigned int8 format (1b)
    static constexpr Field SrcBUnsigned {RegisterFile::State, 32, 1, 0, 16, 1, 1, 0}; // srcB unsigned mode (valid only for int8 format) When set unpacker
                                                                                      // output will be treated as unsigned int8 format (1b)
    static constexpr Field SrcA {RegisterFile::State, 32, 1, 0, 17, 4, 1, 0};         // srcA format spec (4b)
};

struct AluFormatSpecReg1
{                                                                             // SrcB spec register - matches unpacker1 dst format
    static constexpr Field SrcB {RegisterFile::State, 32, 1, 0, 21, 4, 1, 0}; // srcB format spec (4b)
};

struct AluFormatSpecReg2
{                                                                               // Dest spec register - matches packer src format
    static constexpr Field Dstacc {RegisterFile::State, 32, 1, 0, 25, 4, 1, 0}; // Dest format spec (4b)
};

struct AluAccCtrl
{                                                                                              // Control the math accumulation format
    static constexpr Field Fp32_enabled {RegisterFile::State, 32, 1, 0, 29, 1, 1, 0};          // Enable fp32 accumulation (1b)
    static constexpr Field SFPU_Fp32_enabled {RegisterFile::State, 32, 1, 0, 30, 1, 1, 0};     // Enable fp32 in SFPU (1b)
    static constexpr Field INT8_math_enabled {RegisterFile::State, 32, 1, 0, 31, 1, 1, 0};     // Run Math with int8 mode (1b)
    static constexpr Field Zero_Flag_disabled_src {RegisterFile::State, 32, 1, 1, 0, 1, 1, 0}; // Disables zero flag detection in srcA and srcB (1b)
    static constexpr Field Zero_Flag_disabled_dst {RegisterFile::State, 32, 1, 1, 1, 1, 1, 0}; // Disables zero flag reads in DEST (1b)
};

struct StaccRelu
{ // Apply RELU as part of STACC
    static constexpr Field ApplyRelu {
        RegisterFile::State, 32, 2, 0, 2, 4, 1, 0}; // If set to 1, apply ReLU on the word being written to DataRAM (2=Threshold Relu, 3=MaxRelu) (4b)
    static constexpr Field ReluThreshold {RegisterFile::State, 32, 2, 0, 6, 16, 1, 0}; // Threshold to use for Relu (16b)
};

struct DisableRiscBp
{                                                                                                 // Brisc BP control
    static constexpr Field Disable_main {RegisterFile::State, 32, 2, 0, 22, 1, 1, 0};             // Disable branch prediction (Brisc) (1b)
    static constexpr Field Disable_trisc {RegisterFile::State, 32, 2, 0, 23, 3, 1, 0};            // Disable branch prediction (Trisc0-2) (3b)
    static constexpr Field Disable_ncrisc {RegisterFile::State, 32, 2, 0, 26, 1, 1, 0};           // Disable branch prediction (NcRisc) (1b)
    static constexpr Field Disable_bmp_clear_main {RegisterFile::State, 32, 2, 0, 27, 1, 1, 0};   // Disable branch misprediction clearing (Brisc) (1b)
    static constexpr Field Disable_bmp_clear_trisc {RegisterFile::State, 32, 2, 0, 28, 3, 1, 0};  // Disable branch misprediction clearing (Trisc0-2) (3b)
    static constexpr Field Disable_bmp_clear_ncrisc {RegisterFile::State, 32, 2, 0, 31, 1, 1, 0}; // Disable branch misprediction clearing (NcRisc) (1b)
};

struct EccScrubber
{                                                                                                  // ECC scrubber control
    static constexpr Field Enable {RegisterFile::State, 32, 3, 0, 0, 1, 1, 0};                     // Enable L1 ECC scrubber (1b)
    static constexpr Field Scrub_On_Error {RegisterFile::State, 32, 3, 0, 1, 1, 1, 0};             // Send the next scrub to location of last SBE (1b)
    static constexpr Field Scrub_On_Error_Immediately {RegisterFile::State, 32, 3, 0, 2, 1, 1, 0}; // Send an immediate scrub to location of last SBE (1b)
    static constexpr Field Delay {RegisterFile::State, 32, 3, 0, 3, 11, 1, 0}; // Number of 16-bit counter ticks between scrub requests (11b)
};

struct RiscDestAccessCtrl
{ // TRISC DEST Access Control - section0 for TRISC0, section1 for TRISC1 and section2 for TRISC2
    static constexpr Field no_swizzle {RegisterFile::State, 32, 3, 0, 14, 1, 3, 5};   // No swizzling of bits or saturation detection (1b)
    static constexpr Field unsigned_int {RegisterFile::State, 32, 3, 0, 15, 1, 3, 5}; // Unsigned INT format (1b)
    static constexpr Field fmt {
        RegisterFile::State, 32, 3, 0, 16, 3, 3, 5}; // Dest format -> 000=FP32, 001=INT32, 010=FP16A, 011=FP16B, 100=INT16, 101=INT8 (3b)
};

struct StateReset
{                                                                          // Reset cfg state bank
    static constexpr Field EN {RegisterFile::State, 32, 4, 0, 0, 1, 1, 0}; // Enable reset (1b)
};

struct DestOffset
{                                                                              // Apply dest register offset settings
    static constexpr Field Enable {RegisterFile::State, 32, 5, 0, 0, 1, 1, 0}; // Enable dest offset addition (1b)
};

struct DestRegwBase
{                                                                             // reg window counter base register
    static constexpr Field Base {RegisterFile::State, 32, 6, 0, 0, 16, 1, 0}; // Base for window counter (16b)
};

struct DestSpBase
{                                                                             // SP base register
    static constexpr Field Base {RegisterFile::State, 32, 7, 0, 0, 16, 1, 0}; // Base for window counter (16b)
};

struct IntDescale
{                                                                              // Apply integer de-scaling while packing
    static constexpr Field Enable {RegisterFile::State, 32, 8, 0, 0, 1, 1, 0}; // Enable fast integer de-scaling (1b)
    static constexpr Field Mode {RegisterFile::State, 32, 8, 0, 1, 1, 1, 0}; // de-scaling mode (0 - single value per tensor; 1 - value per output filter) (1b)
};

} // namespace cfg
} // namespace hal
