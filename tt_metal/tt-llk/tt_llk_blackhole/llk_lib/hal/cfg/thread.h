// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// offsets source: tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h
#pragma once

#include <cstdint>

#include "field.h"

namespace hal
{
namespace cfg
{
// ============================================================
// THREAD
// ============================================================

class CfgStateId
{ // Cfg state id for this thread
public:
    static constexpr Field StateID {RegisterScope::Thread, 16, 0, 0, 0, 1, 1, 0}; // Configuration state context to use for this thread (1b)
};

class DestTargetRegCfgMath
{ // Set destination register offset for math and packer
public:
    static constexpr Field Offset {RegisterScope::Thread, 16, 1, 0, 0, 12, 1, 0}; // Math source/target dest register static offset (12b)
};

class SrcASelector
{
};

class SrcBSelector
{
};

class SrcSelector
{
};

class DestSelector
{
};

class FidelitySelector
{
};

class BiasSelector
{
};

class YSelector
{
};

class ZSelector
{
};

inline constexpr SrcASelector SrcA {};
inline constexpr SrcBSelector SrcB {};
inline constexpr SrcSelector Src {};
inline constexpr DestSelector Dest {};
inline constexpr FidelitySelector Fidelity {};
inline constexpr BiasSelector Bias {};
inline constexpr YSelector Y {};
inline constexpr ZSelector Z {};

class DisableImpliedFmtFields
{
private:
    static constexpr Field SrcAField {RegisterScope::Thread, 16, 2, 0, 0, 1, 1, 0}; // Disable implied Unp0-SrcA Fmt (1b)
    static constexpr Field SrcBField {RegisterScope::Thread, 16, 3, 0, 0, 1, 1, 0}; // Disable implied Unp1-SrcB Fmt (1b)

public:
    constexpr const Field& operator[](SrcASelector) const
    {
        return SrcAField;
    }

    constexpr const Field& operator[](SrcBSelector) const
    {
        return SrcBField;
    }
};

// Disable Implied format for Src Registers
inline constexpr DisableImpliedFmtFields DisableImpliedFmt {};

class SfpuDestFmt
{ // Format SFPU expects in Dest (used to determine 8-bit/5-bit exponent formats)
public:
    static constexpr Field Enable {RegisterScope::Thread, 16, 4, 0, 0, 1, 1, 0}; // Enable SFPU format from thread register (1b)
    static constexpr Field Base {RegisterScope::Thread, 16, 4, 0, 1, 4, 1, 0};   // Format SFPU expects in Dest (4b)
};

class SrcASet
{ // SrcA Base Set
public:
    static constexpr Field Base {RegisterScope::Thread, 16, 5, 0, 0, 2, 1, 0};            // SrcA Base Set (2b)
    static constexpr Field SetOvrdWithAddr {RegisterScope::Thread, 16, 5, 0, 2, 1, 1, 0}; // Ovrd set index with higher wr addr bits (1b)
};

class SrcBSet
{ // SrcB Base Set
public:
    static constexpr Field Base {RegisterScope::Thread, 16, 6, 0, 0, 2, 1, 0}; // SrcB Base Set (2b)
};

class ClrDvalid
{ // Disable data valid clear unless its CLEARDVALID inst. Banks will still switch (useful when using both banks)
public:
    static constexpr Field SrcA_Disable {RegisterScope::Thread, 16, 7, 0, 0, 1, 1, 0}; // SrcA Data valid clear disable (1b)
    static constexpr Field SrcB_Disable {RegisterScope::Thread, 16, 7, 0, 1, 1, 1, 0}; // SrcB Data valid clear disable (1b)
};

class ScbdBankMask32b
{ // Generate bank masks for screboard assuming double buffering at 32-bit datum gran.
public:
    static constexpr Field Enable {RegisterScope::Thread, 16, 8, 0, 0, 1, 1, 0}; // Enable (1b)
};

class PackScbdBankMask32b
{ // Generate bank masks for screboard assuming double buffering at 32-bit datum gran. (pack)
public:
    static constexpr Field Enable {RegisterScope::Thread, 16, 9, 0, 0, 1, 1, 0}; // Enable (1b)
};

class UnpackScbdBankMask32b
{ // Generate bank masks for screboard assuming double buffering at 32-bit datum gran. (unpack)
public:
    static constexpr Field Enable {RegisterScope::Thread, 16, 10, 0, 0, 1, 1, 0}; // Enable (1b)
};

class FidelityBase
{ // Base fidelity phase
public:
    static constexpr Field Phase {RegisterScope::Thread, 16, 11, 0, 0, 2, 1, 0}; // Base fidelity phase (2b)
};

class AddrModSrcEntry
{
public:
    const Field& Incr;
    const Field& Incr2;
    const Field& CR;
    const Field& Clear;
};

class AddrModPackYEntry
{
public:
    const Field& Incr;
    const Field& CR;
    const Field& Clear;
};

class AddrModPackZEntry
{
public:
    const Field& Incr;
    const Field& Clear;
};

class AddrModPackEntry
{
public:
    AddrModPackYEntry YFields;
    AddrModPackZEntry ZFields;

    constexpr AddrModPackYEntry operator[](YSelector) const
    {
        return YFields;
    }

    constexpr AddrModPackZEntry operator[](ZSelector) const
    {
        return ZFields;
    }
};

class AddrModDestEntry
{
public:
    const Field& Incr;
    const Field& CR;
    const Field& Clear;
    const Field& CToCR;

    constexpr AddrModDestEntry(const Field& incr, const Field& cr, const Field& clear, const Field& c_to_cr, AddrModPackEntry pack) :
        Incr(incr), CR(cr), Clear(clear), CToCR(c_to_cr), Pack(pack)
    {
    }

    constexpr AddrModPackYEntry operator[](YSelector y) const
    {
        return Pack[y];
    }

    constexpr AddrModPackZEntry operator[](ZSelector z) const
    {
        return Pack[z];
    }

private:
    AddrModPackEntry Pack;
};

class AddrModFidelityEntry
{
public:
    const Field& Incr;
    const Field& Clear;
};

class AddrModBiasEntry
{
public:
    const Field& Incr;
    const Field& Clear;
};

class AddrModFields
{
private:
    template <bool IsSrcB>
    class SrcFields
    {
    public:
        static constexpr std::uint32_t Shift = IsSrcB ? 8 : 0;

        static constexpr Field Incr {RegisterScope::Thread, 16, 12, 0, Shift, 6, 8, 16};             // Src A/B autoincrement amount (6b)
        static constexpr Field Incr2 {RegisterScope::Thread, 16, 20, 0, IsSrcB ? 1u : 0u, 1, 8, 16}; // Bit 6 of Src A/B autoincrement amount (1b)
        static constexpr Field CR {RegisterScope::Thread, 16, 12, 0, Shift + 6, 1, 8, 16};           // Src A/B CR (1b)
        static constexpr Field Clear {RegisterScope::Thread, 16, 12, 0, Shift + 7, 1, 8, 16};        // Src A/B Clear (1b)
    };

    template <bool IsSrcB>
    static constexpr AddrModSrcEntry make_src()
    {
        return {SrcFields<IsSrcB>::Incr, SrcFields<IsSrcB>::Incr2, SrcFields<IsSrcB>::CR, SrcFields<IsSrcB>::Clear};
    }

    static constexpr Field DestIncr {RegisterScope::Thread, 16, 28, 0, 0, 10, 8, 16};  // Dest autoincrement amount (10b)
    static constexpr Field DestCR {RegisterScope::Thread, 16, 28, 0, 10, 1, 8, 16};    // Dest CR (1b)
    static constexpr Field DestClear {RegisterScope::Thread, 16, 28, 0, 11, 1, 8, 16}; // Dest Clear (1b)
    static constexpr Field DestCToCR {RegisterScope::Thread, 16, 28, 0, 12, 1, 8, 16}; // Copy dest counter to CR (post-increment) (1b)

    static constexpr Field FidelityIncr {RegisterScope::Thread, 16, 28, 0, 13, 2, 8, 16};  // Fidelity autoincrement amount (2b)
    static constexpr Field FidelityClear {RegisterScope::Thread, 16, 28, 0, 15, 1, 8, 16}; // Fidelity Clear (1b)

    static constexpr Field BiasIncr {RegisterScope::Thread, 16, 47, 0, 0, 4, 8, 16};  // Bias autoincrement amount (4b)
    static constexpr Field BiasClear {RegisterScope::Thread, 16, 47, 0, 4, 1, 8, 16}; // Dest Clear (1b)

    static constexpr Field YSrcIncr {RegisterScope::Thread, 16, 37, 0, 0, 4, 4, 16};   // Y dim src (regfile) autoincrement amount (4b)
    static constexpr Field YSrcCR {RegisterScope::Thread, 16, 37, 0, 4, 1, 4, 16};     // Y dim src (regfile) CR (1b)
    static constexpr Field YSrcClear {RegisterScope::Thread, 16, 37, 0, 5, 1, 4, 16};  // Y dim src (regfile) Clear (1b)
    static constexpr Field YDstIncr {RegisterScope::Thread, 16, 37, 0, 6, 4, 4, 16};   // Y dim dst (l1) autoincrement amount (4b)
    static constexpr Field YDstCR {RegisterScope::Thread, 16, 37, 0, 10, 1, 4, 16};    // Y dim dst (l1) CR (1b)
    static constexpr Field YDstClear {RegisterScope::Thread, 16, 37, 0, 11, 1, 4, 16}; // Y dim dst (l1) Clear (1b)
    static constexpr Field ZSrcIncr {RegisterScope::Thread, 16, 37, 0, 12, 1, 4, 16};  // Z dim src (regfile) autoincrement amount (1b)
    static constexpr Field ZSrcClear {RegisterScope::Thread, 16, 37, 0, 13, 1, 4, 16}; // Z dim src (regfile) Clear (1b)
    static constexpr Field ZDstIncr {RegisterScope::Thread, 16, 37, 0, 14, 1, 4, 16};  // Z dim dst (l1) autoincrement amount (1b)
    static constexpr Field ZDstClear {RegisterScope::Thread, 16, 37, 0, 15, 1, 4, 16}; // Z dim dst (l1) Clear (1b)

    static constexpr AddrModPackEntry src_pack()
    {
        return {{YSrcIncr, YSrcCR, YSrcClear}, {ZSrcIncr, ZSrcClear}};
    }

    static constexpr AddrModPackEntry dest_pack()
    {
        return {{YDstIncr, YDstCR, YDstClear}, {ZDstIncr, ZDstClear}};
    }

public:
    constexpr AddrModSrcEntry operator[](SrcASelector) const
    {
        return make_src<false>();
    }

    constexpr AddrModSrcEntry operator[](SrcBSelector) const
    {
        return make_src<true>();
    }

    constexpr AddrModPackEntry operator[](SrcSelector) const
    {
        return src_pack();
    }

    constexpr AddrModDestEntry operator[](DestSelector) const
    {
        return {DestIncr, DestCR, DestClear, DestCToCR, dest_pack()};
    }

    constexpr AddrModFidelityEntry operator[](FidelitySelector) const
    {
        return {FidelityIncr, FidelityClear};
    }

    constexpr AddrModBiasEntry operator[](BiasSelector) const
    {
        return {BiasIncr, BiasClear};
    }
};

/**
 * @brief Field access for source, destination, fidelity, bias, and pack address modifiers.
 *
 * Select a modifier target with the first `operator[]`. Pack address
 * modifiers take a second selector for the Y or Z dimension. Each expression
 * returns a reference to a complete, static `Field`, so it can be passed as a
 * `const Field&` non-type template argument to `set()` or `write()`.
 *
 * Source A and B each have eight sections:
 * - `AddrMod[SrcA].Incr`: SrcA autoincrement bits 5:0 (6 bits).
 * - `AddrMod[SrcA].Incr2`: bit 6 of the SrcA autoincrement amount (1 bit).
 * - `AddrMod[SrcA].CR`: SrcA CR (1 bit).
 * - `AddrMod[SrcA].Clear`: clear the SrcA counter (1 bit).
 * - `AddrMod[SrcB].Incr`: SrcB autoincrement bits 5:0 (6 bits).
 * - `AddrMod[SrcB].Incr2`: bit 6 of the SrcB autoincrement amount (1 bit).
 * - `AddrMod[SrcB].CR`: SrcB CR (1 bit).
 * - `AddrMod[SrcB].Clear`: clear the SrcB counter (1 bit).
 *
 * Destination and fidelity each have eight sections:
 * - `AddrMod[Dest].Incr`, `.CR`, and `.Clear`: destination autoincrement
 *   (10 bits), CR (1 bit), and clear (1 bit).
 * - `AddrMod[Dest].CToCR`: copy the destination counter to CR after increment
 *   (1 bit).
 * - `AddrMod[Fidelity].Incr` and `.Clear`: fidelity autoincrement (2 bits)
 *   and clear (1 bit).
 * - `AddrMod[Bias].Incr` and `.Clear`: bias autoincrement (4 bits) and clear
 *   (1 bit), with eight sections.
 *
 * Pack source/destination modifiers each have four sections:
 * - `AddrMod[Src][Y]` and `AddrMod[Dest][Y]` expose `.Incr` (4 bits), `.CR`
 *   (1 bit), and `.Clear` (1 bit).
 * - `AddrMod[Src][Z]` and `AddrMod[Dest][Z]` expose `.Incr` (1 bit) and
 *   `.Clear` (1 bit). Z has no CR field.
 *
 * Unsupported combinations, such as `AddrMod[SrcA][Y]` or
 * `AddrMod[Src][Z].CR`, fail at compile time.
 *
 */
inline constexpr AddrModFields AddrMod {};

class SfpuStack
{ // Config bits for the SFPU stack mode
public:
    static constexpr Field Incr {RegisterScope::Thread, 16, 36, 0, 0, 10, 1, 0}; // Dest SP autoincrement amount (10b)
};

class UnpackMiscCfg
{ // Unpacker misc config
public:
    static constexpr Field CfgContextOffset_0 {
        RegisterScope::Thread, 16, 41, 0, 0, 4, 1, 0}; // Unpacker 0 cfg context offset added to the context id from instruction field or context counter. Final
                                                       // context id is computed as CfgContextOffset + (AutoIncContextId ? CfgContextCnt : CfgContextId) (4b)
    static constexpr Field CfgContextCntReset_0 {RegisterScope::Thread, 16, 41, 0, 4, 1, 1, 0}; // Reset unpacker 0 config context counter. Write to the
                                                                                                // register with bit set to 1 will clear context counter (1b)
    static constexpr Field CfgContextCntInc_0 {RegisterScope::Thread, 16, 41, 0, 5, 1, 1, 0};   // Increment unpacker 0 config context counter. Write to the
                                                                                              // register with bit set to 1 will increment context counter. (1b)
    static constexpr Field CfgContextOffset_1 {
        RegisterScope::Thread, 16, 41, 0, 8, 4, 1, 0}; // Unpacker 1 cfg context offset added to the context id from instruction field or context counter. Final
                                                       // context id is computed as CfgContextOffset + (AutoIncContextId ? CfgContextCnt : CfgContextId) (4b)
    static constexpr Field CfgContextCntReset_1 {RegisterScope::Thread, 16, 41, 0, 12, 1, 1, 0}; // Reset unpacker 1 config context counter. Write to the
                                                                                                 // register with bit set to 1 will clear context counter (1b)
    static constexpr Field CfgContextCntInc_1 {
        RegisterScope::Thread, 16, 41, 0, 13, 1, 1, 0}; // Increment unpacker 0 config context counter. Write to the
                                                        // register with bit set to 1 will increment context counter. (1b)
    static constexpr Field CfgContextCntReset_metadata {
        RegisterScope::Thread, 16, 41, 0, 14, 1, 1, 0}; // Reset metadata unpack config context counter. Write to the register with bit set to 1 will clear
                                                        // context counter (1b)
    static constexpr Field CfgContextCntReset_metadata_zstart {
        RegisterScope::Thread, 16, 41, 0, 15, 1, 1, 0}; // Reset metadata unpack z_start. Write to the register with bit set to 1 will clear context counter
                                                        // (1b)
};

class NocOverlayMsgClear
{ // A write to this register triggers pop from data and message FIFOs in NOC overlay to free up space. This register is written after last unpack instruction
public:
    // once data has been read by TDMA engine.
    static constexpr Field StreamId_0 {RegisterScope::Thread, 16, 42, 0, 0, 6, 1, 0}; // Noc overlay stream id for unpacker 0 (6b)
    static constexpr Field MsgNum_0 {RegisterScope::Thread, 16, 42, 0, 8, 3, 1, 0};   // Number of messages(tiles) to pop from message fifo for unpacker 0 (3b)
    static constexpr Field StreamId_1 {RegisterScope::Thread, 16, 42, 1, 0, 6, 1, 0}; // Noc overlay stream id unpacker 1 (6b)
    static constexpr Field MsgNum_1 {RegisterScope::Thread, 16, 42, 1, 8, 3, 1, 0};   // Number of messages(tiles) to pop from message fifo for unpacker 1 (3b)
};

class PerfCntCmdEntry
{
private:
    template <std::uint32_t Index>
    class Fields
    {
    public:
        static constexpr Field Start {RegisterScope::Thread, 16, 44, 0, 2 * Index, 1, 1, 0};    // Start perf count Index (1b)
        static constexpr Field Stop {RegisterScope::Thread, 16, 44, 0, 2 * Index + 1, 1, 1, 0}; // End perf count Index (1b)
    };

public:
    const Field& Start;
    const Field& Stop;

    template <std::uint32_t Index>
    static constexpr PerfCntCmdEntry make()
    {
        static_assert(Index < 4, "performance counter index out of range");
        return {Fields<Index>::Start, Fields<Index>::Stop};
    }
};

inline constexpr PerfCntCmdEntry PerfCntCmd[] = {
    // Performance counter register
    PerfCntCmdEntry::make<0>(),
    PerfCntCmdEntry::make<1>(),
    PerfCntCmdEntry::make<2>(),
    PerfCntCmdEntry::make<3>(),
};

class EnableAccStats
{ // enable generating histogram of exponents
public:
    static constexpr Field Enable {RegisterScope::Thread, 16, 45, 0, 0, 1, 1, 0}; // enable (1b)
};

class FpuBiasSel
{ // Select upper or lower 32 bias values
public:
    static constexpr Field Pointer {RegisterScope::Thread, 16, 46, 0, 0, 1, 1, 0}; // When set, selects bias values 32 to 63 (1b)
};

class Fp16aForce
{ // Read dest like FP16A in int mode (for move int8 ops)
public:
    static constexpr Field Enable {RegisterScope::Thread, 16, 55, 0, 0, 1, 1, 0}; // When set, performs move ops like FP16A (1b)
};

class TensixTriscSync
{ // Selectively enable/disable hardware hazard detection between a TRISC and the Tensix core. The tt_tensix_trisc_sync (TTS) module works by snooping on
public:
    // accesses from the TRISC to  either a register file or the instructios buffer. If a particular resource is "tracked"  (by enabling the corresponding bit
    // in this register), the TTS unit will maintain a shadow  copy of all outstanding accesses to that resource.
    static constexpr Field TrackGlobalCfg {
        RegisterScope::Thread, 16, 56, 0, 0, 1, 1, 0}; // If 1, TRISC memory-mapped accesses to global config registers (in the CfgExu and also  including the
                                                       // so-called THCON register in tt_tdma) will be tracked. If Tensix  instructions are also tracked, then
                                                       // Tensix instructions and CFG accesses will be automatically stalled to prevent hazards. (1b)
    static constexpr Field EnSubdividedCfgForUnpacr {
        RegisterScope::Thread,
        16,
        56,
        0,
        1,
        1,
        1,
        0}; // If 1, the Tensix-TRISC sync unit will automatically determine what subset of CFG registers are targeted by an UNPACR/UNPACR_NOP instruction and
            // RISC memory-mapper accesses at runtime (the supported subsets are unpacker 0 registers, unpacker 1 registers, and everything else). This can
            // prevent unnecessary stalls between two CFG accesses that target different subsets. Otherwise, if this register is 0, the Tensix-TRISC sync unit
            // will pretend that any CFG access (regardless of its subset) can conflict with any other CFG access (regardless of its subset) (1b)
    static constexpr Field TrackGPR {
        RegisterScope::Thread, 16, 56, 0, 2, 1, 1, 0}; // If 1, TRISC memory-mapped accesses to general purpose registers (in tt_gpr_file) will  be tracked. If
                                                       // Tensix instructions are also tracked, then Tensix instructions and GPR  accesses will be automatically
                                                       // stalled to prevent hazards. (1b)
    static constexpr Field TrackTDMARegs {
        RegisterScope::Thread, 16, 56, 0, 3, 1, 1, 0}; // If 1, TRISC memory-mapped accesses to TDMA registers (in the RISC instruction interface in tt_tdma,
                                                       // not to be confused with THCON registers) will be tracked. If Tensix  instructions are also tracked,
                                                       // then Tensix instructions and TDMA reg accesses will be  automatically stalled to prevent hazards. (1b)
    static constexpr Field TrackTensixInstructions {
        RegisterScope::Thread, 16, 56, 0, 4, 1, 1, 0}; // If 1, Tensix instructions will be tracked. If at least one type of register access is also tracked,
                                                       // then Tensix instructions and the selected type of register accesses will be automatically stalled to
                                                       // prevent hazards. (1b)
};

class StreamwaitPhaseHi
{ // This config register stores extra data that don't fit within the STREAMWAIT opcode itself
public:
    static constexpr Field Val {
        RegisterScope::Thread, 16, 57, 0, 0, 10, 1, 0}; // The 10-bit target_val value in the STREAMWAIT opcode is appended to this value. For example, if this
                                                        // was 0x3FF and target_val was 2, then STREAMWAIT would wait for phase 0xFFC02. (10b)
};

class StreamwaitNumMsgsHi
{ // This config register stores extra data that don't fit within the STREAMWAIT opcode itself
public:
    static constexpr Field Val {
        RegisterScope::Thread, 16, 58, 0, 0, 7, 1, 0}; // The 10-bit target_val value in the STREAMWAIT opcode is appended to this value. For example, if this
                                                       // was 0x7F and target_val was 2, then STREAMWAIT would wait for 0x1FC02 messages to be received. (7b)
};

class StreamIdSync
{ // Select a stream to be used for Sync Exu
public:
    static constexpr Field BankSel {
        RegisterScope::Thread, 16, 59, 0, 0, 6, 4, 16}; // Selects which stream to use for stallwait stream instructions (3-bit group id, 3-bit stream id) (6b)
};

class StreamIdTrisc
{ // Select a stream to be used to be read by TRISC
public:
    static constexpr Field BankSel {RegisterScope::Thread, 16, 63, 0, 0, 6, 4, 16}; // Selects which stream to use for internally mapping to Trisc registers
                                                                                    // (3-bit group id, 3-bit stream id) (6b)
};

class TensixCsrConfig
{ // Modifies behaviour of qstatus, bstatus, and stream CSR bits.
public:
    static constexpr Field RawBusyStatus {
        RegisterScope::Thread, 16, 67, 0, 0, 1, 1, 0}; // If high, the bstatus CSR will only report whether a given execution unit is currently busy. If low,
                                                       // the bstatus bits  will be the OR of the busy status and the queue status. In other words, if this bit
                                                       // is 0, reading the bstatus will tell  you whether an execution unit is busy and/or an instruction is
                                                       // queued up for it. If this bit is 1, bstatus will only tell you if the execution unit it busy (1b)
};

} // namespace cfg
} // namespace hal
