// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_unpack_common.h (+ the unpack helpers of common/inc/cunpack_common.h
// it calls). Same instruction sequence and access kinds as production, expressed through the HAL.

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h" // constants, types and pure helpers only (p_gpr_unpack, masked_data_format, canonical_*, ...)
#include "hal/cfg.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_assert.h"
#include "llk_memory_checks.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// If `x` is the result of loading from memory, placing `consume_discard(x)` somewhere
// will ensure that code after `consume_discard(x)` doesn't start until the load is complete.
#define consume_discard(x) __asm volatile("andi x0, %0, 0" : : "r"((x)) : "memory")

// Reconfig behaviour for dim and stride
enum class p_dim_stride_target
{
    IGNORE,        // Do not modify dim/stride
    FACE_ROW_MAJOR // Set dim/stride for unpacking face in row major format
};

/**
 * @brief Store a value to memory then read it back, fencing subsequent code on the store.
 *
 * The dependent load cannot retire until the store completes, so any instruction consuming the
 * returned value observes the store as finished. Used to serialize against pending memory writes.
 *
 * @param addr: Memory location to store to and load back from.
 * @param to_store: Value written to addr.
 * @return The value read back from addr after the store.
 */
inline __attribute__((always_inline)) std::uint32_t store_then_load(volatile std::uint32_t *addr, std::uint32_t to_store)
{
    // RISC-only memory fence, no Tensix instruction or CFG access: kept as in production.
    std::uint32_t result;
    __asm volatile("sw %2, %1; lw %0, %1" : "=r"(result) : "m"(*addr), "r"(to_store));
    return result;
}

namespace hal_port
{

namespace adc = hal::counters::adc;

// Semaphore / mutex indices come from production's ckernel::semaphore / ckernel::mutex.
inline constexpr hal::sync::Semaphore UNPACK_SYNC_SEM = static_cast<hal::sync::Semaphore>(ckernel::semaphore::UNPACK_SYNC);
inline constexpr hal::sync::Mutex REG_RMW_MUTEX       = static_cast<hal::sync::Mutex>(ckernel::mutex::REG_RMW);

// Place value in field F of section S, for composing a complete CFG word (equivalent of production's bitfield unions).
template <const hal::cfg::Field &F, hal::cfg::Sec S = hal::cfg::Sec::S0>
constexpr std::uint32_t field_bits(const std::uint32_t value)
{
    return (value << F.shamt(S)) & F.mask(S);
}

// Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
inline void unpacker_addr_counter_init()
{
    // TTI_SETADCXY(UNP_A | UNP_B, 0, 0, 0, 0, 0b1011): channel 0 X and Y, channel 1 Y
    adc::set<adc::AddrCounter {.channel0 = {.x = 0, .y = 0}, .channel1 = {.y = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
    // TTI_SETADCZW(UNP_A | UNP_B, 0, 0, 0, 0, 0b1111): Z and W of both channels
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
}

// Wait for threshold of busy contexts to fall below total available contexts
inline void wait_for_next_context(const std::uint32_t num_contexts)
{
    while (hal::sync::semaphore::read<UNPACK_SYNC_SEM>() >= num_contexts)
    {
    }
}

inline void switch_config_context(std::uint32_t &unp_cfg_context)
{
    namespace cfg = hal::cfg;
    // Switch config context. Branch between the two immediate SETC16 encodings, as production: a runtime-encoded
    // SETC16 is an instruction-buffer store computed on the RISC, which costs cycles on RISC-bound unpack loops.
    unp_cfg_context = 1 - unp_cfg_context;
    if (unp_cfg_context == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 1>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 1>());
    }
}

inline void reset_config_context()
{
    namespace cfg = hal::cfg;
    // Switch config context
    unp_cfg_context = 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
}

// Sync on unpacker idle via waiting busy contexts counter 0
inline void wait_for_idle()
{
    while (hal::sync::semaphore::read<UNPACK_SYNC_SEM>() > 0)
    {
    }
}

/**
 * @brief How unpack ops order RISC stores to unpacker config against the UNPACRs that read it.
 *
 * Handshake: production's two config contexts, the UNPACK_SYNC busy-context semaphore and a STALLWAIT on RISC config
 * writes; Auto TTSync off. AutoTTSync: one config context, rewritten in place; Auto TTSync orders the stores
 * (AutoTTSync.md). Each op's init asks for the mode it needs through @ref unpack_sync_mode; Auto TTSync is sticky per
 * thread, so the op's uninit asks for Handshake again. Every kernel starts in Handshake: the boot code (trisc.cpp) turns
 * Auto TTSync off and crt0 zeroes the mode flag. configure_unpack_AB leaves the mode alone, so an init may come before it.
 */
enum class UnpackSyncMode : std::uint8_t
{
    Handshake,
    AutoTTSync
};

// Mode the hardware is in, as last set by unpack_sync_mode (Handshake at kernel start).
inline UnpackSyncMode unpack_sync_mode_state = UnpackSyncMode::Handshake;

// Push the TENSIX_TRISC_SYNC write for mode M, then a Tensix SEMPOST of UNPACK_SYNC that cannot pass the Wait Gate before
// the Configuration Unit has executed that write. Callers have waited for UNPACK_SYNC to read 0.
template <UnpackSyncMode M>
inline void push_unpack_sync_mode()
{
    namespace cfg              = hal::cfg;
    namespace sync             = hal::sync;
    constexpr std::uint32_t on = M == UnpackSyncMode::AutoTTSync ? 1 : 0;
    if constexpr (M == UnpackSyncMode::AutoTTSync)
    {
        // SETC16.md erratum: Auto TTSync only tracks the right Config bank once a SETC16 of CFG_STATE_ID has executed
        // after reset. Write the bank the software tracker selects, so the two cannot disagree (llk_hw_cleanup flips it).
        cfg::write<cfg::Access::TensixCfgUnit, cfg::CfgStateId::StateID, cfg::Sec::S0>(ckernel::cfg_state_id);
    }
    // Track backend config (split per unpacker) against Tensix instructions, or nothing.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::TensixTriscSync::TrackGlobalCfg, cfg::Sec::S0, on>(),
        cfg::set<cfg::TensixTriscSync::EnSubdividedCfgForUnpacr, cfg::Sec::S0, on>(),
        cfg::set<cfg::TensixTriscSync::TrackGPR, cfg::Sec::S0, 0>(),
        cfg::set<cfg::TensixTriscSync::TrackTDMARegs, cfg::Sec::S0, 0>(),
        cfg::set<cfg::TensixTriscSync::TrackTensixInstructions, cfg::Sec::S0, on>());
    // AutoTTSync.md: a change to TENSIX_TRISC_SYNC needs a mechanism outside Auto TTSync to wait for it. SETC16 runs in the
    // Configuration Unit; hold the SEMPOST until that unit is idle. Instructions pass the Wait Gate in order, so once the
    // SEMPOST has executed, the SETC16 and every earlier instruction (and so every earlier UNPACR's base address read,
    // MemoryOrdering.md) are done with the gate. Unlike tensix_sync(), this does not wait for UNPACRs still unpacking.
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::ConfigUnitIdle>();
    sync::semaphore::post<UNPACK_SYNC_SEM>();
}

// Second half of push_unpack_sync_mode: wait until its SEMPOST has executed, then take the count back (Tensix SEMGET, so
// a later wait_for_idle also waits for it).
inline void wait_unpack_sync_mode()
{
    while (hal::sync::semaphore::read<UNPACK_SYNC_SEM>() == 0)
    {
    }
    hal::sync::semaphore::get<UNPACK_SYNC_SEM>();
}

/**
 * @brief Put the unpack thread in sync mode M. No-op when it already is.
 *
 * Waits for the two-context ops in flight to free their contexts, writes the mode, and waits until the write and every
 * earlier instruction have passed the Wait Gate, so the next RISC config store may rewrite any unpacker base.
 */
template <UnpackSyncMode M>
inline void unpack_sync_mode()
{
    if (unpack_sync_mode_state == M)
    {
        return;
    }
    wait_for_idle();
    push_unpack_sync_mode<M>();
    wait_unpack_sync_mode();
    unpack_sync_mode_state = M;
}

/**
 * @brief @ref unpack_sync_mode split in two, so an init can overlap the wait with its own setup.
 *
 * begin pushes the mode change (nothing when the mode is already M) and returns whether it did; end waits for it.
 * Between the two the caller may push Tensix instructions and program the MOP, but must not store to unpacker config
 * from the RISC or call @ref unpack_sync_mode.
 */
template <UnpackSyncMode M>
inline bool unpack_sync_mode_begin()
{
    if (unpack_sync_mode_state == M)
    {
        return false;
    }
    wait_for_idle();
    push_unpack_sync_mode<M>();
    unpack_sync_mode_state = M;
    return true;
}

inline void unpack_sync_mode_end(const bool pushed)
{
    if (pushed)
    {
        wait_unpack_sync_mode();
    }
}

template <bool is_fp32_dest_acc_en, bool row_pool = false, bool fpu_srnd_en = false, bool pack_srnd_en = false>
inline void configure_unpack_AB(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const bool transpose_xy_srca_en     = false,
    const std::uint32_t unpA_num_faces  = 4,
    const std::uint32_t unpB_num_faces  = 4)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpA_src_format), static_cast<DataFormat>(unpA_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");
    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpB_src_format), static_cast<DataFormat>(unpB_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");

    // Check that unpacker is done (all contexts freed up) before starting hw configuration
    hal_port::wait_for_idle();

    // Reset address counters
    hal_port::unpacker_addr_counter_init();

    const std::uint32_t unpA_src_format_masked = masked_data_format(unpA_src_format);
    const std::uint32_t unpB_src_format_masked = masked_data_format(unpB_src_format);
    const std::uint32_t unpA_dst_format_masked = masked_data_format(unpA_dst_format);
    const std::uint32_t unpB_dst_format_masked = masked_data_format(unpB_dst_format);

    std::uint32_t unpA_ch1_x_stride = datum_size_in_bytes(unpA_dst_format_masked);
    std::uint32_t unpB_ch1_x_stride = datum_size_in_bytes(unpB_dst_format_masked);
    std::uint32_t unpA_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpA_ch1_x_stride;
    std::uint32_t unpB_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpB_ch1_x_stride;
    std::uint32_t exp_width         = (static_cast<std::uint32_t>(unpA_dst_format_masked) >> 2) & 0x1; // 0=5-bit, 1=8-bit

    // Strides for incrementing ch1 address to srcA and srcB
    // Z and W(not used) stride for dest address (ch1): whole-word MMIO stores
    constexpr auto Unp0Ch1 = cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1];
    constexpr auto Unp1Ch1 = cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1];
    cfg::write<cfg::Access::MMIO, Unp0Ch1.Zstride, cfg::Sec::S0, 1>(
        std::array<std::uint32_t, 1> {field_bits<Unp0Ch1.Wstride>(0) | field_bits<Unp0Ch1.Zstride>(unpA_ch1_z_stride)});
    cfg::write<cfg::Access::MMIO, Unp1Ch1.Zstride, cfg::Sec::S0, 1>(
        std::array<std::uint32_t, 1> {field_bits<Unp1Ch1.Wstride>(0) | field_bits<Unp1Ch1.Zstride>(unpB_ch1_z_stride)});

    // Establish the canonical Y-stride baseline for srcA (ch1) here, so per-op inits and
    // uninits that touch the Y stride (e.g. tilizeA_B) set and re-assert this baseline
    // rather than snapshotting the previous register value into a GPR.
    cfg::write<cfg::Access::TensixCfgUnit, Unp0Ch1.Ystride, cfg::Sec::S0>(canonical_unpA_y_stride(unpA_dst_format_masked));

    // Math ALU_FORMAT_REG
    sync::mutex::acquire<REG_RMW_MUTEX>();

    // FP32 accumulation and SFPU to read dest as FP32
    // NOTE: This assumes these config fields are adjacent and in same register!!
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu_srnd_en>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack_srnd_en>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack_srnd_en>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0>(unpA_src_format == to_underlying(DataFormat::UInt8) ? 1 : 0),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0>(unpB_src_format == to_underlying(DataFormat::UInt8) ? 1 : 0));

    // Set FP8 E4M3 mode, bit is accessible by unpacker/packer
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S0>(
        ((unpA_src_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3)) ? 1 : 0);
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S1>(
        ((unpB_src_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3)) ? 1 : 0);

    sync::mutex::release<REG_RMW_MUTEX>();

    // Set tile descriptor (TILE_DESC_SIZE words per unpacker)
    static_assert(TILE_DESC_SIZE == 2, "tile descriptor is composed of 2 words");
    constexpr auto TileDsc = cfg::Thcon[cfg::Reg0].TileDescriptor;
    // Input tile is uncompressed; x_dim is not used for unpA as the value is overridden by per context x_dim set below. Used for unpB
    cfg::write<cfg::Access::MMIO, TileDsc.Raw, cfg::Sec::S0, TILE_DESC_SIZE>(std::array<std::uint32_t, TILE_DESC_SIZE> {
        field_bits<TileDsc.InDataFormat, cfg::Sec::S0>(unpA_src_format_masked) | field_bits<TileDsc.Uncompressed, cfg::Sec::S0>(1) |
            field_bits<TileDsc.XDim, cfg::Sec::S0>(0),
        field_bits<TileDsc.YDim, cfg::Sec::S0>(1) | field_bits<TileDsc.ZDim, cfg::Sec::S0>(unpA_num_faces)});
    cfg::write<cfg::Access::MMIO, TileDsc.Raw, cfg::Sec::S1, TILE_DESC_SIZE>(std::array<std::uint32_t, TILE_DESC_SIZE> {
        field_bits<TileDsc.InDataFormat, cfg::Sec::S1>(row_pool ? to_underlying(DataFormat::Float32) : unpB_src_format_masked) |
            field_bits<TileDsc.Uncompressed, cfg::Sec::S1>(1) | field_bits<TileDsc.XDim, cfg::Sec::S1>(unpB_face_r_dim * FACE_C_DIM),
        field_bits<TileDsc.YDim, cfg::Sec::S1>(1) | field_bits<TileDsc.ZDim, cfg::Sec::S1>(unpB_num_faces)});

    // Set unpacker config (CONFIG_SIZE words per unpacker)
    static_assert(CONFIG_SIZE == 2, "unpacker config is composed of 2 words");
    constexpr auto R2 = cfg::Thcon[cfg::Reg2];
    // word 1: uncompress_cntx0_3 = uncompress_cntx4_7 = 0xf, i.e. zero-compression disabled in all 8 contexts
    const std::uint32_t uncompress_all =
        field_bits<R2.Disable_zero_compress_cntx0, cfg::Sec::S0>(1) | field_bits<R2.Disable_zero_compress_cntx1, cfg::Sec::S0>(1) |
        field_bits<R2.Disable_zero_compress_cntx2, cfg::Sec::S0>(1) | field_bits<R2.Disable_zero_compress_cntx3, cfg::Sec::S0>(1) |
        field_bits<R2.Disable_zero_compress_cntx4, cfg::Sec::S0>(1) | field_bits<R2.Disable_zero_compress_cntx5, cfg::Sec::S0>(1) |
        field_bits<R2.Disable_zero_compress_cntx6, cfg::Sec::S0>(1) | field_bits<R2.Disable_zero_compress_cntx7, cfg::Sec::S0>(1);
    cfg::write<cfg::Access::MMIO, R2.Out_data_format, cfg::Sec::S0, CONFIG_SIZE>(std::array<std::uint32_t, CONFIG_SIZE> {
        field_bits<R2.Out_data_format, cfg::Sec::S0>(unpA_dst_format_masked) | field_bits<R2.Throttle_mode, cfg::Sec::S0>(2) |
            field_bits<R2.Haloize_mode, cfg::Sec::S0>(transpose_xy_srca_en ? 1 : 0),
        uncompress_all});

    cfg::write<cfg::Access::MMIO, R2.Out_data_format, cfg::Sec::S1, CONFIG_SIZE>(std::array<std::uint32_t, CONFIG_SIZE> {
        field_bits<R2.Out_data_format, cfg::Sec::S1>(row_pool ? (to_underlying(DataFormat::Float16) | (exp_width << 2)) : unpB_dst_format_masked) |
            field_bits<R2.Throttle_mode, cfg::Sec::S1>(2) | field_bits<R2.Haloize_mode, cfg::Sec::S1>(0),
        uncompress_all});

    // Program base address for all 2 sections (each section address is loaded to corresponding context)
    // Load dummy data to unused location if face height is 0
    constexpr auto R5                      = cfg::Thcon[cfg::Reg5];
    const std::uint32_t Dest_cntx0_address = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    const std::uint32_t Dest_cntx1_address = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    cfg::write<cfg::Access::MMIO, R5.Dest_cntx0_address, cfg::Sec::S0, 1>(
        std::array<std::uint32_t, 1> {field_bits<R5.Dest_cntx0_address>(Dest_cntx0_address) | field_bits<R5.Dest_cntx1_address>(Dest_cntx1_address)});

    // Program unpacker0 per context x_dim (face size in l1)
    // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
    const std::uint32_t face_dim = unpA_face_r_dim * FACE_C_DIM;
    cfg::write<cfg::Access::MMIO, R5.Tile_x_dim_cntx0, cfg::Sec::S0, 1>(
        std::array<std::uint32_t, 1> {field_bits<R5.Tile_x_dim_cntx0>(face_dim) | field_bits<R5.Tile_x_dim_cntx1>(face_dim)});

    // HAL-GAP: RISC MMIO stores to the GPR register file (regfile[...] = v) and the load that fences them
    // (sync_regfile_write) have no HAL API; kept as in production.
    constexpr std::uint32_t face_dim_16x16        = FACE_R_DIM * FACE_C_DIM;
    ckernel::regfile[p_gpr_unpack::FACE_DIM_8x16] = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    ckernel::regfile[p_gpr_unpack::FACE_DIM_4x16] = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    ckernel::regfile[p_gpr_unpack::FACE_DIM_2x16] = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    ckernel::regfile[p_gpr_unpack::FACE_DIM_1x16] = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    ckernel::sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);

    // TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4)
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::SrcASet::Base, cfg::Sec::S0, 0>(), cfg::set<cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, 1>());

    // Enable address counter for unpacker ch1/dst address
    // final address is calculated as: Dest_cntx0/1_address + address_counter_ch1
    // used for face by face unpacking of entire tile into srcA
    cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>(
        std::array<std::uint32_t, 1> {field_bits<cfg::Unpacker[0].AddDestAddrCntr>(1)});

    // Clear context ID
    hal_port::reset_config_context();
    // The unpack sync mode is left as it is: ops that use one context with Auto TTSync turn it on in their init
    // (hal_port::unpack_sync_mode), which may come before this function.
}

template <std::uint32_t UNP_SEL, std::uint32_t face_r_dim>
inline void set_unpacker_x_end()
{
    // TTI_SETADCXX(UNP_SEL, face_r_dim * FACE_C_DIM - 1, 0x0): X start on channel 0, X end on channel 1
    constexpr adc::AddrCounter x_end {.channel0 = {.x = 0}, .channel1 = {.x = face_r_dim * FACE_C_DIM - 1}};
    if constexpr (UNP_SEL == p_setadc::UNP_A)
    {
        adc::set<x_end.for_clients<adc::Client::Unpacker0>()>();
    }
    else if constexpr (UNP_SEL == p_setadc::UNP_B)
    {
        adc::set<x_end.for_clients<adc::Client::Unpacker1>()>();
    }
    else
    {
        adc::set<x_end.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
    }
}

#ifndef HAL_PORT_UNPACK_TO_DEST_SEM_DEFINED
#define HAL_PORT_UNPACK_TO_DEST_SEM_DEFINED
// Shared with llk_math_common.h; whichever header is included first defines it.
inline constexpr hal::sync::Semaphore UNPACK_TO_DEST_SEM = static_cast<hal::sync::Semaphore>(ckernel::semaphore::UNPACK_TO_DEST);
#endif

// Wait (on the unpacker) until math has freed dest for an unpack-to-dest tile.
inline void wait_for_dest_available()
{
    // t6_semaphore_wait_on_max<p_stall::STALL_UNPACK>(semaphore::UNPACK_TO_DEST)
    hal::sync::wait::semaphore<hal::sync::StallTarget::Unpack, hal::sync::SemaphoreCondition::WhileMaximum, UNPACK_TO_DEST_SEM>();
}

// Restore srcA channel-1 Z-stride to the canonical baseline derived from unpack_dst_format.
// This pairs with set_dst_write_addr to bracket the unpack-to-dest section: rather than
// snapshotting the prior register value into a GPR, we recompute the canonical baseline so the
// restore is deterministic and order-independent.
// Callers release UNPACK_SYNC after this returns, so configure_unpack_AB doesn't race these config writes.
inline void unpack_to_dest_tile_done(std::uint32_t &context_id, const std::uint32_t unpack_dst_format)
{
    namespace cfg          = hal::cfg;
    namespace sync         = hal::sync;
    constexpr auto Unp0Ch1 = cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1];
    constexpr auto R2      = cfg::Thcon[cfg::Reg2];
    constexpr auto R5      = cfg::Thcon[cfg::Reg5];
    static_assert(Unp0Ch1.Zstride.addr32(cfg::Sec::S0) == UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32, "Unp0 ch1 Z-stride word");
    static_assert(R2.Unpack_if_sel_cntx0.addr32(cfg::Sec::S0) == THCON_SEC0_REG2_Unpack_if_sel_cntx0_ADDR32, "Unpack_if_sel_cntx0 word");
    static_assert(R2.Unpack_if_sel_cntx1.addr32(cfg::Sec::S0) == THCON_SEC0_REG2_Unpack_if_sel_cntx1_ADDR32, "Unpack_if_sel_cntx1 word");
    static_assert(R5.Dest_cntx0_address.addr32(cfg::Sec::S0) == THCON_SEC0_REG5_Dest_cntx0_address_ADDR32, "Dest_cntx0_address word");
    static_assert(R5.Dest_cntx1_address.addr32(cfg::Sec::S0) == THCON_SEC0_REG5_Dest_cntx1_address_ADDR32, "Dest_cntx1_address word");

    // t6_semaphore_post<p_stall::UNPACK0>(semaphore::UNPACK_TO_DEST)
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::Unpacker0Idle>();
    sync::semaphore::post<UNPACK_TO_DEST_SEM>();
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TMP_LO>(), LOWER_HALFWORD(canonical_unpA_z_stride(unpack_dst_format) << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT));
    // Restore canonical Z-stride
    cfg::write<cfg::Access::TensixCfgUnit, Unp0Ch1.Zstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_unpack::TMP_LO>());
    // Restore config context
    if (context_id == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit, R2.Unpack_if_sel_cntx0, cfg::Sec::S0, 0>();
        cfg::write<cfg::Access::TensixCfgUnit, R5.Dest_cntx0_address, cfg::Sec::S0, 4 * 16>();
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit, R2.Unpack_if_sel_cntx1, cfg::Sec::S0, 0>();
        cfg::write<cfg::Access::TensixCfgUnit, R5.Dest_cntx1_address, cfg::Sec::S0, 4 * 16>();
    }
    // TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4): re-enable address bit swizzle
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::SrcASet::Base, cfg::Sec::S0, 0>(), cfg::set<cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, 1>());
    // No TEN-3868 dummy unpack to SrcA here, unlike Wormhole: Blackhole's LLK sets SrcA's dvalid without an unpack
    // only in an UNPACR_NOP that also zeroes SrcA (UNP_ZEROSRC with SET_DVALID), and that clear moves both of SrcA's
    // bank pointers
}

inline void set_dst_write_addr(const std::uint32_t &context_id, const std::uint32_t &unpack_dst_format)
{
    namespace cfg          = hal::cfg;
    constexpr auto Unp0Ch1 = cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1];
    constexpr auto R2      = cfg::Thcon[cfg::Reg2];
    constexpr auto R5      = cfg::Thcon[cfg::Reg5];

    // HAL-GAP: no HAL API for the RISC mailbox read of math's dest offset; production's mailbox_read is kept.
    std::uint32_t dst_byte_addr = 16 * (4 + mailbox_read(ThreadId::MathThreadId)); // Apply fixed offset of 4*16 to dest address
    // TTI_SETC16(SRCA_SET_Base_ADDR32, 0x0): disable address bit swizzle
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::SrcASet::Base, cfg::Sec::S0, 0>(), cfg::set<cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, 0>());
    // Set unpacker Z-stride to the canonical baseline for unpack_dst_format. Paired with
    // unpack_to_dest_tile_done's canonical restore; no GPR snapshot of the prior value.
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TMP_LO>(), LOWER_HALFWORD(canonical_unpA_z_stride(unpack_dst_format) << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT));
    cfg::write<cfg::Access::TensixCfgUnit, Unp0Ch1.Zstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_unpack::TMP_LO>());
    if (context_id == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit, R2.Unpack_if_sel_cntx0, cfg::Sec::S0, 1>();
        cfg::write<cfg::Access::TensixCfgUnit, R5.Dest_cntx0_address, cfg::Sec::S0>(dst_byte_addr);
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit, R2.Unpack_if_sel_cntx1, cfg::Sec::S0, 1>();
        // HAL-GAP (codegen): cfg::write of the upper-halfword field re-derives its bytes with three shifts per tile and
        // costs the caller's tile loop a register; production's RMW lets GCC share the context-0 byte. Same RMWCIBs.
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(dst_byte_addr);
    }
}

template <std::uint32_t UNP_SEL = p_setadc::UNP_AB>
inline void config_unpacker_x_end(const std::uint32_t face_r_dim)
{
    static_assert(UNP_SEL == p_setadc::UNP_A || UNP_SEL == p_setadc::UNP_B || UNP_SEL == p_setadc::UNP_AB, "UNP_SEL must be UNP_A, UNP_B, or UNP_AB");
    LLK_ASSERT(
        face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == FACE_R_DIM, "face_r_dim must be 1, 2, 4, 8, or FACE_R_DIM");

    switch (face_r_dim)
    {
        case 1:
            set_unpacker_x_end<UNP_SEL, 1>();
            break;
        case 2:
            set_unpacker_x_end<UNP_SEL, 2>();
            break;
        case 4:
            set_unpacker_x_end<UNP_SEL, 4>();
            break;
        case 8:
            set_unpacker_x_end<UNP_SEL, 8>();
            break;
        default:
            set_unpacker_x_end<UNP_SEL, FACE_R_DIM>();
            break;
    }
}

} // namespace hal_port

/**
 * @brief Configure the unpacker hardware for both operands A and B.
 *
 * Programs the per-operand source/destination data formats, face dimensions and face counts via
 * configure_unpack_AB, and stores the per-operand tile sizes into the unpack GPRs.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @param unpA_src_format: Source data format of operand A in L1.
 * @param unpB_src_format: Source data format of operand B in L1.
 * @param unpA_dst_format: Destination data format operand A is converted to.
 * @param unpB_dst_format: Destination data format operand B is converted to.
 * @param unpA_face_r_dim: Rows per face for operand A.
 * @param unpB_face_r_dim: Rows per face for operand B.
 * @param unpA_num_faces: Number of faces for operand A, valid values = <1, 2, 4>.
 * @param unpB_num_faces: Number of faces for operand B, valid values = <1, 2, 4>.
 * @param unpA_tile_size: Tile size of operand A stored to the tile-size GPR.
 * @param unpB_tile_size: Tile size of operand B stored to the tile-size GPR.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_hw_configure_(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t unpA_face_r_dim,
    const std::uint32_t unpB_face_r_dim,
    const std::uint32_t unpA_num_faces,
    const std::uint32_t unpB_num_faces,
    const std::uint32_t unpA_tile_size = 0,
    const std::uint32_t unpB_tile_size = 0)
{
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");

    hal_port::configure_unpack_AB<is_fp32_dest_acc_en, false, false, false>(
        unpA_src_format, unpB_src_format, unpA_dst_format, unpB_dst_format, unpA_face_r_dim, unpB_face_r_dim, 0, unpA_num_faces, unpB_num_faces);

    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_A>(), LOWER_HALFWORD(unpA_tile_size));
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_B>(), LOWER_HALFWORD(unpB_tile_size));
}

/**
 * @brief Configure stochastic rounding for the unpacker, FPU and packer.
 *
 * Sets the ALU rounding-mode register bits enabling stochastic rounding on the units selected by
 * the template mode.
 *
 * @tparam stoch_rnd_mode: Which units use stochastic rounding, values = <None/Fpu/Pack/All>
 */
template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    namespace cfg               = hal::cfg;
    constexpr bool fpu_srnd_en  = (stoch_rnd_mode == StochRndType::All) || (stoch_rnd_mode == StochRndType::Fpu);
    constexpr bool pack_srnd_en = (stoch_rnd_mode == StochRndType::All) || (stoch_rnd_mode == StochRndType::Pack);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, fpu_srnd_en>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, pack_srnd_en>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, pack_srnd_en>());
}

/**
 * @brief Reprogram only the SrcA unpacker tile size and face geometry, leaving the data format untouched.
 *
 * Updates the SrcA tile-size GPR and programs the per-context X-dim (face width) and Z-dim (num_faces).
 * This is the shape half of @ref _llk_unpack_reconfig_data_format_srca_impl_, split out so a caller that
 * changes only the tile shape (not the format) can reprogram it without the THCON format writes or the
 * int8/unsigned RMW. The ch1 Z/Y strides are format-derived and are NOT touched here (they are owned by the
 * format reconfig); a shape-only change does not alter them.
 *
 * @tparam issue_stall: Issue the STALL_CFG/UNPACK0 drain before the config writes. Leave true for standalone
 *                      callers; @ref _llk_unpack_reconfig_data_format_srca_impl_ passes false because it has
 *                      already stalled for its format writes.
 * @param tile_size: New tile size (bytes) of operand A, stored to the tile-size GPR (tracks the shape).
 * @param unpack_face_r_dim: Rows per face.
 * @param unpack_num_faces: Number of faces, valid values = <1, 2, 4>.
 * @note A face_r_dim change also needs the op init re-run: the unpacker ADC X-end is programmed by the op
 *       init (config_unpacker_x_end / SETADCXX), not here. A num_faces-only change is fully handled here.
 */
template <bool issue_stall = true>
inline void _llk_unpack_reconfig_tile_shape_srca_(
    const std::uint32_t tile_size, const std::uint32_t unpack_face_r_dim = FACE_R_DIM, const std::uint32_t unpack_num_faces = 4)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    LLK_ASSERT(unpack_num_faces == 1 || unpack_num_faces == 2 || unpack_num_faces == 4, "unpack_num_faces must be 1, 2, or 4");

    if constexpr (issue_stall)
    {
        sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle>();
    }

    // Tile size (bytes) tracks the tile shape; refresh the GPR the unpack MOP uses to step the SrcA L1 base.
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_A>(), LOWER_HALFWORD(tile_size));

    // Program unpacker0 per context x_dim (face size in l1)
    // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
    constexpr auto R5 = cfg::Thcon[cfg::Reg5];
    static_assert(R5.Tile_x_dim_cntx0.addr32(cfg::Sec::S0) == THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, "Tile_x_dim word");
    const std::uint32_t face_dim = unpack_face_r_dim * FACE_C_DIM;
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R5.Tile_x_dim_cntx0, cfg::Sec::S0>(face_dim), cfg::set<R5.Tile_x_dim_cntx1, cfg::Sec::S0>(face_dim));

    // Set Z-dim to number of faces
    constexpr auto TileDsc = cfg::Thcon[cfg::Reg0].TileDescriptor;
    static_assert(TileDsc.ZDim.addr32(cfg::Sec::S0) == THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, "Z-dim word");
    cfg::write<cfg::Access::TensixCfgUnit, TileDsc.ZDim, cfg::Sec::S0>(unpack_num_faces);
}

/**
 * @brief Reprogram only the SrcB unpacker tile size and face geometry, leaving the data format untouched.
 *
 * Updates the SrcB tile-size GPR and programs the tile-descriptor X-dim (face width) and Z-dim (num_faces).
 * Shape half of @ref _llk_unpack_reconfig_data_format_srcb_impl_, split out so a caller that changes only the
 * tile shape can reprogram it without the THCON format writes or the int8/unsigned RMW. The ch1 Z-stride is
 * format-derived and is NOT touched here (owned by the format reconfig).
 *
 * @tparam issue_stall: Issue the STALL_CFG/UNPACK1 drain before the config writes. Leave true for standalone
 *                      callers; @ref _llk_unpack_reconfig_data_format_srcb_impl_ passes false because it has
 *                      already stalled for its format writes.
 * @param tile_size: New tile size (bytes) of operand B, stored to the tile-size GPR (tracks the shape).
 * @param unpack_face_r_dim: Rows per face.
 * @param unpack_num_faces: Number of faces, valid values = <1, 2, 4>.
 * @note See @ref _llk_unpack_reconfig_tile_shape_srca_: a face_r_dim change also needs the op init re-run.
 */
template <bool issue_stall = true>
inline void _llk_unpack_reconfig_tile_shape_srcb_(
    const std::uint32_t tile_size, const std::uint32_t unpack_face_r_dim = FACE_R_DIM, const std::uint32_t unpack_num_faces = 4)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    LLK_ASSERT(unpack_num_faces == 1 || unpack_num_faces == 2 || unpack_num_faces == 4, "unpack_num_faces must be 1, 2, or 4");

    if constexpr (issue_stall)
    {
        sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker1Idle>();
    }

    // Tile size (bytes) tracks the tile shape; refresh the GPR the unpack MOP uses to step the SrcB L1 base.
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_B>(), LOWER_HALFWORD(tile_size));

    constexpr auto TileDsc = cfg::Thcon[cfg::Reg0].TileDescriptor;
    static_assert(TileDsc.XDim.addr32(cfg::Sec::S1) == THCON_SEC1_REG0_TileDescriptor_ADDR32, "SrcB X-dim word");

    // Set X-dim to face_r_dim * FACE_C_DIM
    cfg::write<cfg::Access::TensixCfgUnit, TileDsc.XDim, cfg::Sec::S1>(unpack_face_r_dim * FACE_C_DIM);

    // Set Z-dim to number of faces
    cfg::write<cfg::Access::TensixCfgUnit, TileDsc.ZDim, cfg::Sec::S1>(unpack_num_faces);
}

/**
 * @brief Reconfigure the operand A (SrcA) data format at runtime.
 *
 * Updates the SrcA tile-descriptor source format, FP8 exponent mode, destination format and
 * tile-size GPR; optionally reprograms dim/stride registers for face-row-major unpacking.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam dim_stride_target: Whether to reprogram dim/stride, values = <IGNORE/FACE_ROW_MAJOR>
 * @tparam skip_int8: Skip re-deriving the SrcA signedness bit from the new format. Leave false (the default)
 *                    unless the caller guarantees the reconfig never crosses an Int8/UInt8/Int32 boundary and
 *                    wants to avoid the extra RMW; skipping it can leave a stale SrcAUnsigned bit (tt-metal#34499).
 * @param unpack_src_format: New source data format of operand A in L1.
 * @param unpack_dst_format: New destination data format operand A is converted to.
 * @param tile_size: New tile size of operand A stored to the tile-size GPR.
 * @param unpack_face_r_dim: Rows per face, used when reprogramming dim/stride.
 * @param unpack_num_faces: Number of faces, valid values = <1, 2, 4>.
 * @note The SrcA-unsigned ALU bit (ALU_FORMAT_SPEC_REG0_SrcAUnsigned), and the math-side INT8 math-enable in
 *       @ref _llk_math_reconfig_data_format_srca_, are re-derived from the new format by default, so a reconfig
 *       that transitions to OR from an 8-bit-integer / Int8 / Int32 format (e.g. UInt8 -> float) is handled
 *       automatically. Set skip_int8 = true only when the caller guarantees no such boundary is crossed;
 *       skipping leaves the previous unsigned/INT8 state, which the next op would then misinterpret.
 */
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool skip_int8 = false>
inline void _llk_unpack_reconfig_data_format_srca_impl_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t unpack_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpack_num_faces  = 4)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    LLK_ASSERT(unpack_num_faces == 1 || unpack_num_faces == 2 || unpack_num_faces == 4, "unpack_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpack_src_format), static_cast<DataFormat>(unpack_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle>();
    if constexpr (!skip_int8)
    {
        // Always re-derive SrcAUnsigned from the new format so a reconfig cannot leave a stale signedness bit
        // (tt-metal#34499). For non-int8 formats this simply writes 0, which is exactly what a reconfig away
        // from UInt8 needs. FP32 dest is required only when the new register (dst) format drives int8/int32 math,
        // so key the assertion on unpack_dst_format (matching math-side reconfig), not the L1 source: a supported
        // dequant like Int8->Float16_b needs no FP32 dest.
        LLK_ASSERT(
            is_fp32_dest_acc_en || !is_int8_or_int32_format(unpack_dst_format),
            "Reconfiguring unpack to/from Int8/UInt8/Int32 formats requires FP32 Dest mode enabled");
        cfg::write<cfg::Access::TensixCfgUnit, cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0>(
            (unpack_src_format == to_underlying(DataFormat::UInt8)) ? 1 : 0);
    }

    // The 4-bit format fields take the format's low 4 bits: production RMWs the unmasked format under the same
    // byte mask (the RMWCIB mask drops the upper bits); the HAL asserts the value fits the field, so mask it here.
    constexpr auto TileDsc = cfg::Thcon[cfg::Reg0].TileDescriptor;
    cfg::write<cfg::Access::TensixCfgUnit, TileDsc.InDataFormat, cfg::Sec::S0>(masked_data_format(unpack_src_format));

    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S0>(
        ((unpack_src_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3)) ? 1 : 0);

    static_assert(cfg::Thcon[cfg::Reg2].Out_data_format.addr32(cfg::Sec::S0) == THCON_SEC0_REG2_Out_data_format_ADDR32, "Out_data_format word");
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S0>(masked_data_format(unpack_dst_format));
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_A>(), LOWER_HALFWORD(tile_size)); // update gpr which holds tile size A

    // The ch1 (register-side) Z/Y strides are format-derived (datum size), so re-commit them on EVERY format
    // change, independent of dim_stride_target. They drive partial-face unpack addressing (e.g. partial-face
    // matmul, which reads them without setting them), so leaving them stale across a datum-size change (e.g.
    // fp16 -> fp32) mislands faces (tt-llk#1161 follow-up). Per-op brackets (unpack-to-dest, bcastA_B) restore
    // to this same canonical baseline, so re-committing it here is what keeps those restores correct.
    constexpr auto Unp0Ch1 = cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1];
    static_assert(Unp0Ch1.Zstride.addr32(cfg::Sec::S0) == UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32, "Unp0 ch1 Z-stride word");
    cfg::write<cfg::Access::TensixCfgUnit, Unp0Ch1.Zstride, cfg::Sec::S0>(canonical_unpA_z_stride(unpack_dst_format));
    cfg::write<cfg::Access::TensixCfgUnit, Unp0Ch1.Ystride, cfg::Sec::S0>(canonical_unpA_y_stride(unpack_dst_format));

    if constexpr (dim_stride_target == p_dim_stride_target::FACE_ROW_MAJOR)
    {
        // Reprogram tile size + face geometry. issue_stall=false: the STALL_CFG/UNPACK0 above already drained.
        // (tile_size is re-written here; it is the same value stored just above -- a harmless duplicate.)
        _llk_unpack_reconfig_tile_shape_srca_<false /*issue_stall*/>(tile_size, unpack_face_r_dim, unpack_num_faces);
    }
}

/**
 * @brief Reconfigure the operand B (SrcB) data format at runtime.
 *
 * Updates the SrcB tile-descriptor source format, FP8 exponent mode, destination format and
 * tile-size GPR; optionally reprograms dim/stride registers for face-row-major unpacking.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam dim_stride_target: Whether to reprogram dim/stride, values = <IGNORE/FACE_ROW_MAJOR>
 * @tparam skip_int8: Skip re-deriving the SrcB signedness bit from the new format. Leave false (the default)
 *                    unless the caller guarantees the reconfig never crosses an Int8/UInt8/Int32 boundary and
 *                    wants to avoid the extra RMW; skipping it can leave a stale SrcBUnsigned bit (tt-metal#34499).
 * @param unpack_src_format: New source data format of operand B in L1.
 * @param unpack_dst_format: New destination data format operand B is converted to.
 * @param tile_size: New tile size of operand B stored to the tile-size GPR.
 * @param unpack_face_r_dim: Rows per face, used when reprogramming dim/stride.
 * @param unpack_num_faces: Number of faces, valid values = <1, 2, 4>.
 * @note The SrcB-unsigned ALU bit (ALU_FORMAT_SPEC_REG0_SrcBUnsigned), and the math-side INT8 math-enable in
 *       @ref _llk_math_reconfig_data_format_srcb_, are re-derived from the new format by default, so a reconfig
 *       that transitions to OR from an 8-bit-integer / Int8 / Int32 format (e.g. UInt8 -> float) is handled
 *       automatically. Set skip_int8 = true only when the caller guarantees no such boundary is crossed;
 *       skipping leaves the previous unsigned/INT8 state, which the next op would then misinterpret.
 */
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool skip_int8 = false>
inline void _llk_unpack_reconfig_data_format_srcb_impl_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t unpack_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpack_num_faces  = 4)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    LLK_ASSERT(unpack_num_faces == 1 || unpack_num_faces == 2 || unpack_num_faces == 4, "unpack_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpack_src_format), static_cast<DataFormat>(unpack_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker1Idle>();
    if constexpr (!skip_int8)
    {
        // Always re-derive SrcBUnsigned from the new format so a reconfig cannot leave a stale signedness bit
        // (tt-metal#34499). For non-int8 formats this simply writes 0, which is exactly what a reconfig away
        // from UInt8 needs. FP32 dest is required only when the new register (dst) format drives int8/int32 math,
        // so key the assertion on unpack_dst_format (matching math-side reconfig), not the L1 source: a supported
        // dequant like Int8->Float16_b needs no FP32 dest.
        LLK_ASSERT(
            is_fp32_dest_acc_en || !is_int8_or_int32_format(unpack_dst_format),
            "Reconfiguring unpack to/from Int8/UInt8/Int32 formats requires FP32 Dest mode enabled");
        cfg::write<cfg::Access::TensixCfgUnit, cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0>(
            (unpack_src_format == to_underlying(DataFormat::UInt8)) ? 1 : 0);
    }

    // 4-bit format fields: see the srcA impl for the masking.
    constexpr auto TileDsc = cfg::Thcon[cfg::Reg0].TileDescriptor;
    static_assert(TileDsc.InDataFormat.addr32(cfg::Sec::S1) == THCON_SEC1_REG0_TileDescriptor_ADDR32, "SrcB tile descriptor word");
    cfg::write<cfg::Access::TensixCfgUnit, TileDsc.InDataFormat, cfg::Sec::S1>(masked_data_format(unpack_src_format));

    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S1>(
        ((unpack_src_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3)) ? 1 : 0);

    static_assert(cfg::Thcon[cfg::Reg2].Out_data_format.addr32(cfg::Sec::S1) == THCON_SEC1_REG2_Out_data_format_ADDR32, "Out_data_format word");
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S1>(masked_data_format(unpack_dst_format));
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_B>(), LOWER_HALFWORD(tile_size)); // update gpr which holds tile size B

    // Re-commit the format-derived srcB ch1 Z-stride on every format change (see the srcA impl for why).
    constexpr auto Unp1Ch1 = cfg::Unpacker[1].AddrCtrl[cfg::UnpackerReg::Reg1];
    static_assert(Unp1Ch1.Zstride.addr32(cfg::Sec::S0) == UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32, "Unp1 ch1 Z-stride word");
    cfg::write<cfg::Access::TensixCfgUnit, Unp1Ch1.Zstride, cfg::Sec::S0>(datum_size_in_bytes(unpack_dst_format) * FACE_C_DIM * FACE_R_DIM);

    if constexpr (dim_stride_target == p_dim_stride_target::FACE_ROW_MAJOR)
    {
        // Reprogram tile size + face geometry. issue_stall=false: the STALL_CFG/UNPACK1 above already drained.
        _llk_unpack_reconfig_tile_shape_srcb_<false /*issue_stall*/>(tile_size, unpack_face_r_dim, unpack_num_faces);
    }
}

/**
 * @brief Mark SrcA and SrcB as data-valid without unpacking real data.
 *
 * Issues zero-source UNPACR NOPs that set the data-valid flag on both source registers, e.g. to
 * unblock the math thread when an operand is not actually fed from L1.
 *
 * @ref _llk_math_transpose_dest_ on the math thread relies on this to mark SrcB valid for its MOVD2B/MOVB2D sequence.
 */
inline void _llk_unpack_set_srcb_dummy_valid_()
{
    namespace sync = hal::sync;
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();
    // Stall_Clr_Cntrl=1: each publication waits on Unpackers[i].SrcBank -- the bank it
    // actually clears -- rather than MatrixUnit.Src?Bank, a different bank once
    // double-buffering reaches steady state. Carried by the instruction itself, so it
    // cannot go stale the way a separate preceding stall can once SET_DVALID flips the
    // bank pointer. Wormhole uses a preceding stall only because it has no such operand.
    // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the zero-source + set-data-valid flavor
    // (UNP_ZEROSRC, SET_DVALID, Stall_Clr_Cntrl) has no HAL descriptor, so the production encoding is kept.
    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
}

/**
 * @brief Validates L1 addresses and configures unpack base addresses in the configuration registers
 *
 * This helper function validates that both address_a and address_b are within the valid L1 memory region,
 * then configures the appropriate THCON base address registers based on the unpack configuration context.
 *
 * @param address_a: Address for unpacker A (THCON_SEC0).
 * @param address_b: Address for unpacker B (THCON_SEC1).
 * @param cfg: Pointer to configuration registers.
 * @param context: Unpack config context (0 or 1) whose base registers are written; defaults to unp_cfg_context.
 */
inline void _llk_unpack_configure_addresses_(
    const std::uint32_t address_a, const std::uint32_t address_b, volatile std::uint32_t tt_reg_ptr *cfg, const std::uint32_t context = unp_cfg_context)
{
    namespace hcfg = hal::cfg;
    LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");

    // Whole-word MMIO stores into the state-CFG bank `cfg` (the bank get_cfg_pointer() selected), as production.
    constexpr auto BaseA = hcfg::Unpacker[0].Cntx[0].Base;
    constexpr auto BaseB = hcfg::Unpacker[1].Cntx[0].Base;
    static_assert(BaseA.shamt(hcfg::Sec::S0) == 0 && BaseA.width == 32 && BaseB.shamt(hcfg::Sec::S0) == 0 && BaseB.width == 32, "base is a whole word");
    static_assert(hcfg::Unpacker[0].Cntx[1].Base.addr32(hcfg::Sec::S0) == BaseA.addr32(hcfg::Sec::S0) + 1, "context-1 base A follows context 0");
    static_assert(hcfg::Unpacker[1].Cntx[1].Base.addr32(hcfg::Sec::S0) == BaseB.addr32(hcfg::Sec::S0) + 1, "context-1 base B follows context 0");
    LLK_ASSERT(context <= 1, "unpack config context must be 0 or 1");
    // Branch on the context as production: the context-0 path (the common one in loops that reset the context per
    // tile) then stores at fixed offsets with no per-tile address arithmetic between the semaphore poll and the stores.
    if (0 == context)
    {
        cfg[BaseA.addr32(hcfg::Sec::S0)] = address_a;
        cfg[BaseB.addr32(hcfg::Sec::S0)] = address_b;
    }
    else
    {
        cfg[BaseA.addr32(hcfg::Sec::S0) + 1] = address_a;
        cfg[BaseB.addr32(hcfg::Sec::S0) + 1] = address_b;
    }
}

/**
 * @brief Validates L1 address and configures unpack base address for a single unpacker
 *
 * This helper function validates that the address is within the valid L1 memory region,
 * then configures the appropriate THCON_SEC0 base address register based on the unpack configuration context.
 *
 * @param address: Address for unpacker A (THCON_SEC0).
 * @param cfg: Pointer to configuration registers.
 */
inline void _llk_unpack_configure_single_address_(const std::uint32_t address, volatile std::uint32_t tt_reg_ptr *cfg)
{
    namespace hcfg = hal::cfg;
    LLK_ASSERT(is_valid_L1_address(address), "L1 base_address must be in valid L1 memory region");

    // Whole-word MMIO stores into the state-CFG bank `cfg` (the bank get_cfg_pointer() selected), as production.
    constexpr auto BaseCntx0 = hcfg::Unpacker[0].Cntx[0].Base;
    constexpr auto BaseCntx1 = hcfg::Unpacker[0].Cntx[1].Base;
    static_assert(BaseCntx0.shamt(hcfg::Sec::S0) == 0 && BaseCntx0.width == 32, "context-0 base is a whole word");
    static_assert(BaseCntx1.shamt(hcfg::Sec::S0) == 0 && BaseCntx1.width == 32, "context-1 base is a whole word");
    static_assert(BaseCntx0.addr32(hcfg::Sec::S0) == THCON_SEC0_REG3_Base_address_ADDR32, "context-0 base A word");
    static_assert(BaseCntx1.addr32(hcfg::Sec::S0) == THCON_SEC0_REG3_Base_cntx1_address_ADDR32, "context-1 base A word");

    if (0 == unp_cfg_context)
    {
        cfg[BaseCntx0.addr32(hcfg::Sec::S0)] = address;
    }
    else
    {
        cfg[BaseCntx1.addr32(hcfg::Sec::S0)] = address;
    }
}
