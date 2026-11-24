// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h. Same instruction sequence and access kinds as production,
// expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h" // constants and types only (p_gpr_unpack, ...)
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/nop.h"
#include "hal/replay.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

namespace hal_port
{

// One whole-tile (or one half-face of a partial-face tile) UNPACR of a matmul operand, configuration context 0 with the
// thread override: TT_OP_UNPACR(Src, addr_mode, 0, 0, 0, 1 /*OvrdThreadId*/, set_dvalid, RAREFYB_DISABLE, 0, 0, 0, 0, 1).
constexpr hal::unpack::DataTransfer matmul_unpack(const hal::unpack::Engine engine, const std::uint8_t z_increment, const hal::unpack::SourceHandoff handoff)
{
    return {
        .engine     = engine,
        .increments = {.channel0 = {.z = z_increment}, .channel1 = {.z = z_increment}},
        .handoff    = handoff,
    };
}

template <hal::unpack::Engine E>
struct MatmulUnpackOps
{
    // Whole tile: TT_OP_UNPACR(Src, 0, ..., 1 /*Set Dvalid*/, ...)
    static constexpr std::uint32_t tile = hal::unpack::get_operation<matmul_unpack(E, 0, hal::unpack::SourceHandoff::FlipAndSetDataValid)>();
    // Partial face, first half: TT_OP_UNPACR(Src, 0b00010001, ..., 0 /*Set Dvalid*/, ...)
    static constexpr std::uint32_t face_keep = hal::unpack::get_operation<matmul_unpack(E, 1, hal::unpack::SourceHandoff::Keep)>();
    // Partial face, second half: TT_OP_UNPACR(Src, 0b00010001, ..., 1 /*Set Dvalid*/, ...)
    static constexpr std::uint32_t face_valid = hal::unpack::get_operation<matmul_unpack(E, 1, hal::unpack::SourceHandoff::FlipAndSetDataValid)>();
};

using MatmulUnpackA = MatmulUnpackOps<hal::unpack::Engine::Unpacker0>; // SrcA
using MatmulUnpackB = MatmulUnpackOps<hal::unpack::Engine::Unpacker1>; // SrcB

static_assert(MatmulUnpackA::tile == TT_OP_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
static_assert(MatmulUnpackA::face_keep == TT_OP_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
static_assert(MatmulUnpackA::face_valid == TT_OP_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
static_assert(MatmulUnpackB::tile == TT_OP_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
static_assert(MatmulUnpackB::face_keep == TT_OP_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
static_assert(MatmulUnpackB::face_valid == TT_OP_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));

// TTI_SETADCZW(UNP_A or UNP_B, 0, 0, 0, 0, 0b0101): channel-0 and channel-1 Z of one unpacker back to 0.
template <adc::Client C>
inline constexpr adc::AddrCounter matmul_z_reset = adc::AddrCounter {.channel0 = {.z = 0}, .channel1 = {.z = 0}}.for_clients<C>();
static_assert(matmul_z_reset<adc::Client::Unpacker0>.operation() == TT_OP_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0101));
static_assert(matmul_z_reset<adc::Client::Unpacker1>.operation() == TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101));

// One partial-face unpack: zero the bank, unpack two half faces (data valid on the second), rewind Z.
template <hal::unpack::Engine E, adc::Client C>
inline __attribute__((always_inline)) void matmul_unpack_partial_face()
{
    // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the zero-source flavor (UNP_ZEROSRC, WAIT_LIKE_UNPACR)
    // has no HAL descriptor, so the production encoding is kept.
    TTI_UNPACR_NOP(static_cast<std::uint32_t>(E), 0, 0, 0 /*Set Dvalid*/, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_INSN(MatmulUnpackOps<E>::face_keep);
    TTI_INSN(MatmulUnpackOps<E>::face_valid);
    adc::set<matmul_z_reset<C>>();
}

// One replay-buffer step of the streamed operand: unpack, then (unless kernel broadcast pins the tile) advance the base
// address of configuration context Cntx by the tile size held in GPR TileSizeGpr.
template <hal::unpack::Engine E, adc::Client C, std::uint32_t Unpacker, std::uint32_t Cntx, std::uint32_t TileSizeGpr, bool KernelBroadcast>
inline __attribute__((always_inline)) void matmul_unpack_and_advance(const bool partial_face)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    if (partial_face)
    {
        matmul_unpack_partial_face<E, C>();
    }
    else
    {
        TTI_INSN(MatmulUnpackOps<E>::tile);
    }
    if constexpr (KernelBroadcast)
    {
        hal::nop::thread();
        hal::nop::thread();
        hal::nop::thread();
        hal::nop::thread();
    }
    else
    {
        // TTI_RDCFG(TMP0, base); TTI_ADDDMAREG(0, TMP0, TMP0, tile size); TTI_STALLWAIT(STALL_CFG, THCON); TTI_WRCFG(TMP0, 0, base)
        cfg::read<cfg::Access::TensixCfgUnit, cfg::Unpacker[Unpacker].Cntx[Cntx].Base, cfg::Sec::S0>(hal::gpr<p_gpr_unpack::TMP0>());
        hal::scalar::add(hal::gpr<p_gpr_unpack::TMP0>(), hal::gpr<TileSizeGpr>(), hal::gpr<p_gpr_unpack::TMP0>());
        sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
        cfg::write<
            cfg::Access::TensixCfgUnit,
            cfg::Unpacker[Unpacker].Cntx[Cntx].Base,
            cfg::Sec::S0,
            cfg::GprTransferSize::Bits32,
            cfg::WrcfgCompletion::Deferred>(hal::gpr<p_gpr_unpack::TMP0>());
    }
    // Added to ensure WRCFG instruction has finished, since it takes 2 cycles.
    hal::nop::thread();
}

static_assert(hal::cfg::Unpacker[0].Cntx[0].Base.addr32(hal::cfg::Sec::S0) == THCON_SEC0_REG3_Base_address_ADDR32);
static_assert(hal::cfg::Unpacker[0].Cntx[1].Base.addr32(hal::cfg::Sec::S0) == THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
static_assert(hal::cfg::Unpacker[1].Cntx[0].Base.addr32(hal::cfg::Sec::S0) == THCON_SEC1_REG3_Base_address_ADDR32);
static_assert(hal::cfg::Unpacker[1].Cntx[1].Base.addr32(hal::cfg::Sec::S0) == THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
static_assert(
    hal::scalar::Add(hal::gpr<p_gpr_unpack::TMP0>(), hal::gpr<p_gpr_unpack::TILE_SIZE_A>(), hal::gpr<p_gpr_unpack::TMP0>()).operation() ==
    TT_OP_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A));
static_assert(
    hal::scalar::MultiplyLowU16(hal::gpr<p_gpr_unpack::TMP_LO>(), hal::gpr<p_gpr_unpack::KT_DIM>(), hal::gpr<p_gpr_unpack::TILE_SIZE_B>()).operation() ==
    TT_OP_MULDMAREG(0, p_gpr_unpack::TMP_LO, p_gpr_unpack::TILE_SIZE_B, p_gpr_unpack::KT_DIM));

} // namespace hal_port

/**
 * @brief Set the SrcA address step used to stream matmul in1 columns.
 *
 * tile_size is in unpacker L1 address units (16-byte words), not bytes. Only the
 * column-streaming path (ct_dim >= rt_dim, no in1 kernel broadcast) uses this step.
 * A stride of one restores the normal contiguous-tile step.
 */
inline void _llk_unpack_AB_matmul_set_in1_column_stride_(const std::uint32_t tile_size, const std::uint32_t stride_tiles)
{
    LLK_ASSERT(tile_size > 0 && tile_size <= 0xffff, "Matmul tile size must fit the SrcA address-step register");
    LLK_ASSERT(stride_tiles > 0 && stride_tiles <= 0xffff / tile_size, "Matmul column stride must fit the SrcA address-step register");
    // TT_SETDMAREG(0, LOWER_HALFWORD(tile_size * stride_tiles), 0, LO_16(TILE_SIZE_A))
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_A>(), LOWER_HALFWORD(tile_size * stride_tiles));
}

/**
 * @brief Program the unpacker MOP/replay buffer for a matmul operand unpack.
 *
 * Builds a replay buffer that unpacks the streamed (non-reused) operand and advances its L1 base
 * address by one tile each step. Which operand is reused versus streamed is chosen by comparing
 * ct_dim and rt_dim (reuse_a = ct_dim >= rt_dim); operand A maps to SrcB and operand B to SrcA.
 * The address bump is suppressed under kernel broadcast.
 *
 * @tparam kernel_broadcast_a: Tile count to wrap operand A around for kernel broadcast (0 = disabled).
 * @tparam kernel_broadcast_b: Tile count to wrap operand B around for kernel broadcast (0 = disabled).
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param unpA_partial_face: Whether operand A is unpacked face-by-face (partial faces).
 * @param unpB_partial_face: Whether operand B is unpacked face-by-face (partial faces).
 */
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
inline void _llk_unpack_AB_matmul_mop_config_(
    const std::uint32_t ct_dim, const std::uint32_t rt_dim, const bool unpA_partial_face, const bool unpB_partial_face)
{
    // in0/inA - loaded to SrcB
    // in1/inB - loaded to SrcA

    const bool reuse_a                      = ct_dim >= rt_dim;
    const std::uint32_t replay_buf_prog_len = (reuse_a && unpA_partial_face) ? 18 : ((!reuse_a && unpB_partial_face) ? 18 : 12);
    const std::uint32_t replay_buf_run_len  = replay_buf_prog_len / 2;

    const hal::replay::BufferRange replay_range {0 /*start*/, replay_buf_prog_len};
    if (reuse_a)
    {
        static_assert(kernel_broadcast_b <= 1, "kernel_broadcast>1 on matmul input 1 is not supported with reuse enabled");
        // Stream in1 (SrcA): one step per configuration context, each advancing that context's base address.
        hal::replay::record(
            replay_range,
            [unpA_partial_face]
            {
                hal_port::matmul_unpack_and_advance<
                    hal::unpack::Engine::Unpacker0,
                    hal_port::adc::Client::Unpacker0,
                    0 /*Unpacker*/,
                    0 /*Cntx*/,
                    p_gpr_unpack::TILE_SIZE_A,
                    kernel_broadcast_b == 1>(unpA_partial_face);
                hal_port::matmul_unpack_and_advance<
                    hal::unpack::Engine::Unpacker0,
                    hal_port::adc::Client::Unpacker0,
                    0 /*Unpacker*/,
                    1 /*Cntx*/,
                    p_gpr_unpack::TILE_SIZE_A,
                    kernel_broadcast_b == 1>(unpA_partial_face);
            });
    }
    else
    {
        static_assert(kernel_broadcast_a <= 1, "kernel_broadcast>1 on matmul input 0 is not supported with reuse enabled");
        // Stream in0 (SrcB), stepping by the kt_dim-scaled tile size in TMP_LO (see _llk_unpack_AB_matmul_).
        hal::replay::record(
            replay_range,
            [unpB_partial_face]
            {
                hal_port::matmul_unpack_and_advance<
                    hal::unpack::Engine::Unpacker1,
                    hal_port::adc::Client::Unpacker1,
                    1 /*Unpacker*/,
                    0 /*Cntx*/,
                    p_gpr_unpack::TMP_LO,
                    kernel_broadcast_a == 1>(unpB_partial_face);
                hal_port::matmul_unpack_and_advance<
                    hal::unpack::Engine::Unpacker1,
                    hal_port::adc::Client::Unpacker1,
                    1 /*Unpacker*/,
                    1 /*Cntx*/,
                    p_gpr_unpack::TMP_LO,
                    kernel_broadcast_a == 1>(unpB_partial_face);
            });
    }

    // Template 0, A0 only (no B, no A1-A3): the context-0 half of the replay buffer when the mask bit is clear, the
    // context-1 half when it is set. Production also stores 0 into the unused B, A1-A3 and SkipB slots; the HAL stores
    // only the slots the flags enable.
    using namespace hal::mop;
    program(foreach_mask_bit(
        if_clear(hal::replay::get_operation(hal::replay::BufferRange {0 /*start*/, replay_buf_run_len})),                     // runs when context is 0
        otherwise(hal::replay::get_operation(hal::replay::BufferRange {replay_buf_run_len /*start*/, replay_buf_run_len})))); // runs when context is 1
}

/**
 * @brief Initialize the unpacker for a matmul (A x B) operation.
 *
 * Re-enables within-face transpose if needed, programs per-unpacker datum counts (full-tile or
 * face-by-face for partial faces), stashes kt_dim into a GPR for tile-size scaling, and programs
 * the matmul MOP.
 *
 * @tparam kernel_broadcast_a: Tile count to wrap operand A around for kernel broadcast (0 = disabled).
 * @tparam kernel_broadcast_b: Tile count to wrap operand B around for kernel broadcast (0 = disabled).
 * @param transpose: Nonzero to enable within-face (16x16) transpose for SrcA.
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param kt_dim: Number of tiles along the contraction (K) dimension.
 * @param unpA_face_r_dim: Rows per face for operand A.
 * @param unpB_face_r_dim: Rows per face for operand B.
 * @param unpA_num_faces: Number of faces for operand A, valid values = <1, 2, 4>.
 * @param unpB_num_faces: Number of faces for operand B, valid values = <1, 2, 4>.
 * @param unpA_partial_face: Whether operand A is unpacked face-by-face (partial faces).
 * @param unpB_partial_face: Whether operand B is unpacked face-by-face (partial faces).
 * @note Call @ref _llk_unpack_AB_matmul_uninit_ to restore the modified datum-count state.
 * @ref _llk_unpack_AB_matmul_ is the matching execute call.
 * @ref _llk_math_matmul_init_ is the matching init on the math thread (consumes SrcA/SrcB).
 */
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
__attribute__((always_inline)) inline void _llk_unpack_AB_matmul_init_(
    const std::uint32_t transpose       = 0,
    const std::uint32_t ct_dim          = 1,
    const std::uint32_t rt_dim          = 1,
    const std::uint32_t kt_dim          = 1,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpA_num_faces  = 4,
    const std::uint32_t unpB_num_faces  = 4,
    const bool unpA_partial_face        = false,
    const bool unpB_partial_face        = false)
{
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");
    // 16x16 matmul not supported - no dedicated math path; falls to 32x32 default which is incorrect for < 4 faces
    LLK_ASSERT(!(unpA_num_faces == 1 && unpB_num_faces == 1), "16x16 by 16x16 matmul is not supported");

    // also turn on within_face_16x16_transpose if it was turned off by datacopy at runtime
    // on WH, the unpacker performs both transpose of faces as well as transpose each face.
    // the former is configured in mop, the latter is configured in cfg register in hw_configure
    // in large matmul, datacopy will disable the transpose of faces, so we need it turn it back on for matmul.
    namespace cfg = hal::cfg;
    namespace adc = hal::counters::adc;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(transpose);

    // TTI_SETADCZW(UNP_A | UNP_B, 0, 0, 0, 0, 0b1111): Z and W of both channels of both unpackers
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();

    if (unpA_partial_face)
    {
        // Do face by face unpacking. Need to program correct face dim
        // to compute address of the next face
        hal_port::config_unpacker_x_end<p_setadc::UNP_A>(unpA_face_r_dim);
    }
    else
    {
        const std::uint32_t unpA_x_end = unpA_num_faces * unpA_face_r_dim * FACE_C_DIM - 1;
        // TT_SETADCXX(UNP_A, unpA_x_end, 0x0): X start on channel 0, X end on channel 1
        adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = unpA_x_end}}.for_clients<adc::Client::Unpacker0>());
    }

    if (unpB_partial_face)
    {
        // Do face by face unpacking. Need to program correct face dim
        // to compute address of the next face
        hal_port::config_unpacker_x_end<p_setadc::UNP_B>(unpB_face_r_dim);
    }
    else
    {
        // Do full tile unpacking. No need to program face dim
        // as address counter pointing to the face is not incremented
        const std::uint32_t unpB_x_end = unpB_num_faces * unpB_face_r_dim * FACE_C_DIM - 1;
        // TT_SETADCXX(UNP_B, unpB_x_end, 0x0)
        adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = unpB_x_end}}.for_clients<adc::Client::Unpacker1>());
    }

    // TT_SETDMAREG(0, LOWER_HALFWORD(kt_dim), 0, LO_16(KT_DIM)): store kt_dim to gpr for scaling tile size
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::KT_DIM>(), LOWER_HALFWORD(kt_dim));

    _llk_unpack_AB_matmul_mop_config_<kernel_broadcast_a, kernel_broadcast_b>(ct_dim, rt_dim, unpA_partial_face, unpB_partial_face);
}

/**
 * @brief No-op after a matmul operation.
 *
 * x-start/x-end is transient and reprogrammed by the next operation's init (see tt-llk#1036), so
 * there is nothing to restore here.
 *
 * @note Call @ref _llk_unpack_AB_matmul_init_ before this function.
 */
inline void _llk_unpack_AB_matmul_uninit_()
{
}

/**
 * @brief Unpack the operand tiles for a matmul (A x B) into SrcA and SrcB.
 *
 * Iterates over the reused dimension, computing per-tile L1 addresses (with optional kernel-
 * broadcast wraparound and kt_dim striding), and unpacks operand A to SrcB / operand B to SrcA
 * for each step while synchronizing through the unpack semaphore and config-context switching.
 *
 * @tparam kernel_broadcast_a: Tile count to wrap operand A around for kernel broadcast (0 = disabled).
 * @tparam kernel_broadcast_b: Tile count to wrap operand B around for kernel broadcast (0 = disabled).
 * @param base_address_a: L1 base address of operand A's tile buffer.
 * @param base_address_b: L1 base address of operand B's tile buffer.
 * @param tile_index_a: Starting tile index into operand A.
 * @param tile_index_b: Starting tile index into operand B.
 * @param tile_size_a: Size of one operand A tile, used to compute per-tile offsets.
 * @param tile_size_b: Size of one operand B tile, used to compute per-tile offsets.
 * @param unpA_partial_face: Whether operand A is unpacked face-by-face (partial faces).
 * @param unpB_partial_face: Whether operand B is unpacked face-by-face (partial faces).
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param kt_dim: Number of tiles along the contraction (K) dimension.
 * @note Call @ref _llk_unpack_AB_matmul_init_ with matching template args before this function, and
 *       @ref _llk_unpack_AB_matmul_uninit_ after it to restore modified state.
 * @ref _llk_math_matmul_ on the math thread consumes the SrcA/SrcB tiles unpacked here.
 */
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
inline void _llk_unpack_AB_matmul_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const bool unpA_partial_face = false,
    const bool unpB_partial_face = false,
    std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim   = 1,
    const std::uint32_t kt_dim   = 1)
{
    // In0/InA -> srcB (supports partial face)
    // In1/InB -> srcA

    namespace sync  = hal::sync;
    using MopRunner = hal::mop::Runner<hal::mop::MopTemplate::Template0>;

    volatile std::uint32_t tt_reg_ptr *cfg = hal::cfg::detail::state_cfg_bank(); // get pointer to registers for current state ID

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    if (!reuse_a)
    {
        // TTI_MULDMAREG(0, TMP_LO, TILE_SIZE_B, KT_DIM): in0 step = tile size * kt_dim
        hal::scalar::multiply_low_u16(hal::gpr<p_gpr_unpack::TMP_LO>(), hal::gpr<p_gpr_unpack::KT_DIM>(), hal::gpr<p_gpr_unpack::TILE_SIZE_B>());
    }

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        std::uint32_t offset_address_a = tile_size_a * (tile_index_a + (reuse_a ? (t * kt_dim) : (0)));
        std::uint32_t offset_address_b = tile_size_b * (tile_index_b + (reuse_a ? (0) : (t)));
        if constexpr (kernel_broadcast_a > 0)
        {
            offset_address_a = tile_size_a * ((tile_index_a + (reuse_a ? ((t * kt_dim)) : (0))) % kernel_broadcast_a);
        }
        if constexpr (kernel_broadcast_b > 0)
        {
            offset_address_b = tile_size_b * ((tile_index_b + (reuse_a ? (0) : (t))) % kernel_broadcast_b);
        }

        std::uint32_t address_a = base_address_a + offset_address_a;
        std::uint32_t address_b = base_address_b + offset_address_b;

        // Wait for free context
        hal_port::wait_for_next_context(2);

        // Validate and configure addresses (note: address_b goes to SEC0, address_a to SEC1 for matmul)
        _llk_unpack_configure_addresses_(address_b, address_a, cfg);

        sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>(); // Trisc::SEMPOST for context acquire

        // Stall unpacker until pending CFG writes from Trisc have completed
        sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

        if (reuse_a)
        {
            if (unpB_partial_face)
            {
                // Do face by face unpacking
                hal_port::matmul_unpack_partial_face<hal::unpack::Engine::Unpacker1, hal_port::adc::Client::Unpacker1>();
            }
            else
            {
                TTI_INSN(hal_port::MatmulUnpackB::tile);
            }
        }
        else
        {
            if (unpA_partial_face)
            {
                // Do face by face unpacking
                hal_port::matmul_unpack_partial_face<hal::unpack::Engine::Unpacker0, hal_port::adc::Client::Unpacker0>();
            }
            else
            {
                TTI_INSN(hal_port::MatmulUnpackA::tile);
            }
        }

        // Run the MOP: TT_MOP(0, count - 1, context ? 0xff : 0); no MOP_CFG, as production
        MopRunner::run<hal::mop::MaskHigh::Preset>((reuse_a ? ct_dim : rt_dim), unp_cfg_context == 0 ? 0 : 0xff);

        // T6::SEMGET for context release
        sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

        // Switch unpacker config context
        hal_port::switch_config_context(unp_cfg_context);
    }
}
