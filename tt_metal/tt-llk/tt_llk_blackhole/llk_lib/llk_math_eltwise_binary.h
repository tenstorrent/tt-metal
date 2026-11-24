// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h. Same Tensix instruction sequence as production.

#pragma once

#include <cstdint>
#include <utility>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/replay.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_math.h"

using namespace ckernel;

namespace hal_port
{

/*************************************************************************
 * Common Helpers
 *************************************************************************/

// One ADDR_MOD slot, in the units of ckernel::addr_mod_t.
struct MathAddrMod
{
    std::uint32_t srca_incr = 0;
    std::uint32_t srca_clr  = 0;
    std::uint32_t srcb_incr = 0;
    std::uint32_t srcb_clr  = 0;
    std::uint32_t dest_incr = 0;
    std::uint32_t dest_clr  = 0;
    std::uint32_t dest_cr   = 0;
    std::uint32_t dest_c2cr = 0;
    std::uint32_t fid_incr  = 0;
    std::uint32_t fid_clr   = 0;
};

// addr_mod_t::set(): three SETC16 words (SrcA|SrcB, Dest|Fidelity, Bias), each fully specified.
template <cfg::Sec S, MathAddrMod M>
inline void set_math_addr_mod()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, S, M.srca_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, S, M.srca_clr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, S, M.srcb_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, S, M.srcb_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, S, M.dest_incr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, S, M.dest_cr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, S, M.dest_clr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, S, M.dest_c2cr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, S, M.fid_incr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, S, M.fid_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::Bias].Incr, S, 0>(), cfg::set<cfg::AddrMod[cfg::Bias].Clear, S, 0>());
}

template <BroadcastType bcast_type>
constexpr hal::fpu::SrcBBroadcast srcb_broadcast()
{
    return (bcast_type == BroadcastType::COL)      ? hal::fpu::SrcBBroadcast::Column
           : (bcast_type == BroadcastType::ROW)    ? hal::fpu::SrcBBroadcast::Row
           : (bcast_type == BroadcastType::SCALAR) ? hal::fpu::SrcBBroadcast::Scalar
                                                   : hal::fpu::SrcBBroadcast::None;
}

// p_setrwc::CLR_A / CLR_AB as SETRWC flip / ELW source release.
template <BroadcastType bcast_type>
inline constexpr hal::fpu::SourceRelease CLR_SRC_RELEASE =
    (bcast_type == BroadcastType::COL || bcast_type == BroadcastType::SCALAR) ? hal::fpu::SourceRelease::SrcA : hal::fpu::SourceRelease::Both;

template <BroadcastType bcast_type>
inline constexpr rwc::Src CLR_SRC_FLIP = (bcast_type == BroadcastType::COL || bcast_type == BroadcastType::SCALAR) ? rwc::Src::A : (rwc::Src::A | rwc::Src::B);

/**
 * @brief Program the four address-mod slots used by eltwise binary MOPs (per-row, no-op, fidelity-step, face-step).
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod()
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    static_assert(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    constexpr std::uint32_t fidelity_increment = is_high_fidelity(math_fidelity) ? 1 : 0;
    constexpr std::uint8_t srcb_incr           = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? MAX_FPU_ROWS : 0;

    set_math_addr_mod<cfg::Sec::S0, MathAddrMod {.srca_incr = MAX_FPU_ROWS, .srcb_incr = srcb_incr, .dest_incr = MAX_FPU_ROWS}>();
    set_math_addr_mod<cfg::Sec::S1, MathAddrMod {}>();
    set_math_addr_mod<cfg::Sec::S2, MathAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_cr = 1, .fid_incr = fidelity_increment}>();
    set_math_addr_mod<cfg::Sec::S3, MathAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_incr = MAX_FPU_ROWS, .dest_c2cr = 1, .fid_clr = 1}>();
}

/**
 * @brief Build the encoded FPU instruction (ELWADD/ELWSUB/ELWMUL) for the given binary op type.
 *
 * @param clr_src: Source release (p_setrwc::CLR_* as SourceRelease).
 * @param acc_to_dest: Accumulate result into dest instead of overwriting.
 * @param addr_mod: Address-mod slot the instruction uses.
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type>
inline std::uint32_t eltwise_binary_func(const hal::fpu::SourceRelease clr_src, const std::uint8_t acc_to_dest, const std::uint8_t addr_mod)
{
    static_assert(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    constexpr hal::fpu::ElementwiseOperation operation = (eltwise_binary_type == EltwiseBinaryType::ELWADD)   ? hal::fpu::ElementwiseOperation::Add
                                                         : (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ? hal::fpu::ElementwiseOperation::Subtract
                                                                                                              : hal::fpu::ElementwiseOperation::Multiply;
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = operation,
        .broadcast        = srcb_broadcast<bcast_type>(),
        .dest_write       = acc_to_dest ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = addr_mod,
        .release          = clr_src});
}

namespace detail
{
// INCRWC(0, 8, 8, 8): moves SrcA/SrcB/Dest counters 8 more rows (partial-face padding to 16-row spacing).
inline constexpr std::uint32_t PARTIAL_FACE_INCRWC =
    rwc::MathCounter {.src_a = rwc::Increment(MAX_FPU_ROWS), .src_b = rwc::Increment(MAX_FPU_ROWS), .dest = rwc::Increment(MAX_FPU_ROWS)}.operation();

// Programs the eltwise binary MOP shared by the standard and the dest-reuse configurators (they differ only in the
// outer loop count). The partial-face branch is folded into the loop's second op so the program() is emitted once.
// start_op (ELWADD/ELWSUB only) runs before each face; the MOP skips it when it is a NOP.
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void program_eltwise_binary_mop(
    const std::uint32_t outerloop, const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape, const std::uint32_t start_op = TT_OP_NOP)
{
    using hal::fpu::SourceRelease;
    constexpr bool high_fidelity    = is_high_fidelity(math_fidelity);
    constexpr std::uint8_t addr_mod = ADDR_MOD_0;

    // Inner loop: number of MAX_FPU_ROWS (8-row) operations per face
    // Even if face_r_dim < 16, we still process at least 1 inner loop iteration
    const std::uint8_t innerloop = tensor_shape.face_r_dim > MAX_FPU_ROWS ? (tensor_shape.face_r_dim >> MAX_FPU_ROWS_LOG2) : 1;

    // Scalar and Col broadcast should not Clear B within a MOP - B is cleared outside of MOP
    constexpr auto CLR_SRC = CLR_SRC_RELEASE<bcast_type>;
    // End op: TT_OP_SETRWC(CLR_SRC, CR_AB, 0, 0, 0, SET_AB)
    constexpr std::uint32_t end_op =
        rwc::MathCounter {.src_a = rwc::AdvanceAndReset(0), .src_b = rwc::AdvanceAndReset(0), .flip = CLR_SRC_FLIP<bcast_type>}.operation();

    // Partial faces: INCRWC as the second body op keeps 16-row face spacing. As the body's last op it is also both
    // last-iteration ops, which is what production sets explicitly (or what its two-arg constructor defaults to).
    const std::uint32_t second_op = (tensor_shape.face_r_dim <= MAX_FPU_ROWS) ? PARTIAL_FACE_INCRWC : TT_OP_NOP;

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        const std::uint32_t elw = eltwise_binary_func<eltwise_binary_type, bcast_type>(SourceRelease::None, acc_to_dest, addr_mod);
        hal::mop::program(hal::mop::loop(outerloop, start_op, hal::mop::loop(innerloop, elw, second_op), end_op));
    }
    else
    {
        const std::uint32_t elw = eltwise_binary_func<EltwiseBinaryType::ELWMUL, bcast_type>(SourceRelease::None, 0 /*acc_to_dest*/, addr_mod);
        if constexpr (high_fidelity)
        {
            // No end op: HiFi partial face advancement is handled by runtime INCRWC between face MOP runs,
            // NOT by end_op (which would incorrectly advance dest between fidelity phases)
            hal::mop::program(
                hal::mop::loop(to_underlying(math_fidelity), hal::mop::loop(innerloop, elw))
                    .patch(hal::mop::last_outer, eltwise_binary_func<EltwiseBinaryType::ELWMUL, bcast_type>(CLR_SRC, 0 /*acc_to_dest*/, ADDR_MOD_3))
                    .patch(
                        hal::mop::last_inner, eltwise_binary_func<EltwiseBinaryType::ELWMUL, bcast_type>(SourceRelease::None, 0 /*acc_to_dest*/, ADDR_MOD_2)));
        }
        else
        {
            hal::mop::program(hal::mop::loop(outerloop, hal::mop::loop(innerloop, elw, second_op), end_op));
        }
    }
}
} // namespace detail

/*************************************************************************
 * Eltwise Binary Standard (No Dest Reuse)
 *************************************************************************/

/**
 * @brief Configure MOP for standard eltwise binary operations (no dest reuse).
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_standard(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("eltwise_binary_configure_mop_standard", tensor_shape);
    const std::uint32_t num_faces       = tensor_shape.total_num_faces();
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;

    // Outer loop depends on broadcast type:
    // - COL broadcast: MOP processes num_faces_c_dim faces (one row of faces); runtime calls MOP num_faces_r_dim times
    // - Other broadcasts: MOP processes all num_faces in one call
    const std::uint32_t outerloop = (bcast_type == BroadcastType::COL) ? num_faces_c_dim : num_faces;

    detail::program_eltwise_binary_mop<eltwise_binary_type, bcast_type, math_fidelity>(outerloop, acc_to_dest, tensor_shape);
}

/**
 * @brief Initialize FPU for standard elementwise binary operations (no dest reuse).
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_eltwise_binary_standard_init_", tensor_shape);

    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);

    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    reset_counters_abd_f();
}

/**
 * @brief Perform standard elementwise binary operation (no dest reuse): Output = SrcA [+, -, *] SrcB.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index)
{
    using Mop = hal::mop::Runner<hal::mop::MopTemplate::Template1>;
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    static_assert(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_eltwise_binary_standard_", tensor_shape);
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    constexpr bool high_fidelity        = is_high_fidelity(math_fidelity);

    // Dest counter always jumps by 32x32 tile spacing regardless of actual tile size
    set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // COL broadcast: MOP processes num_faces_c_dim faces (one row of faces); runtime calls it once per row of
            // faces, with CLR_B after each row to allow the next B column to be loaded
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
                Mop::run();
                rwc::set<rwc::MathCounter {.flip = rwc::Src::B}>();
            }
        }
        else
        {
            // NONE/ROW/SCALAR: MOP handles all faces in one call
            Mop::run();
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                rwc::set<rwc::MathCounter {.dest = rwc::Set(0), .flip = rwc::Src::B}>();
            }
        }
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // COL broadcast: MOP processes fidelity phases for one face (HiFi) or all face columns (LoFi)
            const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
            const std::uint32_t fidelity_loop   = high_fidelity ? num_faces_c_dim : 1;
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
#pragma GCC unroll 0
                for (std::uint32_t i = 0; i < fidelity_loop; i++)
                {
                    Mop::run();
                    if constexpr (high_fidelity)
                    {
                        if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
                        {
                            rwc::set<rwc::MathCounter {.dest = rwc::AdvanceAndReset(MAX_FPU_ROWS)}>(); // TTI_INCRWC(CR_D, 8, 0, 0)
                        }
                    }
                    // LoFi: MOP handles face spacing internally via loop_op1, no runtime INCRWC needed
                }
                rwc::set<rwc::MathCounter {.flip = rwc::Src::B}>();
            }
        }
        else
        {
            // NONE/ROW/SCALAR: MOP handles all faces, fidelity requires multiple runs
            const std::uint32_t num_faces     = tensor_shape.total_num_faces();
            const std::uint32_t fidelity_loop = high_fidelity ? num_faces : 1;
#pragma GCC unroll 0
            for (std::uint32_t i = 0; i < fidelity_loop; i++)
            {
                Mop::run();
                if constexpr (high_fidelity)
                {
                    if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
                    {
                        rwc::set<rwc::MathCounter {.dest = rwc::AdvanceAndReset(MAX_FPU_ROWS)}>(); // TTI_INCRWC(CR_D, 8, 0, 0)
                    }
                }
            }
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                rwc::set<rwc::MathCounter {.dest = rwc::Set(0), .flip = rwc::Src::B}>();
            }
        }
    }
    clear_dst_reg_addr();
}

/*************************************************************************
 * Eltwise Binary WITH Dest Reuse
 *************************************************************************/

/**
 * @brief Move one face of the destination register into a source register (SrcA or SrcB) for dest-reuse ops.
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_reuse_dest_as_src()
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        move_d2a_fixed_face<ADDR_MOD_1>();
    }
    else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        move_d2b_fixed_face<ADDR_MOD_1>();
    }
}

namespace detail
{
// ZEROACC one-row mode for rows Rows..., spelled out at compile time like production's unrolled TTI_ZEROACC lines.
template <std::size_t... Rows>
inline void zeroacc_rows(std::index_sequence<Rows...>)
{
    (hal::dst::run<hal::dst::Zero {
         .scope        = hal::dst::ZeroScope::SingleRow,
         .index        = Rows,
         .flags        = hal::dst::ZeroFlagAction::MarkUndefined,
         .width        = hal::dst::DestWidth::Bits16,
         .address_mode = ADDR_MOD_1}>(),
     ...);
}
} // namespace detail

/**
 * @brief Clear one 16-row DEST face using ZEROACC's one-row mode.
 */
inline void zeroacc_face_by_row()
{
    detail::zeroacc_rows(std::make_index_sequence<16> {});
}

// Replay slots holding one face's dest-to-source move: its STALLWAIT and four MOVD2A/MOVD2B.
inline constexpr hal::replay::BufferRange REUSE_DEST_MOVE_REPLAY {0 /*start*/, 5 /*count*/};

/**
 * @brief Configure MOP for eltwise binary operations with dest reuse.
 *
 * ELWADD/ELWSUB: the face's dest-to-source move is recorded in the replay buffer and is the MOP's start op, so one MOP
 * covers every face (a row of faces for COL broadcast). A REPLAY inside the MOP also hides the MOP expander's one-cycle
 * penalty after a MOP that is not followed by another MOP (MOPExpander.md).
 * ELWMUL: MOP outer loop = 1 face, called once per face with the move and ZEROACC between calls.
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity, EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_configure_mop_with_dest_reuse(const std::uint32_t acc_to_dest, const ckernel::TensorShape tensor_shape)
{
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("eltwise_binary_configure_mop_with_dest_reuse", tensor_shape);

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        hal::replay::record<REUSE_DEST_MOVE_REPLAY>([] { eltwise_binary_reuse_dest_as_src<binary_reuse_dest>(); });
        const std::uint32_t outerloop = (bcast_type == BroadcastType::COL) ? tensor_shape.num_faces_c_dim : tensor_shape.total_num_faces();
        detail::program_eltwise_binary_mop<eltwise_binary_type, bcast_type, math_fidelity>(
            outerloop, acc_to_dest, tensor_shape, hal::replay::get_operation<REUSE_DEST_MOVE_REPLAY>());
    }
    else
    {
        constexpr std::uint32_t outerloop = 1;
        detail::program_eltwise_binary_mop<eltwise_binary_type, bcast_type, math_fidelity>(outerloop, acc_to_dest, tensor_shape);
    }
}

/**
 * @brief Initialize FPU for elementwise binary operations with dest reuse.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
inline void _llk_math_eltwise_binary_with_dest_reuse_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Use _llk_math_eltwise_binary_standard_init_ for no dest reuse");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_eltwise_binary_with_dest_reuse_init_", tensor_shape);

    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_with_dest_reuse<eltwise_binary_type, src_b_bcast_type, math_fidelity, binary_reuse_dest>(acc_to_dest, tensor_shape);

    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    reset_counters_abd_f();
}

/**
 * @brief Run the dest-reuse MOP once per face, moving dest into a source register and zeroing the dest face first.
 */
template <bool is_fp32_dest_acc_en, EltwiseBinaryReuseDestType binary_reuse_dest, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_run_with_dest_reuse(
    const std::uint32_t loop_count,
    const std::uint32_t face_offset,
    const bool clear_fp32_dst_acc,
    const std::uint32_t dst_index,
    const std::uint32_t face_r_dim)
{
    using Mop                             = hal::mop::Runner<hal::mop::MopTemplate::Template1>;
    constexpr std::uint32_t ZERO_ACC_MODE = p_zeroacc::CLR_16;
    static_assert(ZERO_ACC_MODE == static_cast<std::uint32_t>(hal::dst::ZeroScope::Face), "CLR_16 is the Face scope");
    // DEST rows one face spans -- what ZEROACC's 16-row mode clears in a single instruction.
    constexpr std::uint32_t DEST_ROWS_PER_FACE = FACE_R_DIM;
    // DEST rows one tile slot spans (set_dst_write_addr<Tile32x32> stride; see production for the full rationale).
    constexpr std::uint32_t DEST_ROWS_PER_TILE = 1u << DstTileSizeLog2[DstTileShape::Tile32x32];
    static_assert(DEST_ROWS_PER_TILE == TILE_NUM_FACES * DEST_ROWS_PER_FACE, "DEST slot must hold a full tile");
    static_assert(MAX_TILES_IN_HALF_DEST * DEST_ROWS_PER_TILE == DEST_REGISTER_HALF_SIZE, "DEST slots tile the bank");
    // Rows in one 16-bit DEST bank. The 32-bit bank is half this, but it is excluded below.
    constexpr std::uint32_t DEST_ROWS_PER_BANK = DEST_REGISTER_HALF_SIZE;

#pragma GCC unroll 0
    for (std::uint32_t n = 0; n < loop_count; n++)
    {
        eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();

        // Clear DEST face-by-face when reusing dest as source
        int clear_fp32                     = is_fp32_dest_acc_en && clear_fp32_dst_acc ? 1 : 0;
        const std::uint32_t tiles_per_bank = clear_fp32 ? MAX_TILES_IN_HALF_DEST >> 1 : MAX_TILES_IN_HALF_DEST;
        const std::uint32_t local_tile     = dst_index & (tiles_per_bank - 1);
        const std::uint32_t face_index     = get_dest_index_in_faces(local_tile, face_offset + n);

        // ZEROACC's 16-row mode takes an absolute 16-row block index within the DEST bank, but Blackhole derives that
        // instruction's bank-select from (dest row offset + block index). Once the sum reaches the bank size the clear
        // lands in the other DEST half; that only happens on the very last face of the last tile of a 16-bit bank,
        // which is cleared with ZEROACC's one-row mode instead (tt-metal#53693; see production for the measurements).
        constexpr bool bank_is_16bit        = !is_fp32_dest_acc_en;
        const std::uint32_t dest_row_offset = (local_tile * DEST_ROWS_PER_TILE) + ((face_offset + n) * DEST_ROWS_PER_FACE);
        const bool crosses_bank             = bank_is_16bit && (dest_row_offset + face_index >= DEST_ROWS_PER_BANK);

        if (crosses_bank)
        {
            zeroacc_face_by_row();
        }
        else
        {
            // TT_ZEROACC(CLR_16, clear_fp32, 0, ADDR_MOD_1, face_index): the ADDR_MOD applies in Face mode too.
            hal::dst::run(hal::dst::Zero {
                .scope        = hal::dst::ZeroScope::Face,
                .index        = face_index,
                .flags        = hal::dst::ZeroFlagAction::MarkUndefined,
                .width        = clear_fp32 ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16,
                .address_mode = ADDR_MOD_1});
        }

        Mop::run();

        if constexpr (is_high_fidelity(math_fidelity))
        {
            if (face_r_dim <= MAX_FPU_ROWS)
            {
                rwc::set<rwc::MathCounter {.dest = rwc::AdvanceAndReset(MAX_FPU_ROWS)}>(); // TTI_INCRWC(CR_D, 8, 0, 0)
            }
        }
    }
}

/**
 * @brief Perform elementwise binary operation with dest reuse: Output = SrcA [+, -, *] SrcB, where one src comes from the dest register.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest>
inline void _llk_math_eltwise_binary_with_dest_reuse_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index, const bool clear_fp32_dst_acc)
{
    using Mop                           = hal::mop::Runner<hal::mop::MopTemplate::Template1>;
    const std::uint32_t num_faces       = tensor_shape.total_num_faces();
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;

    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Use _llk_math_eltwise_binary_standard_ for no dest reuse");
    static_assert(
        math_fidelity == MathFidelity::LoFi || eltwise_binary_type == EltwiseBinaryType::ELWMUL,
        "Math fidelity larger than LoFi only works with Eltwise multiply");
    static_assert(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_eltwise_binary_with_dest_reuse_", tensor_shape);

    // Dest counter always jumps by 32x32 tile spacing regardless of actual tile size
    set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // COL broadcast with dest reuse: the MOP moves and processes one row of faces, then CLR_B
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
                Mop::run();
                rwc::set<rwc::MathCounter {.flip = rwc::Src::B}>();
            }
        }
        else
        {
            // NONE/ROW/SCALAR: the MOP moves and processes every face
            Mop::run();
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                rwc::set<rwc::MathCounter {.dest = rwc::Set(0), .flip = rwc::Src::B}>();
            }
        }
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // COL broadcast with dest reuse and multiply: process num_faces_c_dim faces per row with ZEROACC
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
                // face_offset = face_row * num_faces_c_dim (position in face array)
                const std::uint32_t face_offset = face_row * num_faces_c_dim;
                eltwise_binary_run_with_dest_reuse<is_fp32_dest_acc_en, binary_reuse_dest, math_fidelity>(
                    num_faces_c_dim, face_offset, clear_fp32_dst_acc, dst_index, tensor_shape.face_r_dim);
                rwc::set<rwc::MathCounter {.flip = rwc::Src::B}>();
            }
        }
        else
        {
            // NONE/ROW/SCALAR: process all faces with ZEROACC
            eltwise_binary_run_with_dest_reuse<is_fp32_dest_acc_en, binary_reuse_dest, math_fidelity>(
                num_faces, 0 /*face_offset*/, clear_fp32_dst_acc, dst_index, tensor_shape.face_r_dim);

            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                rwc::set<rwc::MathCounter {.dest = rwc::Set(0), .flip = rwc::Src::B}>();
            }
        }
    }
    clear_dst_reg_addr();
}

} // namespace hal_port

/*************************************************************************
 * Public API
 *************************************************************************/

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t acc_to_dest)
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE)
    {
        hal_port::_llk_math_eltwise_binary_standard_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity>(tensor_shape, acc_to_dest);
    }
    else
    {
        hal_port::_llk_math_eltwise_binary_with_dest_reuse_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity, binary_reuse_dest>(
            tensor_shape, acc_to_dest);
    }

    // Re-establish the operand-driven DEFAULT Src zero-substitution flag (a preceding copy_init/datacopy may have left PRESERVE).
    hal_port::_configure_default_zero_flag_state_();
}

/**
 * @brief Perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_(const ckernel::TensorShape tensor_shape, std::uint32_t dst_index, const bool clear_fp32_dst_acc = false)
{
    // Zero-flag leak guard (fires only under LLK asserts); see production.
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "eltwise_binary: Src zero-substitution flag is not in DEFAULT state — a prior op (copy_init/datacopy) leaked "
        "PRESERVE into ELWADD/ELWMUL/ELWSUB without a format-changing reconfig; denormal Src results will differ");

    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE)
    {
        hal_port::_llk_math_eltwise_binary_standard_<eltwise_binary_type, src_b_bcast_type, Dst, is_fp32_dest_acc_en, math_fidelity>(tensor_shape, dst_index);
    }
    else
    {
        hal_port::_llk_math_eltwise_binary_with_dest_reuse_<eltwise_binary_type, src_b_bcast_type, Dst, is_fp32_dest_acc_en, math_fidelity, binary_reuse_dest>(
            tensor_shape, dst_index, clear_fp32_dst_acc);
    }
}

/**
 * @brief Uninitialize/cleanup after elementwise binary operations; currently a no-op since all state is transient.
 */
inline void _llk_math_eltwise_binary_uninit_()
{
    // No state to restore - all states are transient or default
}
