// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"

// [#55076 / 0x19 DIAGNOSTIC -- DO NOT UPSTREAM] See the note in llk_lib/llk_math_matmul.h.
// __has_include guard so the standalone tt-llk test build is unaffected; WAYPOINT is already a no-op
// unless WATCHER_ENABLED.
#if __has_include("api/debug/waypoint.h")
#include "api/debug/waypoint.h"
#endif
#ifndef WAYPOINT
#define WAYPOINT(x)
#endif

namespace ckernel::math
{

// Number of rows for MATH functions
constexpr static std::uint32_t ELTWISE_MATH_ROWS = MATH_ROWS; // 8 for quasar, 4 for quasar automotive
constexpr static std::uint32_t MOVE_MATH_ROWS[3] = {8, 4, 1};
constexpr static unsigned int SFP_ROWS           = 2;

// SFPU register-file base addresses: dest region vs SrcS (used by SFPU load/store)
constexpr static unsigned int SFPU_DEST_BASE_ADDR = 0x0;
constexpr static unsigned int SFPU_SRCS_BASE_ADDR = 0x400;

// Struct for the ALU addresses
constexpr std::uint32_t NUM_WORDS_ALU_FORMAT = 3;

typedef struct
{
    // word 0
    std::uint32_t ALU_FORMAT_SPEC_REG_SrcA_val        : 8;
    std::uint32_t ALU_FORMAT_SPEC_REG_SrcA_override   : 1;
    std::uint32_t ALU_FORMAT_SPEC_REG_SrcB_val        : 8;
    std::uint32_t ALU_FORMAT_SPEC_REG_SrcB_override   : 1;
    std::uint32_t ALU_FORMAT_SPEC_REG_Dstacc_val      : 8;
    std::uint32_t ALU_FORMAT_SPEC_REG_Dstacc_override : 1;
    std::uint32_t EMPTY0                              : 5;
    // word 1
    std::uint32_t ALU_ROUNDING_MODE_Fpu_srnd_en : 1;
    std::uint32_t UNUSED0                       : 2;
    std::uint32_t ALU_ROUNDING_MODE_Padding     : 10;
    std::uint32_t ALU_ROUNDING_MODE_GS_LF       : 1;
    std::uint32_t ALU_ROUNDING_MODE_Bfp8_HF     : 1;
    std::uint32_t ALU_FORMAT_SPEC_REG0_SrcA     : 8;
    std::uint32_t ALU_FORMAT_SPEC_REG1_SrcB     : 8;
    std::uint32_t EMPTY1                        : 1;
    // word 2
    std::uint32_t ALU_FORMAT_SPEC_REG2_Dstacc    : 8;
    std::uint32_t ALU_ACC_CTRL_Fp32_enabled      : 1;
    std::uint32_t ALU_ACC_CTRL_SFPU_Fp32_enabled : 1;
    std::uint32_t ALU_ACC_CTRL_INT8_math_enabled : 1;
    std::uint32_t UNUSED1                        : 21;
} alu_config_t;

static_assert(sizeof(alu_config_t) == NUM_WORDS_ALU_FORMAT * sizeof(std::uint32_t));

typedef union
{
    std::uint32_t val[NUM_WORDS_ALU_FORMAT];
    alu_config_t f;
} alu_config_u;

// List of possible data format config states
enum class DataFormatConfigSet : std::uint8_t
{
    UNCONFIGURED         = 0,
    DEFAULT              = 1,
    MOV_OPS_EXPLICIT_FMT = 2
};

// /**
// * @brief Helper function to calculate log2,
// * only works for 32 bit unsigned inputs
// * @param val: Input value to log2 operation
// */
// inline uint32_t trisc_log2(const uint32_t val) {
//     return 31 - __builtin_clz(val);
// }

/**
 * @brief Increments given counters
 * @tparam: SRCA_INCR: SrcA increment values = 0 - 15
 * @tparam: SRCB_INCR: SrcA increment values = 0 - 15
 * @tparam: SRCD_INCR: SrcA increment values = 0 - 15
 * @tparam: CR_INCR: SrcA increment values = 0 - 63
 */
template <std::uint32_t SRCA_INCR, std::uint32_t SRCB_INCR, std::uint32_t DEST_INCR, std::uint32_t CR_INCR>
inline void _incr_counters_()
{
    static_assert(SRCA_INCR < 32, "Value exceeds RWC_A width of 5 bits");
    static_assert(SRCB_INCR < 32, "Value exceeds RWC_B width of 5 bits");
    static_assert(DEST_INCR < 256, "Value exceeds RWC_D width of 8 bits");
    static_assert(CR_INCR < 64, "Value exceeds RWC_CR width of 6 bits");
    TTI_INCRWC(CR_INCR, SRCA_INCR, SRCB_INCR, DEST_INCR);
}

// TODO (RT): Is there now an alternative to this?
inline void _sfpu_load_config32_(const std::uint32_t dest, const std::uint32_t upper16, const std::uint32_t lower16)
{
    // registers 11 through 14 are programmable "constants" which are shared across all 4 rows
    // They are updated only through the CONFIG path, which uses LREG[0] first and then copies it to the desired register location
    TTI_SFPLOADI(p_sfpu::LREG0, 10, lower16); // insmod == A will write the lower bits, and not affect the upper bits;
    TTI_SFPLOADI(p_sfpu::LREG0, 8, upper16);  // insmod == 8 will write the upper bits, and not affect the lower bits;
    TTI_SFPCONFIG(0, dest, 0);
}

/**
 * @brief Initializes the programmable registers for the SFPU
 */
inline void _init_sfpu_config_reg_()
{
    TTI_SFPCONFIG(0, 0xF, 1);
    // Quasar simulator doesn't apply the SFPU const-lreg reset default at boot.
    // Reload programmable constant LREG11 = -1.0 (its RTL reset default) each launch: config_dest=0xB,
    // instr_mod1[0]=1 loads the default. sfpi materializes -1.0 and subtract-based float compares via LREG11.
    TTI_SFPCONFIG(0, 0xB, 1);
}

/**
 * @brief Reset given counters to 0
 * @tparam: SETRWC: which counter to reset, values = p_setrwc::[SET_A, SET_B, SET_D, SET_F]
 */
template <std::uint32_t SETRWC>
inline void _reset_counters_()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, SETRWC);
}

/**
 * @brief Inc dest counter using carriage return (why use the CR?)
 * @tparam NUM_ROWS: number of 16 datum rows to increment dest by, value must be <=255
 */
template <std::uint32_t NUM_ROWS>
inline void _inc_dst_addr_()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, NUM_ROWS, p_setrwc::SET_D);
}

/**
 * @brief Sets destination register base address depending on tile idx
 * @param tile_idx: Tile index in the dest reg
 * 16bit dest reg data format -> tile_idx = 0 - 7
 * 32bit dest reg data format -> tile_idx = 0 - 3
 */
template <ckernel::trisc::DstTileShape TILE_SHAPE>
inline void _set_dst_write_addr_(const std::uint32_t tile_index)
{
    // [#55076 / 0x19] Drain math + SFPU before repointing the DEST section base.
    //
    // This writes DEST_TARGET_REG_CFG_MATH_SEC<TRISC_ID>_Offset, which is the base every in-flight
    // FPU/SFPU dest access is resolved against. Moving it while math is still in flight retargets
    // those accesses. The LLK already treats that as requiring a guard everywhere else it touches
    // this class of CFG register:
    //   llk_math_common.h:357  _llk_math_dest_section_done_  -> STALLWAIT(STALL_CFG, 0, MATH, WAIT_SFPU)
    //                                                           before the SAME _set_dest_section_base_
    //   llk_math_common.h:175  _configure_alu_formats_       -> STALLWAIT(STALL_CFG, 0, WAIT_SFPU, MATH)
    // but all seven _set_dst_write_addr_ callers had none -- including both matmul paths
    // (llk_math_matmul.h:512 _llk_math_matmul_tile_, :540 _llk_math_matmul_block_).
    //
    // Evidence this is the 0x19 wedge: MATH's last waypoint is MB0, written on entry to
    // _llk_math_matmul_block_ immediately BEFORE this call, and MB1 (immediately after) never lands.
    // So MATH stalls inside this function, before the MVMUL MOP is even launched. It also explains
    // the two properties no DEST-semaphore theory could: it needs >= 3 subblocks (calls 1-2 have no
    // prior math in flight to collide with; by call 3 the previous subblock's MVMULs still are), and
    // DPRINT latency masks it (the extra delay lets the math pipe drain before this cfg write).
    //
    // Arg form note: on Quasar the wait conditions are 5-bit enum INDICES in separate slots, never a
    // bitmask -- see qsr_x19_rtl_findings.md §4i. Do not write `p_stall::MATH | p_stall::WAIT_SFPU`.
    //
    // RESULT 2026-09-03 10:55: tested ON, hang unchanged (MATH still last-waypoint MB0). It cannot
    // help: the guard waits on p_stall::MATH, and "math never goes idle" is the very thing that is
    // stuck -- so it blocks in the same place the unguarded cfg store did. Off by default.
    constexpr bool kGuardDestSectionBaseWrite = false;
    if constexpr (kGuardDestSectionBaseWrite)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::MATH, p_stall::WAIT_SFPU);
    }

    // [#55076 diagnostic] Split the three steps between MB0 and MB1. dest_register_offset is
    // thread_local (TRISC local data RAM), so its read cannot time out -- SDW1 being last would
    // confirm the CFG store is the blocking access, which is the current conclusion.
    WAYPOINT("SDW0");
    constexpr std::uint32_t tile_shape_idx = ckernel::trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t dst_index          = (tile_index << tile_shape_idx) + ckernel::trisc::_get_dest_buffer_base_();
    WAYPOINT("SDW1");
    ckernel::trisc::_set_dest_section_base_<TRISC_ID>(dst_index);
    WAYPOINT("SDW2");
}

/**
 * @brief Computes the tile-shape index (a log2-style shift exponent derived from
 *        the number of rows per tile) and stores it in GPR TEMP0 for later reuse
 *        by @ref _set_dst_write_addr_by_gpr_ and the reduce MOP instruction stream.
 *
 *        This is the "compute once" half of the pair that splits
 *        @ref _set_dst_write_addr_by_rows_ so the shift amount is calculated a
 *        single time (when the tile shape is known) and reused across many
 *        per-tile dest-base calculations.
 *
 * @param num_rows_per_tile Number of data rows per tile.
 */
inline void _set_tile_shape_idx_gpr_(const std::uint32_t num_rows_per_tile)
{
    const std::uint32_t tile_shape_idx =
        (num_rows_per_tile == 64)
            ? 6
            : ((num_rows_per_tile == 32) ? 5 : ((num_rows_per_tile == 16) ? 4 : ((num_rows_per_tile == 8) ? 3 : ((num_rows_per_tile == 4) ? 2 : 1))));
    ckernel::regfile[p_gpr_math::TILE_SHAPE_IDX] = tile_shape_idx;
}

/**
 * @brief Sets the destination register base address depending on the tile index,
 *        using the tile-shape index previously stored in GPR TEMP0 by
 *        @ref _set_tile_shape_idx_gpr_ as the left-shift amount that converts
 *        tile_index into a dest offset.
 *
 *        This is the "use many" half of the pair that splits
 *        @ref _set_dst_write_addr_by_rows_; call @ref _set_tile_shape_idx_gpr_
 *        once before invoking this for each tile in the reduce.
 *
 * @param tile_index Tile index in the dest reg.
 *        16-bit dest reg data format -> tile_index = 0 - 7
 *        32-bit dest reg data format -> tile_index = 0 - 3
 */
inline void _set_dst_write_addr_by_rows_(const std::uint32_t tile_index)
{
    const std::uint32_t tile_shape_idx = ckernel::regfile[p_gpr_math::TILE_SHAPE_IDX];
    const std::uint32_t dst_index      = (tile_index << tile_shape_idx) + ckernel::trisc::_get_dest_buffer_base_();
    ckernel::trisc::_set_dest_section_base_<TRISC_ID>(dst_index);
}

inline void move_d2a_fixed_face(const std::uint8_t addrmod)
{
    // MOVD2A src is relative to dest_section_base + dest_counter.
    // Use fixed offsets (0, 8) — the dest counter handles face progression.
    // NOTE: For different tile dimensions we need different amounts of MOV* instructions; see separate issue.
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCA_VLD);
    TTI_MOVD2A(0, 0, addrmod, p_movd2a::MOV_8_ROWS, 0);
    TTI_MOVD2A(0, 8, addrmod, p_movd2a::MOV_8_ROWS, 8);
}

inline void move_d2b_fixed_face(const std::uint8_t addrmod)
{
    // MOVD2B src is relative to dest_section_base + dest_counter.
    // Use fixed offsets (0, 8) — the dest counter handles face progression.
    // NOTE: For different tile dimensions we need different amounts of MOV* instructions; see separate issue.
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::SRCB_VLD);
    TTI_MOVD2B(0, 0, addrmod, p_movd2b::MOV_8_ROWS, 0, 0);
    TTI_MOVD2B(0, 8, addrmod, p_movd2b::MOV_8_ROWS, 0, 8);
}

template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_reuse_dest_as_src()
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        move_d2a_fixed_face(ADDR_MOD_3);
    }
    else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        move_d2b_fixed_face(ADDR_MOD_3);
    }
}

} // namespace ckernel::math
