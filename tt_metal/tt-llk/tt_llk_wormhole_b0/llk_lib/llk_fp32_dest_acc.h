// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "llk_assert.h"

using namespace ckernel;

namespace fp32_dest_acc
{
constexpr std::uint32_t UNPACK_READY = 0x46504101; // 'FPA' | 0x01
constexpr std::uint32_t MATH_DONE    = 0x46504110; // 'FPA' | 0x10
// PACK's word: this tag in bits 31:8, its pack source format (DataFormat underlying value) in bits 7:0.
constexpr std::uint32_t PACK_READY_TAG   = 0x46504200; // 'FPB' << 8
constexpr std::uint32_t PACK_FORMAT_MASK = 0xFF;
} // namespace fp32_dest_acc

/**
 * @brief Coordinate a mid-kernel FP32 dest-acc reconfiguration across Unpack, Math, and Pack.
 *
 * Dest-acc CFG is MATH-owned. tensix_sync drains each thread's Tensix FIFO (including FPU / SFPU /
 * packer); the mailbox then holds RISC so no new work is issued until MATH has written dest-acc:
 *   1. UNPACK/PACK tensix_sync, signal MATH (PACK's word carries its pack source format), and wait.
 *   2. MATH tensix_syncs, waits for both, programs ALU_ACC_CTRL and PCK_DEST_RD_CTRL (derived from the
 *      new mode and PACK's source format the way configure_pack derives it), drains until
 *      those writes have retired, and only then releases UNPACK/PACK. The release is a RISC store,
 *      so the drain -- not a stall mask -- is what orders it behind the writes, and it is what makes
 *      the writes visible to the released threads.
 *   3. UNPACK/PACK need no stall of their own: the drain in step 2 returns only once the writes
 *      have landed, so they are in effect before either thread is released.
 *
 * @tparam thread_id: TRISC thread compiling this specialization, values = <MathThreadId>
 * @param enable: True to enable FP32 dest accumulation, false to disable.
 * @note All three TRISC threads must call their specialization together: PACK passes its pack source
 *       format, UNPACK takes no argument. Not supported on Quasar.
 */
template <ThreadId thread_id>
inline void _llk_set_fp32_dest_acc_(bool enable)
{
    static_assert(thread_id == ThreadId::MathThreadId, "_llk_set_fp32_dest_acc_(enable) is the MATH arm; PACK passes its pack source format, UNPACK nothing");

    tensix_sync();

    const std::uint32_t unpack_ready = mailbox_read(ThreadId::UnpackThreadId);
    const std::uint32_t pack_word    = mailbox_read(ThreadId::PackThreadId);
    LLK_ASSERT(unpack_ready == fp32_dest_acc::UNPACK_READY, "Unexpected dest-acc message from unpack thread.");
    LLK_ASSERT((pack_word & ~fp32_dest_acc::PACK_FORMAT_MASK) == fp32_dest_acc::PACK_READY_TAG, "Unexpected dest-acc message from pack thread.");

    // The packer's Dest-read control, derived as configure_pack derives it: a 32-bit pack source keeps the
    // 32-bit Dest read whatever the mode, an 8-bit source takes the 8-bit integer read only in a 16-bit Dest.
    const std::uint32_t pack_src_format = pack_word & fp32_dest_acc::PACK_FORMAT_MASK;
    const bool is_32b_format            = pack_src_format == to_underlying(DataFormat::Int32) || pack_src_format == to_underlying(DataFormat::UInt32) ||
                               pack_src_format == to_underlying(DataFormat::Float32);
    const bool is_int8_format = pack_src_format == to_underlying(DataFormat::Int8) || pack_src_format == to_underlying(DataFormat::UInt8);
    const bool read_32b_data  = enable || is_32b_format;
    const bool read_int8      = !read_32b_data && is_int8_format;

    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(enable);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(enable);
    static_assert(
        PCK_DEST_RD_CTRL_Read_32b_data_ADDR32 == PCK_DEST_RD_CTRL_Read_int8_ADDR32 && PCK_DEST_RD_CTRL_Read_int8_SHAMT > PCK_DEST_RD_CTRL_Read_32b_data_SHAMT,
        "PCK_DEST_RD_CTRL_Read_32b_data and Read_int8 must share ADDR32 for a combined RMW");
    cfg_reg_rmw_tensix<
        PCK_DEST_RD_CTRL_Read_32b_data_ADDR32,
        PCK_DEST_RD_CTRL_Read_32b_data_SHAMT,
        PCK_DEST_RD_CTRL_Read_32b_data_MASK | PCK_DEST_RD_CTRL_Read_int8_MASK>(
        (read_32b_data ? 1u : 0u) | (read_int8 ? (1u << (PCK_DEST_RD_CTRL_Read_int8_SHAMT - PCK_DEST_RD_CTRL_Read_32b_data_SHAMT)) : 0u));

    // The releases below are RISC stores, which the Wait Gate does not gate. Drain so the
    // config writes have retired before either thread is let go.
    tensix_sync();

    mailbox_write(ThreadId::UnpackThreadId, fp32_dest_acc::MATH_DONE);
    mailbox_write(ThreadId::PackThreadId, fp32_dest_acc::MATH_DONE);
}

/**
 * @brief PACK arm of the mid-kernel FP32 dest-acc reconfiguration above.
 *
 * Drains the packer, hands MATH the pack source format it is configured for, and returns once MATH has
 * programmed the new mode.
 *
 * @tparam thread_id: TRISC thread compiling this specialization, values = <PackThreadId>
 * @param pack_src_format: Pack source (Dest-side) data format of the output the packer is configured for.
 *        MATH derives PCK_DEST_RD_CTRL from it together with the new mode.
 */
template <ThreadId thread_id>
inline void _llk_set_fp32_dest_acc_(DataFormat pack_src_format)
{
    static_assert(thread_id == ThreadId::PackThreadId, "_llk_set_fp32_dest_acc_(pack_src_format) is the PACK arm; MATH passes enable, UNPACK nothing");
    static_assert(sizeof(DataFormat) == 1, "the pack source format must fit the low byte of PACK's mailbox word");

    tensix_sync();

    mailbox_write(ThreadId::MathThreadId, fp32_dest_acc::PACK_READY_TAG | (to_underlying(pack_src_format) & fp32_dest_acc::PACK_FORMAT_MASK));
    const std::uint32_t math_done = mailbox_read(ThreadId::MathThreadId);
    LLK_ASSERT(math_done == fp32_dest_acc::MATH_DONE, "Unexpected dest-acc message from math thread.");
}

/**
 * @brief UNPACK arm of the mid-kernel FP32 dest-acc reconfiguration above.
 *
 * Drains the unpacker, signals MATH, and returns once MATH has programmed the new mode.
 *
 * @tparam thread_id: TRISC thread compiling this specialization, values = <UnpackThreadId>
 */
template <ThreadId thread_id>
inline void _llk_set_fp32_dest_acc_()
{
    static_assert(
        thread_id == ThreadId::UnpackThreadId, "the no-argument _llk_set_fp32_dest_acc_ is the UNPACK arm; MATH passes enable, PACK its pack source format");

    tensix_sync();

    mailbox_write(ThreadId::MathThreadId, fp32_dest_acc::UNPACK_READY);
    const std::uint32_t math_done = mailbox_read(ThreadId::MathThreadId);
    LLK_ASSERT(math_done == fp32_dest_acc::MATH_DONE, "Unexpected dest-acc message from math thread.");
}
