// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cfg_defines.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "tensix_types.h"

#define RISCV_DEST_START_ADDR 0xFFBD8000

#define RISC_DEST_FMT_FP32  0b000
#define RISC_DEST_FMT_INT32 0b001
#define RISC_DEST_FMT_FP16A 0b010
#define RISC_DEST_FMT_FP16B 0b011
#define RISC_DEST_FMT_INT16 0b100
#define RISC_DEST_FMT_INT8  0b101

namespace ckernel
{

inline std::uint8_t fmt_to_dest_type(DataFormat fmt)
{
    switch (fmt)
    {
        case DataFormat::Float32:
            return RISC_DEST_FMT_FP32;
        case DataFormat::Float16:
            return RISC_DEST_FMT_FP16A;
        case DataFormat::Float16_b:
            return RISC_DEST_FMT_FP16B;
        case DataFormat::Int32:
        case DataFormat::UInt32:
            return RISC_DEST_FMT_INT32;
        case DataFormat::UInt16:
            return RISC_DEST_FMT_INT16;
        case DataFormat::Int8:
        case DataFormat::UInt8:
            return RISC_DEST_FMT_INT8;
        default:
            return RISC_DEST_FMT_INT16;
    }
}

template <ThreadId thread_id>
inline void set_dest_fmt(std::uint32_t fmt)
{
    static_assert(IS_TRISC_THREAD<thread_id>, "Thread must be UnpackThreadId or MathThreadId or PackThreadId");

    if constexpr (thread_id == UnpackThreadId)
    {
        cfg_reg_rmw_tensix<RISC_DEST_ACCESS_CTRL_SEC0_fmt_RMW>(fmt);
    }
    else if constexpr (thread_id == MathThreadId)
    {
        cfg_reg_rmw_tensix<RISC_DEST_ACCESS_CTRL_SEC1_fmt_RMW>(fmt);
    }
    else if constexpr (thread_id == PackThreadId)
    {
        cfg_reg_rmw_tensix<RISC_DEST_ACCESS_CTRL_SEC2_fmt_RMW>(fmt);
    }
}

template <ThreadId thread_id>
inline void set_dest_fmt(DataFormat fmt)
{
    set_dest_fmt<thread_id>(fmt_to_dest_type(fmt));
}

template <ThreadId thread_id>
inline void set_dest_unsigned_int_rmw(const int val)
{
    static_assert(IS_TRISC_THREAD<thread_id>, "Thread must be UnpackThreadId or MathThreadId or PackThreadId");

    if constexpr (thread_id == UnpackThreadId)
    {
        cfg_reg_rmw_tensix<RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int_RMW>(val);
    }
    else if constexpr (thread_id == MathThreadId)
    {
        cfg_reg_rmw_tensix<RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int_RMW>(val);
    }
    else if constexpr (thread_id == PackThreadId)
    {
        cfg_reg_rmw_tensix<RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW>(val);
    }
}

template <ThreadId thread_id>
inline void set_dest_int8_int16_signed(const bool is_signed)
{
    set_dest_unsigned_int_rmw<thread_id>(is_signed ? 0 : 1);
}

template <ThreadId thread_id>
inline void set_dest_no_swizzle_rmw(const int val)
{
    static_assert(IS_TRISC_THREAD<thread_id>, "Thread must be UnpackThreadId or MathThreadId or PackThreadId");

    // In unswizzled mode, values are written into dest as-is with
    // no saturation checks and no bit shuffling. This means they
    // are incompatible with the FPU, but it could be useful for
    // debugging
    if constexpr (thread_id == UnpackThreadId)
    {
        cfg_reg_rmw_tensix<RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle_RMW>(val);
    }
    else if constexpr (thread_id == MathThreadId)
    {
        cfg_reg_rmw_tensix<RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle_RMW>(val);
    }
    else if constexpr (thread_id == PackThreadId)
    {
        cfg_reg_rmw_tensix<RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW>(val);
    }
}

template <ThreadId thread_id>
inline void set_dest_enable_swizzling(const bool enable)
{
    set_dest_no_swizzle_rmw<thread_id>(enable ? 0 : 1);
}

inline bool dest_fmt_is_signed(DataFormat fmt)
{
    return fmt != DataFormat::UInt8 && fmt != DataFormat::UInt16 && fmt != DataFormat::UInt32;
}

// Program this thread's RISC_DEST_ACCESS_CTRL section for MMIO DEST access.
template <ThreadId thread_id>
inline void configure_dest_access(DataFormat fmt, bool enable_swizzle = true)
{
    set_dest_fmt<thread_id>(fmt);
    set_dest_enable_swizzling<thread_id>(enable_swizzle);
    set_dest_int8_int16_signed<thread_id>(dest_fmt_is_signed(fmt));
}

} // namespace ckernel
