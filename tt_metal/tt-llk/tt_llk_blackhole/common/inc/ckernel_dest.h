// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef CKERNEL_DEST_H
#define CKERNEL_DEST_H 1

#include <cstdint>

#include "cfg_defines.h"
#include "ckernel.h"
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

template <ThreadId thread_id>
inline void set_dest_fmt(std::uint32_t fmt)
{
    static_assert(
        thread_id == MathThreadId || thread_id == PackThreadId || thread_id == UnpackThreadId, "Thread must be UnpackThreadId or MathThreadId or PackThreadId");

    if constexpr (thread_id == UnpackThreadId)
    {
        cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_fmt_RMW, fmt);
    }
    else if constexpr (thread_id == MathThreadId)
    {
        cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_fmt_RMW, fmt);
    }
    else if constexpr (thread_id == PackThreadId)
    {
        cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_fmt_RMW, fmt);
    }
}

inline void set_dest_fmt(std::uint32_t fmt, ThreadId thread_id)
{
    // FWLOG1("Setting RISC-dest access format to %d", fmt);
    if (thread_id == UnpackThreadId)
    {
        set_dest_fmt<UnpackThreadId>(fmt);
    }
    else if (thread_id == MathThreadId)
    {
        set_dest_fmt<MathThreadId>(fmt);
    }
    else
    {
        set_dest_fmt<PackThreadId>(fmt);
    }
}

template <ThreadId thread_id, bool is_signed>
inline void set_dest_int8_int16_signed()
{
    static_assert(
        thread_id == MathThreadId || thread_id == PackThreadId || thread_id == UnpackThreadId, "Thread must be UnpackThreadId or MathThreadId or PackThreadId");

    if (is_signed)
    {
        // FWLOG0("Setting RISC-dest int8 access mode to signed");
        if constexpr (thread_id == UnpackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int_RMW, 0);
        }
        else if constexpr (thread_id == MathThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int_RMW, 0);
        }
        else if constexpr (thread_id == PackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 0);
        }
    }
    else
    {
        // FWLOG0("Setting RISC-dest int8 access mode to unsigned");
        if constexpr (thread_id == UnpackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int_RMW, 1);
        }
        else if constexpr (thread_id == MathThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int_RMW, 1);
        }
        else if constexpr (thread_id == PackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 1);
        }
    }
}

template <ThreadId thread_id>
inline void set_dest_int8_int16_signed(bool const is_signed)
{
    static_assert(
        thread_id == MathThreadId || thread_id == PackThreadId || thread_id == UnpackThreadId, "Thread must be UnpackThreadId or MathThreadId or PackThreadId");

    if (is_signed)
    {
        if constexpr (thread_id == UnpackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int_RMW, 0);
        }
        else if constexpr (thread_id == MathThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int_RMW, 0);
        }
        else if constexpr (thread_id == PackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 0);
        }
    }
    else
    {
        if constexpr (thread_id == UnpackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int_RMW, 1);
        }
        else if constexpr (thread_id == MathThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int_RMW, 1);
        }
        else if constexpr (thread_id == PackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 1);
        }
    }
}

inline void set_dest_int8_int16_signed(ThreadId const thread_id, bool const is_signed)
{
    if (thread_id == UnpackThreadId)
    {
        set_dest_int8_int16_signed<UnpackThreadId>(is_signed);
    }
    else if (thread_id == MathThreadId)
    {
        set_dest_int8_int16_signed<MathThreadId>(is_signed);
    }
    else
    {
        set_dest_int8_int16_signed<PackThreadId>(is_signed);
    }
}

template <ThreadId thread_id, bool enable>
inline void set_dest_enable_swizzling()
{
    static_assert(
        thread_id == MathThreadId || thread_id == PackThreadId || thread_id == UnpackThreadId, "Thread must be UnpackThreadId or MathThreadId or PackThreadId");

    if (enable)
    {
        // In unswizzled mode, values are written into dest as-is with
        // no saturation checks and no bit shuffling. This means they
        // are incompatible with the FPU, but it could be useful for
        // debugging
        if constexpr (thread_id == UnpackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle_RMW, 0);
        }
        else if constexpr (thread_id == MathThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle_RMW, 0);
        }
        else if constexpr (thread_id == PackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 0);
        }
    }
    else
    {
        if constexpr (thread_id == UnpackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle_RMW, 1);
        }
        else if constexpr (thread_id == MathThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle_RMW, 1);
        }
        else if constexpr (thread_id == PackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 1);
        }
    }
}

template <ThreadId thread_id>
inline void set_dest_enable_swizzling(bool const enable)
{
    static_assert(
        thread_id == MathThreadId || thread_id == PackThreadId || thread_id == UnpackThreadId, "Thread must be UnpackThreadId or MathThreadId or PackThreadId");

    if (enable)
    {
        // In unswizzled mode, values are written into dest as-is with
        // no saturation checks and no bit shuffling. This means they
        // are incompatible with the FPU, but it could be useful for
        // debugging
        if constexpr (thread_id == UnpackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle_RMW, 0);
        }
        else if constexpr (thread_id == MathThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle_RMW, 0);
        }
        else if constexpr (thread_id == PackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 0);
        }
    }
    else
    {
        if constexpr (thread_id == UnpackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle_RMW, 1);
        }
        else if constexpr (thread_id == MathThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle_RMW, 1);
        }
        else if constexpr (thread_id == PackThreadId)
        {
            cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 1);
        }
    }
}

inline void set_dest_enable_swizzling(ThreadId const thread_id, bool const enable)
{
    if (thread_id == UnpackThreadId)
    {
        set_dest_enable_swizzling<UnpackThreadId>(enable);
    }
    else if (thread_id == MathThreadId)
    {
        set_dest_enable_swizzling<MathThreadId>(enable);
    }
    else
    {
        set_dest_enable_swizzling<PackThreadId>(enable);
    }
}

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
inline void set_dest_fmt(DataFormat fmt)
{
    set_dest_fmt<thread_id>(fmt_to_dest_type(fmt));
}

} // namespace ckernel

#endif
