// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpi.h"
#include "sfpu_reg.h"

namespace ckernel::sfpu
{

namespace detail
{

template <SfpuReg REG>
sfpi_inline auto sfpu_operand_access(int index)
{
    static_assert(REG == SfpuReg::Dest || REG == SfpuReg::SrcS, "Unsupported SFPU register space");
    if constexpr (REG == SfpuReg::Dest)
    {
        return sfpi::dst_reg[index];
    }
    else
    {
        // SFPI's SrcS builtins select the register file (SFP_SRCSREG_BASE is 0); never add
        // SFPU_SRCS_BASE_ADDR here. The returned proxy owns its address, not this local.
        sfpi::UnpackSrcS srcs;
        return srcs[index];
    }
}

} // namespace detail

/**
 * @brief Load/store policy using SFPI's layout and vector-type conversions.
 *
 * LAYOUT is the register-file representation, VALUE the vector type it converts to (e.g. F16b
 * with vFloat). SFPI enforces the legal pairs.
 *
 * @tparam STORE_ADDR_MODE: NOINC (default) or a Dest ADDR_MOD_* that advances the cursor; the
 *         caller programs that mode. SrcS stores must use NOINC.
 */
template <sfpi::DataLayout LAYOUT, typename VALUE, int STORE_ADDR_MODE = sfpi::SFPSTORE_ADDR_MODE_NOINC>
struct SfpiFormat
{
    using value_type                         = VALUE;
    static constexpr sfpi::DataLayout layout = LAYOUT;

    template <SfpuReg REG>
    sfpi_inline static value_type load(int index)
    {
        return detail::sfpu_operand_access<REG>(index).template mode<LAYOUT>();
    }

    template <SfpuReg REG>
    sfpi_inline static void store(int index, value_type value)
    {
        static_assert(REG == SfpuReg::Dest || STORE_ADDR_MODE == sfpi::SFPSTORE_ADDR_MODE_NOINC, "SrcS requires no-increment addressing");
        detail::sfpu_operand_access<REG>(index).template mode<LAYOUT>(STORE_ADDR_MODE) = value;
    }
};

/**
 * @brief One SFPU input or output operand: register file, format policy and base offset.
 *
 * Indices are SFPI steps (one step = SFP_ROWS rows), not tile indices or raw addresses. Dest is
 * relative to the current cursor; SrcS to UnpackSrcS's base. An operand never advances the Dest
 * cursor (unless FORMAT's store mode does) and never completes SrcS slices; the caller owns
 * traversal and synchronization. Constant offsets compile to immediate addresses.
 */
template <SfpuReg REG, class FORMAT>
class SfpuOperand
{
    static_assert(REG == SfpuReg::Dest || REG == SfpuReg::SrcS, "Unsupported SFPU register space");

public:
    using format_type            = FORMAT;
    using value_type             = typename FORMAT::value_type;
    static constexpr SfpuReg reg = REG;

    sfpi_inline constexpr explicit SfpuOperand(int base_offset = 0) : base_offset_(base_offset)
    {
    }

    sfpi_inline value_type load(int index) const
    {
        return FORMAT::template load<REG>(base_offset_ + index);
    }

    sfpi_inline void store(int index, value_type value) const
    {
        FORMAT::template store<REG>(base_offset_ + index, value);
    }

private:
    int base_offset_;
};

} // namespace ckernel::sfpu
