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
        sfpi::UnpackSrcS srcs;
        // The returned proxy owns its address; it does not reference this local accessor.
        return srcs[index];
    }
}

} // namespace detail

/**
 * @brief Load/store policy using SFPI's layout and vector-type conversions.
 *
 * LAYOUT describes memory representation; VALUE is the loaded/stored vector type.
 * For example, F16b with vFloat loads BF16 data into a floating-point vector.
 * Supported layout/value pairs are enforced by SFPI. Formats not exposed by SFPI's
 * register proxies require a separate policy with the same load/store interface.
 * STORE_ADDR_MODE defaults to no increment. A Dest adapter may select a different
 * mode to preserve hardware cursor advancement; its caller must configure that mode.
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
 * @brief One independently located SFPU input or output operand on Quasar.
 *
 * FORMAT supplies value_type and load<REG>(index) / store<REG>(index, value).
 * Base and access indices are in SFPI units, not tile indices or raw instruction
 * addresses. Dest is relative to the current hardware cursor; SrcS is relative
 * to UnpackSrcS's base (do not include SFPU_SRCS_BASE_ADDR).
 *
 * Accesses through SfpiFormat default to the no-increment address mode. The caller sets
 * up ADDR_MOD_7, formats and geometry, checks ranges/overlap, and handles traversal
 * and synchronization. Only an explicitly selected store address mode advances Dest;
 * an operand never completes SrcS slices. Use such modes with single-access kernels
 * when traversal is controlled by an outer loop, to avoid advancing twice.
 * Keep offsets constant where possible to enable immediate instruction addresses.
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
