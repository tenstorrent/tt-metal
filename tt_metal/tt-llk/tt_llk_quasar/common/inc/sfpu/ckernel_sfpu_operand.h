// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <utility>

#include "sfpi.h"
#include "sfpu_reg.h"

namespace ckernel::sfpu
{

namespace detail
{

inline constexpr sfpi::UnpackSrcS srcs_reg {};

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
        // SFPI selects the SrcS register file itself; do not add SFPU_SRCS_BASE_ADDR.
        return srcs_reg[index];
    }
}

} // namespace detail

/// Load/store policy: register LAYOUT <-> VALUE vector type. SrcS stores must use NOINC.
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

/// One SFPU operand: register file, format and base offset. Indices are SFPI steps (SFP_ROWS rows each).
template <SfpuReg REG, typename FORMAT>
class SfpuOperand
{
    static_assert(REG == SfpuReg::Dest || REG == SfpuReg::SrcS, "Unsupported SFPU register space");

public:
    using value_type = typename FORMAT::value_type;

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

/// output[d] = MATH::apply(input[d]) for d < ITERATIONS; shared by Dest and SrcS.
template <typename MATH, int ITERATIONS, typename Input, typename Output>
sfpi_inline void calculate_unary_operands(const Input& input, const Output& output)
{
    static_assert(ITERATIONS > 0, "A unary SFPU op requires at least one SFPI access");
    static_assert(
        std::is_same_v<decltype(MATH::apply(std::declval<typename Input::value_type>())), typename Output::value_type>,
        "MATH::apply must map the input operand's value type to the output operand's value type");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        output.store(d, MATH::apply(input.load(d)));
    }
}

/// output[d] = MATH::apply(input0[d], input1[d]) for d < ITERATIONS; shared by Dest and SrcS.
template <typename MATH, int ITERATIONS, typename Input0, typename Input1, typename Output>
sfpi_inline void calculate_binary_operands(const Input0& input0, const Input1& input1, const Output& output)
{
    static_assert(ITERATIONS > 0, "A binary SFPU op requires at least one SFPI access");
    static_assert(
        std::is_same_v<
            decltype(MATH::apply(std::declval<typename Input0::value_type>(), std::declval<typename Input1::value_type>())),
            typename Output::value_type>,
        "MATH::apply must map the input operands' value types to the output operand's value type");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        output.store(d, MATH::apply(input0.load(d), input1.load(d)));
    }
}

} // namespace ckernel::sfpu
