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
        // SFPI's SrcS builtins select the register file (SFP_SRCSREG_BASE is 0); never add
        // SFPU_SRCS_BASE_ADDR here.
        return srcs_reg[index];
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
 * traversal and synchronization.
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

/**
 * @brief Apply a unary math policy over ITERATIONS SFPI steps: output[d] = MATH::apply(input[d]).
 *
 * The one load -> math -> store loop shared by every register file: Dest and SrcS callers differ
 * only in the operands they pass. Advances explicit indices only; the caller owns traversal and
 * synchronization. Input/output ranges must coincide or be disjoint.
 *
 * @tparam MATH: Math policy with static apply(value) -> value, e.g. ExpHwLut.
 * @tparam ITERATIONS: Number of SFPI steps.
 */
template <class MATH, int ITERATIONS, class Input, class Output>
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

/**
 * @brief Apply a binary math policy over ITERATIONS SFPI steps: output[d] = MATH::apply(input0[d], input1[d]).
 *
 * Binary counterpart of @ref calculate_unary_operands; any rounding is the output operand's store
 * policy. Each input range must coincide with the output or be disjoint.
 *
 * @tparam MATH: Math policy with static apply(a, b) -> value, e.g. AddMath.
 * @tparam ITERATIONS: Number of SFPI steps.
 */
template <class MATH, int ITERATIONS, class Input0, class Input1, class Output>
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
