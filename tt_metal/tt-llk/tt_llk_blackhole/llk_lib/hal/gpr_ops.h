// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "utils/gpr.h"

namespace hal::gpr_ops
{
namespace detail
{
inline constexpr std::uint32_t DynamicImmediateValue = 0xffffffffu;
}

/** @brief Identify a compile-time or runtime-selected scalar-unit immediate operand. */
template <std::uint32_t Value>
struct Immediate
{
    static constexpr std::uint32_t value = Value;
};

template <>
struct Immediate<detail::DynamicImmediateValue>
{
    std::uint32_t value;
};

/** @brief Construct a compile-time scalar-unit immediate operand. */
template <std::uint32_t Value>
inline constexpr auto immediate()
{
    static_assert(Value != detail::DynamicImmediateValue, "Immediate value is reserved by hal::gpr_ops::immediate()");
    return Immediate<Value> {};
}

/** @brief Construct a runtime scalar-unit immediate operand. */
inline constexpr auto immediate(const std::uint32_t value)
{
    return Immediate<detail::DynamicImmediateValue> {value};
}

/** @brief Select the unsigned comparison performed by compare(). */
enum class Compare : std::uint8_t
{
    GreaterThan,
    LessThan,
    Equal
};

namespace detail
{
enum class BinaryOperation : std::uint8_t
{
    Add,
    Subtract,
    MultiplyLowU16,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    CompareGreaterThan,
    CompareLessThan,
    CompareEqual
};

template <typename Operand>
struct OperandTraits;

template <std::uint32_t Index>
struct OperandTraits<hal::Gpr<Index>>
{
    static constexpr bool is_static             = Index != hal::detail::DynamicGprIndex;
    static constexpr bool is_immediate          = false;
    static constexpr std::uint32_t static_value = Index;

    static constexpr std::uint32_t value(const hal::Gpr<Index> operand)
    {
        if constexpr (is_static)
        {
            return Index;
        }
        else
        {
            return operand.index;
        }
    }
};

template <std::uint32_t Value>
struct OperandTraits<Immediate<Value>>
{
    static constexpr bool is_static             = Value != DynamicImmediateValue;
    static constexpr bool is_immediate          = true;
    static constexpr std::uint32_t static_value = Value;

    static constexpr std::uint32_t value(const Immediate<Value> operand)
    {
        if constexpr (is_static)
        {
            return Value;
        }
        else
        {
            return operand.value;
        }
    }
};

template <typename Operand>
using OperandType = std::decay_t<Operand>;

template <typename Operand>
inline constexpr bool is_static_operand = OperandTraits<OperandType<Operand>>::is_static;

template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline constexpr bool all_static_operands = Destination != hal::detail::DynamicGprIndex && Lhs != hal::detail::DynamicGprIndex && is_static_operand<Rhs>;

inline constexpr std::uint32_t ImmediateLimit = 63u;

template <BinaryOperation Operation>
inline constexpr std::uint32_t encode(const std::uint32_t destination, const std::uint32_t lhs, const std::uint32_t rhs, const bool rhs_is_immediate)
{
    if constexpr (Operation == BinaryOperation::Add)
    {
        return TT_OP_ADDDMAREG(rhs_is_immediate, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::Subtract)
    {
        return TT_OP_SUBDMAREG(rhs_is_immediate, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::MultiplyLowU16)
    {
        return TT_OP_MULDMAREG(rhs_is_immediate, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::BitAnd)
    {
        return TT_OP_BITWOPDMAREG(rhs_is_immediate, 0, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::BitOr)
    {
        return TT_OP_BITWOPDMAREG(rhs_is_immediate, 1, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::BitXor)
    {
        return TT_OP_BITWOPDMAREG(rhs_is_immediate, 2, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::ShiftLeft)
    {
        return TT_OP_SHIFTDMAREG(rhs_is_immediate, 0, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::ShiftRight)
    {
        return TT_OP_SHIFTDMAREG(rhs_is_immediate, 1, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::CompareGreaterThan)
    {
        return TT_OP_CMPDMAREG(rhs_is_immediate, 0, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::CompareLessThan)
    {
        return TT_OP_CMPDMAREG(rhs_is_immediate, 1, destination, rhs, lhs);
    }
    else
    {
        static_assert(Operation == BinaryOperation::CompareEqual);
        return TT_OP_CMPDMAREG(rhs_is_immediate, 2, destination, rhs, lhs);
    }
}

template <std::uint32_t Index>
inline __attribute__((always_inline)) void validate_gpr(const hal::Gpr<Index> operand)
{
    if constexpr (Index == hal::detail::DynamicGprIndex)
    {
        LLK_ASSERT(operand.index < 64u, "Scalar-unit GPR index must be in [0, 63]");
    }
    else
    {
        static_assert(Index < 64u, "Scalar-unit GPR index must be in [0, 63]");
    }
}

template <BinaryOperation Operation, typename Rhs>
inline __attribute__((always_inline)) void validate_rhs(const Rhs rhs)
{
    using Traits = OperandTraits<OperandType<Rhs>>;
    if constexpr (Traits::is_immediate)
    {
        if constexpr (Traits::is_static)
        {
            static_assert(Traits::static_value <= ImmediateLimit, "Scalar-unit immediate must fit in six bits");
        }
        else
        {
            LLK_ASSERT(Traits::value(rhs) <= ImmediateLimit, "Scalar-unit immediate must fit in six bits");
        }
    }
    else
    {
        validate_gpr(rhs);
    }
}

template <BinaryOperation Operation, std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline constexpr std::uint32_t static_binary_operation()
{
    using Traits = OperandTraits<OperandType<Rhs>>;
    static_assert(all_static_operands<Destination, Lhs, Rhs>);
    static_assert(Destination < 64u, "Scalar-unit destination GPR index must be in [0, 63]");
    static_assert(Lhs < 64u, "Scalar-unit lhs GPR index must be in [0, 63]");
    if constexpr (Traits::is_immediate)
    {
        static_assert(Traits::static_value <= ImmediateLimit, "Scalar-unit immediate must fit in six bits");
    }
    else
    {
        static_assert(Traits::static_value < 64u, "Scalar-unit rhs GPR index must be in [0, 63]");
    }
    return encode<Operation>(Destination, Lhs, Traits::static_value, Traits::is_immediate);
}

template <BinaryOperation Operation, std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline constexpr std::uint32_t binary_operation(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    using Traits = OperandTraits<OperandType<Rhs>>;
    if constexpr (all_static_operands<Destination, Lhs, Rhs>)
    {
        return static_binary_operation<Operation, Destination, Lhs, Rhs>();
    }
    else
    {
        validate_gpr(destination);
        validate_gpr(lhs);
        validate_rhs<Operation>(rhs);
        return encode<Operation>(
            OperandTraits<hal::Gpr<Destination>>::value(destination), OperandTraits<hal::Gpr<Lhs>>::value(lhs), Traits::value(rhs), Traits::is_immediate);
    }
}

template <BinaryOperation Operation, std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void emit_binary(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    if constexpr (all_static_operands<Destination, Lhs, Rhs>)
    {
        constexpr std::uint32_t operation = static_binary_operation<Operation, Destination, Lhs, Rhs>();
        INSTRUCTION_WORD(operation);
    }
    else
    {
        ckernel::instrn_buffer[0] = binary_operation<Operation>(destination, lhs, rhs);
    }
}

template <Compare Predicate>
inline constexpr BinaryOperation compare_operation = []
{
    static_assert(Predicate == Compare::GreaterThan || Predicate == Compare::LessThan || Predicate == Compare::Equal, "Unsupported scalar-unit comparison");
    if constexpr (Predicate == Compare::GreaterThan)
    {
        return BinaryOperation::CompareGreaterThan;
    }
    else if constexpr (Predicate == Compare::LessThan)
    {
        return BinaryOperation::CompareLessThan;
    }
    else
    {
        return BinaryOperation::CompareEqual;
    }
}();

template <std::uint32_t Index>
inline constexpr std::uint32_t half_index(const bool high)
{
    static_assert(Index < 64u, "Scalar-unit GPR index must be in [0, 63]");
    return Index * 2u + high;
}

} // namespace detail

/** @brief Encode a compile-time 16-bit write to the low half of one GPR. */
template <std::uint32_t Value, std::uint32_t Index>
inline constexpr std::uint32_t set_low_operation(const hal::Gpr<Index>)
{
    static_assert(Value < (1u << 16), "GPR half value must fit in 16 bits");
    return TT_OP_SETDMAREG(0, Value, 0, detail::half_index<Index>(false));
}

/** @brief Encode a runtime 16-bit write to the low half of one GPR. */
template <std::uint32_t Index>
inline __attribute__((always_inline)) std::uint32_t set_low_operation(const hal::Gpr<Index> destination, const std::uint32_t value)
{
    detail::validate_gpr(destination);
    LLK_ASSERT(value < (1u << 16), "GPR half value must fit in 16 bits");
    return TT_OP_SETDMAREG(0, value, 0, detail::OperandTraits<hal::Gpr<Index>>::value(destination) * 2u);
}

/** @brief Encode a compile-time 16-bit write to the high half of one GPR. */
template <std::uint32_t Value, std::uint32_t Index>
inline constexpr std::uint32_t set_high_operation(const hal::Gpr<Index>)
{
    static_assert(Value < (1u << 16), "GPR half value must fit in 16 bits");
    return TT_OP_SETDMAREG(0, Value, 0, detail::half_index<Index>(true));
}

/** @brief Encode a runtime 16-bit write to the high half of one GPR. */
template <std::uint32_t Index>
inline __attribute__((always_inline)) std::uint32_t set_high_operation(const hal::Gpr<Index> destination, const std::uint32_t value)
{
    detail::validate_gpr(destination);
    LLK_ASSERT(value < (1u << 16), "GPR half value must fit in 16 bits");
    return TT_OP_SETDMAREG(0, value, 0, detail::OperandTraits<hal::Gpr<Index>>::value(destination) * 2u + 1u);
}

/** @brief Set a GPR to a compile-time 32-bit value. */
template <std::uint32_t Value, std::uint32_t Index>
inline __attribute__((always_inline)) void set(const hal::Gpr<Index> destination)
{
    (void)set_low_operation<Value & 0xffffu>(destination);
    TTI_SETDMAREG(0, Value & 0xffffu, 0, detail::half_index<Index>(false));
    TTI_SETDMAREG(0, Value >> 16, 0, detail::half_index<Index>(true));
}

/** @brief Set a GPR to a runtime 32-bit value. */
template <std::uint32_t Index>
inline __attribute__((always_inline)) void set(const hal::Gpr<Index> destination, const std::uint32_t value)
{
    detail::validate_gpr(destination);
    const std::uint32_t index = detail::OperandTraits<hal::Gpr<Index>>::value(destination);
    TT_SETDMAREG(0, value & 0xffffu, 0, index * 2u);
    TT_SETDMAREG(0, value >> 16, 0, index * 2u + 1u);
}

/** @brief Set the low half of a GPR to a compile-time 16-bit value. */
template <std::uint32_t Value, std::uint32_t Index>
inline __attribute__((always_inline)) void set_low(const hal::Gpr<Index> destination)
{
    (void)set_low_operation<Value>(destination);
    TTI_SETDMAREG(0, Value, 0, detail::half_index<Index>(false));
}

/** @brief Set the low half of a GPR to a runtime 16-bit value. */
template <std::uint32_t Index>
inline __attribute__((always_inline)) void set_low(const hal::Gpr<Index> destination, const std::uint32_t value)
{
    ckernel::instrn_buffer[0] = set_low_operation(destination, value);
}

/** @brief Set the high half of a GPR to a compile-time 16-bit value. */
template <std::uint32_t Value, std::uint32_t Index>
inline __attribute__((always_inline)) void set_high(const hal::Gpr<Index> destination)
{
    (void)set_high_operation<Value>(destination);
    TTI_SETDMAREG(0, Value, 0, detail::half_index<Index>(true));
}

/** @brief Set the high half of a GPR to a runtime 16-bit value. */
template <std::uint32_t Index>
inline __attribute__((always_inline)) void set_high(const hal::Gpr<Index> destination, const std::uint32_t value)
{
    ckernel::instrn_buffer[0] = set_high_operation(destination, value);
}

/** @brief Encode an unsigned 32-bit wrapping addition without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline constexpr std::uint32_t add_operation(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    return detail::binary_operation<detail::BinaryOperation::Add>(destination, lhs, rhs);
}

/** @brief Add two operands with unsigned 32-bit wraparound. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void add(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::Add>(destination, lhs, rhs);
}

/** @brief Encode an unsigned 32-bit wrapping subtraction without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline constexpr std::uint32_t subtract_operation(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    return detail::binary_operation<detail::BinaryOperation::Subtract>(destination, lhs, rhs);
}

/** @brief Subtract the rhs from the lhs with unsigned 32-bit wraparound. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void subtract(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::Subtract>(destination, lhs, rhs);
}

/** @brief Encode a multiplication of the operands' low 16 bits without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline constexpr std::uint32_t multiply_low_u16_operation(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    return detail::binary_operation<detail::BinaryOperation::MultiplyLowU16>(destination, lhs, rhs);
}

/** @brief Multiply the low 16 bits of each operand and write the full 32-bit result. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void multiply_low_u16(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::MultiplyLowU16>(destination, lhs, rhs);
}

/** @brief Encode a bitwise AND without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline constexpr std::uint32_t bit_and_operation(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    return detail::binary_operation<detail::BinaryOperation::BitAnd>(destination, lhs, rhs);
}

/** @brief Compute a bitwise AND. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void bit_and(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::BitAnd>(destination, lhs, rhs);
}

/** @brief Encode a bitwise OR without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline constexpr std::uint32_t bit_or_operation(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    return detail::binary_operation<detail::BinaryOperation::BitOr>(destination, lhs, rhs);
}

/** @brief Compute a bitwise OR. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void bit_or(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::BitOr>(destination, lhs, rhs);
}

/** @brief Encode a bitwise XOR without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline constexpr std::uint32_t bit_xor_operation(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    return detail::binary_operation<detail::BinaryOperation::BitXor>(destination, lhs, rhs);
}

/** @brief Compute a bitwise XOR. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void bit_xor(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::BitXor>(destination, lhs, rhs);
}

/** @brief Encode a logical left shift without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline constexpr std::uint32_t shift_left_operation(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    return detail::binary_operation<detail::BinaryOperation::ShiftLeft>(destination, lhs, rhs);
}

/** @brief Shift the lhs left logically by the rhs low five bits. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void shift_left(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::ShiftLeft>(destination, lhs, rhs);
}

/** @brief Encode a logical right shift without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline constexpr std::uint32_t shift_right_operation(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    return detail::binary_operation<detail::BinaryOperation::ShiftRight>(destination, lhs, rhs);
}

/** @brief Shift the lhs right logically by the rhs low five bits. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void shift_right(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::ShiftRight>(destination, lhs, rhs);
}

/** @brief Encode an unsigned comparison without issuing it. */
template <Compare Predicate, std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline constexpr std::uint32_t compare_operation(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    return detail::binary_operation<detail::compare_operation<Predicate>>(destination, lhs, rhs);
}

/** @brief Compare two unsigned 32-bit operands and write zero or one. */
template <Compare Predicate, std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void compare(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::compare_operation<Predicate>>(destination, lhs, rhs);
}

} // namespace hal::gpr_ops
