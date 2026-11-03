// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "llk_assert.h"
#include "utils/gpr.h"
#include "utils/helpers.h"

/**
 * @brief Scalar-unit programming: GPR loads, arithmetic, bitwise, shift, and
 *        comparison operations.
 *
 * @note Scalar operations write their destination GPR from the scalar unit,
 *       asynchronously from this RISC. Precede a WRCFG that reads such a GPR
 *       (e.g. a hal::cfg write from a GPR) with a scalar drain: call
 *       @ref hal::sync::wait::stall(StallTarget::Cfg, StallCondition::ScalarIdle)
 *       first. The consumer side of this contract is on hal::cfg's GPR transfer.
 */
namespace hal::scalar
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
    static_assert(Value != detail::DynamicImmediateValue, "Immediate value is reserved by hal::scalar::immediate()");
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
class OperandTraits
{
    static_assert(!sizeof(Operand), "Scalar-unit rhs must be a hal::Gpr or a hal::scalar::Immediate");
};

template <std::uint32_t Index>
class OperandTraits<hal::Gpr<Index>>
{
public:
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
class OperandTraits<Immediate<Value>>
{
public:
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

/**
 * @brief DMAREG instruction field domains.
 */
struct DMAReg
{
    /**
     * @brief SETDMAREG Payload_SigSelSize field: payload width select.
     *
     * This header issues only the immediate (16-bit) payload.
     */
    struct Payload
    {
        static constexpr std::uint32_t IMMEDIATE    = 0;
        static constexpr std::uint32_t WIDTH_32BIT  = 1;
        static constexpr std::uint32_t WIDTH_128BIT = 2;
        static constexpr std::uint32_t TILE_HEADER  = 3;
    };

    /**
     * @brief SETDMAREG SetSignalsMode field: immediate value or
     *        signal/semaphore write.
     *
     * This header issues only immediate values.
     */
    struct Mode
    {
        static constexpr std::uint32_t IMMEDIATE = 0;
        static constexpr std::uint32_t SIGNAL    = 1;
    };

    /** @brief BITWOPDMAREG OpSel field: bitwise operation select. */
    struct BitwiseOpSel
    {
        static constexpr std::uint32_t AND = 0;
        static constexpr std::uint32_t OR  = 1;
        static constexpr std::uint32_t XOR = 2;
    };

    /** @brief SHIFTDMAREG OpSel field: shift direction select. */
    struct ShiftOpSel
    {
        static constexpr std::uint32_t LEFT  = 0;
        static constexpr std::uint32_t RIGHT = 1;
    };
};

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
        return TT_OP_BITWOPDMAREG(rhs_is_immediate, DMAReg::BitwiseOpSel::AND, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::BitOr)
    {
        return TT_OP_BITWOPDMAREG(rhs_is_immediate, DMAReg::BitwiseOpSel::OR, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::BitXor)
    {
        return TT_OP_BITWOPDMAREG(rhs_is_immediate, DMAReg::BitwiseOpSel::XOR, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::ShiftLeft)
    {
        return TT_OP_SHIFTDMAREG(rhs_is_immediate, DMAReg::ShiftOpSel::LEFT, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::ShiftRight)
    {
        return TT_OP_SHIFTDMAREG(rhs_is_immediate, DMAReg::ShiftOpSel::RIGHT, destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::CompareGreaterThan)
    {
        return TT_OP_CMPDMAREG(rhs_is_immediate, to_underlying(Compare::GreaterThan), destination, rhs, lhs);
    }
    else if constexpr (Operation == BinaryOperation::CompareLessThan)
    {
        return TT_OP_CMPDMAREG(rhs_is_immediate, to_underlying(Compare::LessThan), destination, rhs, lhs);
    }
    else
    {
        static_assert(Operation == BinaryOperation::CompareEqual);
        return TT_OP_CMPDMAREG(rhs_is_immediate, to_underlying(Compare::Equal), destination, rhs, lhs);
    }
}

inline __attribute__((always_inline)) void assert_valid(const bool valid, const char* message)
{
    LLK_ASSERT(valid, message);
}

constexpr void require_valid(const bool valid, const char* message)
{
    if (__builtin_is_constant_evaluated())
    {
        if (!valid)
        {
            __builtin_trap();
        }
    }
    else
    {
        assert_valid(valid, message);
    }
}

template <std::uint32_t Index>
constexpr void validate_gpr(const hal::Gpr<Index> operand)
{
    if constexpr (Index == hal::detail::DynamicGprIndex)
    {
        require_valid(operand.index < hal::detail::GprCount, "Scalar-unit GPR index must be in [0, 63]");
    }
    else
    {
        static_assert(Index < hal::detail::GprCount, "Scalar-unit GPR index must be in [0, 63]");
    }
}

template <BinaryOperation Operation, std::uint32_t Value>
inline constexpr void validate_static_immediate()
{
    if constexpr (Operation == BinaryOperation::ShiftLeft || Operation == BinaryOperation::ShiftRight)
    {
        static_assert(Value <= 31u, "Scalar-unit shift amount must fit in five bits");
    }
    else
    {
        static_assert(Value <= ImmediateLimit, "Scalar-unit immediate must fit in six bits");
    }
}

template <BinaryOperation Operation>
constexpr void validate_runtime_immediate(const std::uint32_t value)
{
    if constexpr (Operation == BinaryOperation::ShiftLeft || Operation == BinaryOperation::ShiftRight)
    {
        require_valid(value <= 31u, "Scalar-unit shift amount must fit in five bits");
    }
    else
    {
        require_valid(value <= ImmediateLimit, "Scalar-unit immediate must fit in six bits");
    }
}

template <BinaryOperation Operation, typename Rhs>
constexpr void validate_rhs(const Rhs rhs)
{
    using Traits = OperandTraits<OperandType<Rhs>>;
    if constexpr (Traits::is_immediate)
    {
        if constexpr (Traits::is_static)
        {
            validate_static_immediate<Operation, Traits::static_value>();
        }
        else
        {
            validate_runtime_immediate<Operation>(Traits::value(rhs));
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
    static_assert(Destination < hal::detail::GprCount, "Scalar-unit destination GPR index must be in [0, 63]");
    static_assert(Lhs < hal::detail::GprCount, "Scalar-unit lhs GPR index must be in [0, 63]");
    if constexpr (Traits::is_immediate)
    {
        validate_static_immediate<Operation, Traits::static_value>();
    }
    else
    {
        static_assert(Traits::static_value < hal::detail::GprCount, "Scalar-unit rhs GPR index must be in [0, 63]");
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
        TTI_INSN(operation);
    }
    else
    {
        TT_INSN(binary_operation<Operation>(destination, lhs, rhs));
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
    static_assert(Index != hal::detail::DynamicGprIndex, "Scalar-unit GPR half write requires a compile-time GPR index: use hal::gpr<Index>()");
    static_assert(Index < hal::detail::GprCount, "Scalar-unit GPR index must be in [0, 63]");
    return Index * 2u + high;
}

template <bool High, std::uint32_t Index, std::uint32_t Value>
inline constexpr std::uint32_t set_half_operation(const hal::Gpr<Index> destination, const Immediate<Value> value)
{
    if constexpr (Index != hal::detail::DynamicGprIndex && Value != DynamicImmediateValue)
    {
        static_assert(Value < (1u << 16), "GPR half value must fit in 16 bits");
        return TT_OP_SETDMAREG(DMAReg::Payload::IMMEDIATE, Value, DMAReg::Mode::IMMEDIATE, half_index<Index>(High));
    }
    else
    {
        if constexpr (Value != DynamicImmediateValue)
        {
            static_assert(Value < (1u << 16), "GPR half value must fit in 16 bits");
        }
        validate_gpr(destination);
        const std::uint32_t half_value = OperandTraits<Immediate<Value>>::value(value);
        require_valid(half_value < (1u << 16), "GPR half value must fit in 16 bits");
        return TT_OP_SETDMAREG(DMAReg::Payload::IMMEDIATE, half_value, DMAReg::Mode::IMMEDIATE, OperandTraits<hal::Gpr<Index>>::value(destination) * 2u + High);
    }
}

template <BinaryOperation Operation, std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
class BinaryDescriptor
{
public:
    constexpr BinaryDescriptor(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs) : destination(destination), lhs(lhs), rhs(rhs)
    {
    }

    constexpr std::uint32_t operation() const
    {
        return binary_operation<Operation>(destination, lhs, rhs);
    }

    hal::Gpr<Destination> destination;
    hal::Gpr<Lhs> lhs;
    Rhs rhs;
};

} // namespace detail

/** @brief Set a GPR to a compile-time 32-bit value. */
template <std::uint32_t Value, std::uint32_t Index>
inline __attribute__((always_inline)) void set(const hal::Gpr<Index>)
{
    TTI_SETDMAREG(detail::DMAReg::Payload::IMMEDIATE, Value & 0xffffu, detail::DMAReg::Mode::IMMEDIATE, detail::half_index<Index>(false /*high*/));
    TTI_SETDMAREG(detail::DMAReg::Payload::IMMEDIATE, Value >> 16, detail::DMAReg::Mode::IMMEDIATE, detail::half_index<Index>(true /*high*/));
}

/** @brief Set a GPR to a runtime 32-bit value. */
template <std::uint32_t Index>
inline __attribute__((always_inline)) void set(const hal::Gpr<Index> destination, const std::uint32_t value)
{
    detail::validate_gpr(destination);
    const std::uint32_t index = detail::OperandTraits<hal::Gpr<Index>>::value(destination);
    TT_SETDMAREG(detail::DMAReg::Payload::IMMEDIATE, value & 0xffffu, detail::DMAReg::Mode::IMMEDIATE, index * 2u);
    TT_SETDMAREG(detail::DMAReg::Payload::IMMEDIATE, value >> 16, detail::DMAReg::Mode::IMMEDIATE, index * 2u + 1u);
}

/** @brief Set the low half of a GPR to a compile-time 16-bit value. */
template <std::uint32_t Value, std::uint32_t Index>
inline __attribute__((always_inline)) void set_low(const hal::Gpr<Index>)
{
    static_assert(Value < (1u << 16), "GPR half value must fit in 16 bits");
    TTI_SETDMAREG(detail::DMAReg::Payload::IMMEDIATE, Value, detail::DMAReg::Mode::IMMEDIATE, detail::half_index<Index>(false /*high*/));
}

/** @brief Set the low half of a GPR to a runtime 16-bit value. */
template <std::uint32_t Index>
inline __attribute__((always_inline)) void set_low(const hal::Gpr<Index> destination, const std::uint32_t value)
{
    TT_INSN(detail::set_half_operation<false /*High*/>(destination, immediate(value)));
}

/** @brief Set the high half of a GPR to a compile-time 16-bit value. */
template <std::uint32_t Value, std::uint32_t Index>
inline __attribute__((always_inline)) void set_high(const hal::Gpr<Index>)
{
    static_assert(Value < (1u << 16), "GPR half value must fit in 16 bits");
    TTI_SETDMAREG(detail::DMAReg::Payload::IMMEDIATE, Value, detail::DMAReg::Mode::IMMEDIATE, detail::half_index<Index>(true /*high*/));
}

/** @brief Set the high half of a GPR to a runtime 16-bit value. */
template <std::uint32_t Index>
inline __attribute__((always_inline)) void set_high(const hal::Gpr<Index> destination, const std::uint32_t value)
{
    TT_INSN(detail::set_half_operation<true /*High*/>(destination, immediate(value)));
}

/** @brief Add two operands with unsigned 32-bit wraparound. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void add(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::Add>(destination, lhs, rhs);
}

/** @brief Subtract the rhs from the lhs with unsigned 32-bit wraparound. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void subtract(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::Subtract>(destination, lhs, rhs);
}

/** @brief Multiply the low 16 bits of each operand and write the full 32-bit result. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void multiply_low_u16(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::MultiplyLowU16>(destination, lhs, rhs);
}

/** @brief Compute a bitwise AND. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void bit_and(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::BitAnd>(destination, lhs, rhs);
}

/** @brief Compute a bitwise OR. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void bit_or(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::BitOr>(destination, lhs, rhs);
}

/** @brief Compute a bitwise XOR. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void bit_xor(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::BitXor>(destination, lhs, rhs);
}

/** @brief Shift the lhs left logically by the rhs low five bits. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void shift_left(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::ShiftLeft>(destination, lhs, rhs);
}

/** @brief Shift the lhs right logically by the rhs low five bits. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void shift_right(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::BinaryOperation::ShiftRight>(destination, lhs, rhs);
}

/** @brief Compare two unsigned 32-bit operands and write zero or one. */
template <Compare Predicate, std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
inline __attribute__((always_inline)) void compare(const hal::Gpr<Destination> destination, const hal::Gpr<Lhs> lhs, const Rhs rhs)
{
    detail::emit_binary<detail::compare_operation<Predicate>>(destination, lhs, rhs);
}

// Descriptors encode one scalar-unit instruction. operation() is usable in constant
// expressions; invalid constants fail compilation and runtime operands use LLK_ASSERT.

/** @brief Encode a SETDMAREG write to the low half of one GPR without issuing it. */
template <std::uint32_t Index, std::uint32_t Value>
class SetLow
{
public:
    constexpr SetLow(const hal::Gpr<Index> destination, const Immediate<Value> value) : destination(destination), value(value)
    {
    }

    constexpr std::uint32_t operation() const
    {
        return detail::set_half_operation<false /*High*/>(destination, value);
    }

    hal::Gpr<Index> destination;
    Immediate<Value> value;
};

/** @brief Encode a SETDMAREG write to the high half of one GPR without issuing it. */
template <std::uint32_t Index, std::uint32_t Value>
class SetHigh
{
public:
    constexpr SetHigh(const hal::Gpr<Index> destination, const Immediate<Value> value) : destination(destination), value(value)
    {
    }

    constexpr std::uint32_t operation() const
    {
        return detail::set_half_operation<true /*High*/>(destination, value);
    }

    hal::Gpr<Index> destination;
    Immediate<Value> value;
};

/** @brief Encode ADDDMAREG without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
class Add : public detail::BinaryDescriptor<detail::BinaryOperation::Add, Destination, Lhs, Rhs>
{
public:
    using detail::BinaryDescriptor<detail::BinaryOperation::Add, Destination, Lhs, Rhs>::BinaryDescriptor;
};

template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
Add(hal::Gpr<Destination>, hal::Gpr<Lhs>, Rhs) -> Add<Destination, Lhs, Rhs>;

/** @brief Encode SUBDMAREG without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
class Subtract : public detail::BinaryDescriptor<detail::BinaryOperation::Subtract, Destination, Lhs, Rhs>
{
public:
    using detail::BinaryDescriptor<detail::BinaryOperation::Subtract, Destination, Lhs, Rhs>::BinaryDescriptor;
};

template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
Subtract(hal::Gpr<Destination>, hal::Gpr<Lhs>, Rhs) -> Subtract<Destination, Lhs, Rhs>;

/** @brief Encode MULDMAREG without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
class MultiplyLowU16 : public detail::BinaryDescriptor<detail::BinaryOperation::MultiplyLowU16, Destination, Lhs, Rhs>
{
public:
    using detail::BinaryDescriptor<detail::BinaryOperation::MultiplyLowU16, Destination, Lhs, Rhs>::BinaryDescriptor;
};

template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
MultiplyLowU16(hal::Gpr<Destination>, hal::Gpr<Lhs>, Rhs) -> MultiplyLowU16<Destination, Lhs, Rhs>;

/** @brief Encode a bitwise-AND BITWOPDMAREG without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
class BitAnd : public detail::BinaryDescriptor<detail::BinaryOperation::BitAnd, Destination, Lhs, Rhs>
{
public:
    using detail::BinaryDescriptor<detail::BinaryOperation::BitAnd, Destination, Lhs, Rhs>::BinaryDescriptor;
};

template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
BitAnd(hal::Gpr<Destination>, hal::Gpr<Lhs>, Rhs) -> BitAnd<Destination, Lhs, Rhs>;

/** @brief Encode a bitwise-OR BITWOPDMAREG without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
class BitOr : public detail::BinaryDescriptor<detail::BinaryOperation::BitOr, Destination, Lhs, Rhs>
{
public:
    using detail::BinaryDescriptor<detail::BinaryOperation::BitOr, Destination, Lhs, Rhs>::BinaryDescriptor;
};

template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
BitOr(hal::Gpr<Destination>, hal::Gpr<Lhs>, Rhs) -> BitOr<Destination, Lhs, Rhs>;

/** @brief Encode a bitwise-XOR BITWOPDMAREG without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
class BitXor : public detail::BinaryDescriptor<detail::BinaryOperation::BitXor, Destination, Lhs, Rhs>
{
public:
    using detail::BinaryDescriptor<detail::BinaryOperation::BitXor, Destination, Lhs, Rhs>::BinaryDescriptor;
};

template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
BitXor(hal::Gpr<Destination>, hal::Gpr<Lhs>, Rhs) -> BitXor<Destination, Lhs, Rhs>;

/** @brief Encode a left SHIFTDMAREG without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
class ShiftLeft : public detail::BinaryDescriptor<detail::BinaryOperation::ShiftLeft, Destination, Lhs, Rhs>
{
public:
    using detail::BinaryDescriptor<detail::BinaryOperation::ShiftLeft, Destination, Lhs, Rhs>::BinaryDescriptor;
};

template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
ShiftLeft(hal::Gpr<Destination>, hal::Gpr<Lhs>, Rhs) -> ShiftLeft<Destination, Lhs, Rhs>;

/** @brief Encode a right SHIFTDMAREG without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
class ShiftRight : public detail::BinaryDescriptor<detail::BinaryOperation::ShiftRight, Destination, Lhs, Rhs>
{
public:
    using detail::BinaryDescriptor<detail::BinaryOperation::ShiftRight, Destination, Lhs, Rhs>::BinaryDescriptor;
};

template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
ShiftRight(hal::Gpr<Destination>, hal::Gpr<Lhs>, Rhs) -> ShiftRight<Destination, Lhs, Rhs>;

/** @brief Encode an unsigned greater-than CMPDMAREG without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
class CompareGreaterThan : public detail::BinaryDescriptor<detail::BinaryOperation::CompareGreaterThan, Destination, Lhs, Rhs>
{
public:
    using detail::BinaryDescriptor<detail::BinaryOperation::CompareGreaterThan, Destination, Lhs, Rhs>::BinaryDescriptor;
};

template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
CompareGreaterThan(hal::Gpr<Destination>, hal::Gpr<Lhs>, Rhs) -> CompareGreaterThan<Destination, Lhs, Rhs>;

/** @brief Encode an unsigned less-than CMPDMAREG without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
class CompareLessThan : public detail::BinaryDescriptor<detail::BinaryOperation::CompareLessThan, Destination, Lhs, Rhs>
{
public:
    using detail::BinaryDescriptor<detail::BinaryOperation::CompareLessThan, Destination, Lhs, Rhs>::BinaryDescriptor;
};

template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
CompareLessThan(hal::Gpr<Destination>, hal::Gpr<Lhs>, Rhs) -> CompareLessThan<Destination, Lhs, Rhs>;

/** @brief Encode an unsigned equality CMPDMAREG without issuing it. */
template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
class CompareEqual : public detail::BinaryDescriptor<detail::BinaryOperation::CompareEqual, Destination, Lhs, Rhs>
{
public:
    using detail::BinaryDescriptor<detail::BinaryOperation::CompareEqual, Destination, Lhs, Rhs>::BinaryDescriptor;
};

template <std::uint32_t Destination, std::uint32_t Lhs, typename Rhs>
CompareEqual(hal::Gpr<Destination>, hal::Gpr<Lhs>, Rhs) -> CompareEqual<Destination, Lhs, Rhs>;

} // namespace hal::scalar
