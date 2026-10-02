// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"
#include "llk_memory_checks.h"
#include "sync.h"
#include "utils/gpr.h"

namespace hal::atomic
{
namespace detail
{
inline constexpr std::uint8_t WordsPerLine              = 4;
inline constexpr std::uint8_t CounterWidthMax           = 32;
inline constexpr std::uint8_t FifoCapacityLog2Max       = 15;
inline constexpr std::uint8_t FifoUpdateCapacityLog2Max = 14;
inline constexpr std::uint8_t IncrementLog2Max          = 15;
inline constexpr std::uint8_t CompareSetValueLimit      = 16;
inline constexpr std::uint8_t GprsPerQuad               = 4;
inline constexpr std::uint8_t LockFree                  = 0;
inline constexpr std::uint8_t LockHeld                  = 1;
} // namespace detail

/** @brief Select the hardware-managed FIFO pointer encoded by a low-level ATINCGETPTR descriptor. */
enum class FifoPointer : std::uint8_t
{
    Read  = 0,
    Write = 1
};

/** @brief Select the FIFO condition that blocks a readiness wait. */
enum class FifoState : std::uint8_t
{
    Empty,
    Full
};

/**
 * @brief Describe a posted atomic fetch-and-add on one 32-bit L1 word (ATINCGET).
 *
 * @tparam AddressIndex: Address GPR type index.
 * @tparam DataIndex: Addend and result GPR type index.
 */
template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
struct FetchIncrement
{
    hal::Gpr<AddressIndex> address;
    hal::Gpr<DataIndex> data;
    std::uint8_t word_select   = 0;
    std::uint8_t counter_width = detail::CounterWidthMax;

    /**
     * @brief Return the encoded ATINCGET instruction without issuing it.
     *
     * @note Invalid constant expressions are rejected. Enable LLK assertions to diagnose
     *       invalid runtime fields; disabled assertions add no runtime overhead.
     */
    [[nodiscard]] constexpr std::uint32_t operation() const;
};

/**
 * @brief Describe a blocking FIFO pointer update or state wait (ATINCGETPTR).
 *
 * @note Capacity logarithms in [0, 14] support pointer updates and readiness probes.
 *       Capacity logarithm 15 is supported only for readiness probes: its zero-width encoding does
 *       not advance pointers in the documented model, and Blackhole behavior is unconfirmed.
 *
 * @tparam AddressIndex: FIFOControl address GPR type index.
 * @tparam DataIndex: Pointer-result GPR type index.
 */
template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
struct FifoAcquire
{
    hal::Gpr<AddressIndex> address;
    hal::Gpr<DataIndex> data;
    FifoPointer pointer;
    std::uint8_t capacity_log2;
    std::uint8_t increment_log2 = 0;
    bool probe_only             = false;

    /**
     * @brief Return the encoded ATINCGETPTR instruction without issuing it.
     *
     * @note Invalid constant expressions are rejected. Enable LLK assertions to diagnose
     *       invalid runtime fields; disabled assertions add no runtime overhead.
     */
    [[nodiscard]] constexpr std::uint32_t operation() const;
};

/**
 * @brief Describe a blocking whole-word compare-and-set (ATCAS).
 *
 * @tparam AddressIndex: Address GPR type index.
 */
template <std::uint32_t AddressIndex>
struct CompareSet
{
    hal::Gpr<AddressIndex> address;
    std::uint8_t compare;
    std::uint8_t set;
    std::uint8_t word_select;

    /**
     * @brief Select the required compare/set values and an optional word within the line.
     *
     * @param address: GPR prepared by @ref set_address.
     * @param compare: Whole-word comparison value, in [0, 15].
     * @param set: Whole-word replacement value, in [0, 15].
     * @param word_select: Word within the line, in [0, 3].
     */
    constexpr CompareSet(const hal::Gpr<AddressIndex> address, const std::uint8_t compare, const std::uint8_t set, const std::uint8_t word_select = 0) :
        address(address), compare(compare), set(set), word_select(word_select)
    {
    }

    /**
     * @brief Return the encoded ATCAS instruction without issuing it.
     *
     * @note Invalid constant expressions are rejected. Enable LLK assertions to diagnose
     *       invalid runtime fields; disabled assertions add no runtime overhead.
     */
    [[nodiscard]] constexpr std::uint32_t operation() const;
};

/**
 * @brief Describe a posted masked 16-byte L1 write from a four-aligned GPR quad (ATSWAP).
 *
 * @tparam AddressIndex: Address GPR type index.
 * @tparam DataIndex: Four-aligned source-quad base GPR type index.
 */
template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
struct MaskedWrite
{
    hal::Gpr<AddressIndex> address;
    hal::Gpr<DataIndex> data;
    std::uint8_t granule_mask;

    /**
     * @brief Return the encoded ATSWAP instruction without issuing it.
     *
     * @note Invalid constant expressions are rejected. Enable LLK assertions to diagnose
     *       invalid runtime fields; disabled assertions add no runtime overhead.
     */
    [[nodiscard]] constexpr std::uint32_t operation() const;
};

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
FetchIncrement(hal::Gpr<AddressIndex>, hal::Gpr<DataIndex>, std::uint8_t = 0, std::uint8_t = detail::CounterWidthMax)
    -> FetchIncrement<AddressIndex, DataIndex>;

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
FifoAcquire(hal::Gpr<AddressIndex>, hal::Gpr<DataIndex>, FifoPointer, std::uint8_t, std::uint8_t = 0, bool = false) -> FifoAcquire<AddressIndex, DataIndex>;

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
MaskedWrite(hal::Gpr<AddressIndex>, hal::Gpr<DataIndex>, std::uint8_t) -> MaskedWrite<AddressIndex, DataIndex>;

/**
 * @brief Load an atomic address GPR from an L1 byte address.
 *
 * @tparam AddressIndex: Address GPR type index.
 * @param address: GPR to receive the hardware line address.
 * @param l1_byte_address: Byte address of a 16-byte-aligned L1 line.
 * @note Call before issuing an atomic operation or executing its MOP encoding. Atomics
 *       use an unbiased line address, so this setup loads L1_ADDRESS(l1_byte_address) + 1.
 */
template <std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void set_address(hal::Gpr<AddressIndex> address, std::uint32_t l1_byte_address);

/**
 * @brief Access a wrapping counter in one 32-bit word of a 16-byte L1 line.
 *
 * @tparam WidthBits: Counter width in bits, in [1, 32].
 * @tparam AddressIndex: Address GPR type index.
 * @note Prepare the address GPR with @ref set_address before execution, including MOP use.
 */
template <std::uint8_t WidthBits, std::uint32_t AddressIndex>
struct Counter
{
    /** @brief Encode counter operations without issuing instructions. */
    struct Operations
    {
        hal::Gpr<AddressIndex> address;
        std::uint8_t word;

        /**
         * @brief Encode a fetch-and-add for MOP use without issuing it.
         *
         * @tparam DataIndex: Addend/result GPR type index.
         * @param data: GPR supplying the addend and receiving the pre-increment value.
         */
        template <std::uint32_t DataIndex>
        [[nodiscard]] constexpr std::uint32_t fetch_add(hal::Gpr<DataIndex> data) const;
    };

    hal::Gpr<AddressIndex> address;
    std::uint8_t word = 0;

    /**
     * @brief Set the counter's L1 byte address.
     *
     * @param l1_byte_address: Byte address of a 16-byte-aligned L1 line.
     */
    inline __attribute__((always_inline)) void set_address(std::uint32_t l1_byte_address) const;

    /** @brief Return an encoding view holding a copy of this counter's operands. */
    [[nodiscard]] constexpr Operations operation() const
    {
        return {address, word};
    }

    /**
     * @brief Issue a posted atomic fetch-and-add.
     *
     * @tparam DataIndex: Addend/result GPR type index.
     * @param data: GPR supplying the addend and receiving the pre-increment value.
     * @note Drain ScalarIdle while blocking the dependent consumer before consuming or reusing data.
     */
    template <std::uint32_t DataIndex>
    inline __attribute__((always_inline)) void fetch_add(hal::Gpr<DataIndex> data) const;
};

/**
 * @brief Identify a counter, deducing the address GPR index.
 *
 * @tparam WidthBits: Counter width in bits, in [1, 32].
 * @tparam AddressIndex: Deduced address GPR type index.
 * @param address: Address GPR to prepare with Counter::set_address.
 * @param word: Word within the line, in [0, 3].
 */
template <std::uint8_t WidthBits, std::uint32_t AddressIndex>
inline constexpr Counter<WidthBits, AddressIndex> counter(const hal::Gpr<AddressIndex> address, const std::uint8_t word = 0)
{
    return {address, word};
}

/**
 * @brief Access hardware-managed FIFO counters in one 16-byte L1 line.
 *
 * @tparam CapacityLog2: Base-two logarithm of capacity, in [0, 15]; 15 supports waits only.
 * @tparam AddressIndex: FIFOControl address GPR type index.
 * @note Prepare the address GPR with @ref set_address before execution, including MOP use.
 *       Results are raw counters including the wrap bit; mask with (1u << CapacityLog2) - 1
 *       when deriving a slot index. Move payload separately from the counter operations.
 * @note Establish that the full batch is available and serialize callers advancing the same
 *       endpoint for multi-slot operations. With LLK assertions enabled, pop/push synchronize
 *       with earlier work and check the available batch before advancing. MOP encodings contain
 *       no live-state check; establish batch availability before executing them.
 */
template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
struct Fifo
{
    /** @brief Encode FIFO operations without issuing instructions or checking live counters. */
    struct Operations
    {
        hal::Gpr<AddressIndex> control_address;

        /**
         * @brief Encode a read-counter advance for MOP use without issuing it.
         *
         * @tparam IncrementLog2: Logarithm of the increment, in [0, CapacityLog2]; zero means one slot.
         * @tparam ResultIndex: Result GPR type index.
         * @param result: GPR receiving the raw pre-increment read counter.
         */
        template <std::uint8_t IncrementLog2 = 0, std::uint32_t ResultIndex>
        [[nodiscard]] constexpr std::uint32_t pop(hal::Gpr<ResultIndex> result) const;

        /**
         * @brief Encode a runtime-increment read-counter advance without issuing it.
         *
         * @tparam ResultIndex: Result GPR type index.
         * @param result: GPR receiving the raw pre-increment read counter.
         * @param increment_log2: Logarithm of the increment, in [0, CapacityLog2].
         */
        template <std::uint32_t ResultIndex>
        [[nodiscard]] constexpr std::uint32_t pop(hal::Gpr<ResultIndex> result, std::uint8_t increment_log2) const;

        /**
         * @brief Encode a write-counter advance for MOP use without issuing it.
         *
         * @tparam IncrementLog2: Logarithm of the increment, in [0, CapacityLog2]; zero means one slot.
         * @tparam ResultIndex: Result GPR type index.
         * @param result: GPR receiving the raw pre-increment write counter.
         */
        template <std::uint8_t IncrementLog2 = 0, std::uint32_t ResultIndex>
        [[nodiscard]] constexpr std::uint32_t push(hal::Gpr<ResultIndex> result) const;

        /**
         * @brief Encode a runtime-increment write-counter advance without issuing it.
         *
         * @tparam ResultIndex: Result GPR type index.
         * @param result: GPR receiving the raw pre-increment write counter.
         * @param increment_log2: Logarithm of the increment, in [0, CapacityLog2].
         */
        template <std::uint32_t ResultIndex>
        [[nodiscard]] constexpr std::uint32_t push(hal::Gpr<ResultIndex> result, std::uint8_t increment_log2) const;

        /**
         * @brief Encode a FIFO-state wait for MOP use without issuing it.
         *
         * @tparam State: Blocking condition, values = <Empty/Full>.
         * @tparam ResultIndex: Result GPR type index.
         * @param result: GPR receiving the raw read counter for Empty or write counter for Full.
         */
        template <FifoState State, std::uint32_t ResultIndex>
        [[nodiscard]] constexpr std::uint32_t wait_while(hal::Gpr<ResultIndex> result) const;
    };

    hal::Gpr<AddressIndex> control_address;

    /**
     * @brief Set the control line's L1 byte address.
     *
     * @param l1_byte_address: Byte address of a 16-byte-aligned L1 line.
     */
    inline __attribute__((always_inline)) void set_address(std::uint32_t l1_byte_address) const;

    /** @brief Return an encoding view holding a copy of this FIFO's operands. */
    [[nodiscard]] constexpr Operations operation() const
    {
        return {control_address};
    }

    /**
     * @brief Issue a blocking read-counter advance.
     *
     * @tparam IncrementLog2: Logarithm of the increment, in [0, CapacityLog2]; zero means one slot.
     * @tparam ResultIndex: Result GPR type index.
     * @param result: GPR receiving the raw pre-increment read counter.
     * @note Establish that the complete batch is available before advancing by more than one slot.
     */
    template <std::uint8_t IncrementLog2 = 0, std::uint32_t ResultIndex>
    inline __attribute__((always_inline)) void pop(hal::Gpr<ResultIndex> result) const;

    /**
     * @brief Issue a blocking read-counter advance with a runtime increment.
     *
     * @tparam ResultIndex: Result GPR type index.
     * @param result: GPR receiving the raw pre-increment read counter.
     * @param increment_log2: Logarithm of the increment, in [0, CapacityLog2].
     * @note Establish that the complete batch is available before advancing by more than one slot.
     */
    template <std::uint32_t ResultIndex>
    inline __attribute__((always_inline)) void pop(hal::Gpr<ResultIndex> result, std::uint8_t increment_log2) const;

    /**
     * @brief Issue a blocking write-counter advance.
     *
     * @tparam IncrementLog2: Logarithm of the increment, in [0, CapacityLog2]; zero means one slot.
     * @tparam ResultIndex: Result GPR type index.
     * @param result: GPR receiving the raw pre-increment write counter.
     * @note Establish that the complete batch fits before advancing by more than one slot.
     */
    template <std::uint8_t IncrementLog2 = 0, std::uint32_t ResultIndex>
    inline __attribute__((always_inline)) void push(hal::Gpr<ResultIndex> result) const;

    /**
     * @brief Issue a blocking write-counter advance with a runtime increment.
     *
     * @tparam ResultIndex: Result GPR type index.
     * @param result: GPR receiving the raw pre-increment write counter.
     * @param increment_log2: Logarithm of the increment, in [0, CapacityLog2].
     * @note Establish that the complete batch fits before advancing by more than one slot.
     */
    template <std::uint32_t ResultIndex>
    inline __attribute__((always_inline)) void push(hal::Gpr<ResultIndex> result, std::uint8_t increment_log2) const;

    /**
     * @brief Issue a blocking FIFO-state wait without advancing either counter.
     *
     * @tparam State: Blocking condition, values = <Empty/Full>.
     * @tparam ResultIndex: Result GPR type index.
     * @param result: GPR receiving the raw read counter for Empty or write counter for Full.
     */
    template <FifoState State, std::uint32_t ResultIndex>
    inline __attribute__((always_inline)) void wait_while(hal::Gpr<ResultIndex> result) const;
};

/**
 * @brief Identify a FIFO, deducing the address GPR index.
 *
 * @tparam CapacityLog2: Base-two logarithm of capacity, in [0, 15]; 15 supports waits only.
 * @tparam AddressIndex: Deduced address GPR type index.
 * @param control_address: Address GPR to prepare with Fifo::set_address.
 */
template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
inline constexpr Fifo<CapacityLog2, AddressIndex> fifo(const hal::Gpr<AddressIndex> control_address)
{
    return {control_address};
}

/**
 * @brief Access a lock or flag in one 32-bit word of a 16-byte L1 line.
 *
 * @tparam AddressIndex: Address GPR type index.
 * @note Prepare the address GPR with @ref set_address before execution, including MOP use.
 */
template <std::uint32_t AddressIndex>
struct Lock
{
    /** @brief Encode lock operations without issuing instructions. */
    struct Operations
    {
        hal::Gpr<AddressIndex> address;
        std::uint8_t word;

        /** @brief Encode acquire (free to held) for MOP use without issuing it. */
        [[nodiscard]] constexpr std::uint32_t acquire() const;
        /** @brief Encode release (held to free) for MOP use without issuing it. */
        [[nodiscard]] constexpr std::uint32_t release() const;

        /**
         * @brief Encode a compare-and-set for MOP use without issuing it.
         *
         * @tparam Compare: Whole-word comparison value, in [0, 15].
         * @tparam Set: Whole-word replacement value, in [0, 15].
         */
        template <std::uint8_t Compare, std::uint8_t Set>
        [[nodiscard]] constexpr std::uint32_t compare_set() const;

        /**
         * @brief Encode a runtime-value compare-and-set without issuing it.
         *
         * @param compare: Whole-word comparison value, in [0, 15].
         * @param set: Whole-word replacement value, in [0, 15].
         */
        [[nodiscard]] constexpr std::uint32_t compare_set(std::uint8_t compare, std::uint8_t set) const;
    };

    hal::Gpr<AddressIndex> address;
    std::uint8_t word = 0;

    /**
     * @brief Set the lock's L1 byte address.
     *
     * @param l1_byte_address: Byte address of a 16-byte-aligned L1 line.
     */
    inline __attribute__((always_inline)) void set_address(std::uint32_t l1_byte_address) const;

    /** @brief Return an encoding view holding a copy of this lock's operands. */
    [[nodiscard]] constexpr Operations operation() const
    {
        return {address, word};
    }

    /** @brief Issue a blocking acquire (free to held). */
    inline __attribute__((always_inline)) void acquire() const;
    /** @brief Issue a blocking release (held to free). */
    inline __attribute__((always_inline)) void release() const;

    /**
     * @brief Issue a blocking compare-and-set with compile-time comparison/replacement values.
     *
     * @tparam Compare: Whole-word comparison value, in [0, 15].
     * @tparam Set: Whole-word replacement value, in [0, 15].
     */
    template <std::uint8_t Compare, std::uint8_t Set>
    inline __attribute__((always_inline)) void compare_set() const;

    /**
     * @brief Issue a blocking compare-and-set with runtime comparison/replacement values.
     *
     * @param compare: Whole-word comparison value, in [0, 15].
     * @param set: Whole-word replacement value, in [0, 15].
     */
    inline __attribute__((always_inline)) void compare_set(std::uint8_t compare, std::uint8_t set) const;
};

template <std::uint32_t AddressIndex>
Lock(hal::Gpr<AddressIndex>, std::uint8_t = 0) -> Lock<AddressIndex>;

namespace detail
{
inline constexpr std::uint32_t L1MemoryEncoding = 0;

template <std::uint32_t Index>
constexpr bool is_valid(const hal::Gpr<Index> operand)
{
    return operand.index < hal::detail::GprCount;
}

constexpr bool is_valid(const FifoPointer pointer)
{
    return pointer == FifoPointer::Read || pointer == FifoPointer::Write;
}

template <FifoState State>
constexpr FifoPointer pointer_for()
{
    static_assert(State == FifoState::Empty || State == FifoState::Full, "FIFO state must be Empty or Full");
    return State == FifoState::Empty ? FifoPointer::Read : FifoPointer::Write;
}

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline __attribute__((always_inline)) void assert_valid(const FetchIncrement<AddressIndex, DataIndex> operation)
{
    LLK_ASSERT(is_valid(operation.address), "ATINCGET address GPR index must be in [0, 63]");
    LLK_ASSERT(is_valid(operation.data), "ATINCGET data GPR index must be in [0, 63]");
    LLK_ASSERT(operation.word_select < detail::WordsPerLine, "ATINCGET word selection must be in [0, 3]");
    LLK_ASSERT(operation.counter_width >= 1u && operation.counter_width <= detail::CounterWidthMax, "ATINCGET counter width must be in [1, 32]");
}

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline __attribute__((always_inline)) void assert_valid(const FifoAcquire<AddressIndex, DataIndex> operation)
{
    LLK_ASSERT(is_valid(operation.address), "ATINCGETPTR address GPR index must be in [0, 63]");
    LLK_ASSERT(is_valid(operation.data), "ATINCGETPTR result GPR index must be in [0, 63]");
    LLK_ASSERT(is_valid(operation.pointer), "ATINCGETPTR pointer must be Read or Write");
    LLK_ASSERT(
        operation.capacity_log2 <= detail::FifoCapacityLog2Max && (operation.probe_only || operation.capacity_log2 <= detail::FifoUpdateCapacityLog2Max),
        "ATINCGETPTR FIFO capacity logarithm must be in [0, 14] for pointer updates or [0, 15] for readiness probes");
    LLK_ASSERT(
        operation.increment_log2 <= detail::IncrementLog2Max && operation.increment_log2 <= operation.capacity_log2,
        "ATINCGETPTR increment logarithm must not exceed the FIFO capacity logarithm");
    LLK_ASSERT(!operation.probe_only || operation.increment_log2 == 0u, "ATINCGETPTR readiness probes cannot increment the pointer");
}

template <std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void assert_valid(const CompareSet<AddressIndex> operation)
{
    LLK_ASSERT(is_valid(operation.address), "ATCAS address GPR index must be in [0, 63]");
    LLK_ASSERT(operation.word_select < detail::WordsPerLine, "ATCAS word selection must be in [0, 3]");
    LLK_ASSERT(operation.compare < detail::CompareSetValueLimit, "ATCAS compare value must fit in four bits");
    LLK_ASSERT(operation.set < detail::CompareSetValueLimit, "ATCAS replacement value must fit in four bits");
}

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline __attribute__((always_inline)) void assert_valid(const MaskedWrite<AddressIndex, DataIndex> operation)
{
    LLK_ASSERT(is_valid(operation.address), "ATSWAP address GPR index must be in [0, 63]");
    LLK_ASSERT(is_valid(operation.data), "ATSWAP data GPR index must be in [0, 63]");
    LLK_ASSERT(operation.data.index % detail::GprsPerQuad == 0u, "ATSWAP data GPR must be four-aligned");
}

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t encode(const FetchIncrement<AddressIndex, DataIndex> operation)
{
    return TT_OP_ATINCGET(L1MemoryEncoding, operation.counter_width - 1u, operation.word_select, operation.data.index, operation.address.index);
}

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t encode(const FifoAcquire<AddressIndex, DataIndex> operation)
{
    return TT_OP_ATINCGETPTR(
        L1MemoryEncoding,
        operation.probe_only,
        operation.increment_log2,
        (operation.capacity_log2 + 1u) & detail::FifoCapacityLog2Max,
        static_cast<std::uint32_t>(operation.pointer),
        operation.data.index,
        operation.address.index);
}

template <std::uint32_t AddressIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t encode(const CompareSet<AddressIndex> operation)
{
    return TT_OP_ATCAS(L1MemoryEncoding, operation.set, operation.compare, operation.word_select, 0 /*DataRegIndex: unused by ATCAS*/, operation.address.index);
}

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t encode(const MaskedWrite<AddressIndex, DataIndex> operation)
{
    return TT_OP_ATSWAP(L1MemoryEncoding, operation.granule_mask, operation.data.index, operation.address.index);
}

template <typename Descriptor>
inline constexpr __attribute__((always_inline)) std::uint32_t checked_encode(const Descriptor& descriptor)
{
    if (__builtin_is_constant_evaluated() && !is_valid(descriptor))
    {
        __builtin_trap();
    }
    if (!__builtin_is_constant_evaluated())
    {
        assert_valid(descriptor);
    }
    return encode(descriptor);
}

// Object methods issue known encodings immediately and runtime encodings through the buffer.
inline __attribute__((always_inline)) void issue(const std::uint32_t encoding)
{
    if (__builtin_constant_p(encoding))
    {
        TTI_INSN(encoding);
    }
    else
    {
        TT_INSN(encoding);
    }
}
} // namespace detail

/** @brief Return whether a fetch-increment descriptor can be encoded without truncation. */
template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
constexpr bool is_valid(const FetchIncrement<AddressIndex, DataIndex> operation)
{
    return detail::is_valid(operation.address) && detail::is_valid(operation.data) && operation.word_select < detail::WordsPerLine &&
           operation.counter_width >= 1u && operation.counter_width <= detail::CounterWidthMax;
}

/** @brief Return whether a FIFO descriptor has supported capacity, pointer-update semantics, and field values. */
template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
constexpr bool is_valid(const FifoAcquire<AddressIndex, DataIndex> operation)
{
    return detail::is_valid(operation.address) && detail::is_valid(operation.data) && detail::is_valid(operation.pointer) &&
           operation.capacity_log2 <= detail::FifoCapacityLog2Max && (operation.probe_only || operation.capacity_log2 <= detail::FifoUpdateCapacityLog2Max) &&
           operation.increment_log2 <= detail::IncrementLog2Max && operation.increment_log2 <= operation.capacity_log2 &&
           (!operation.probe_only || operation.increment_log2 == 0u);
}

/** @brief Return whether a compare-set descriptor can be encoded without truncation. */
template <std::uint32_t AddressIndex>
constexpr bool is_valid(const CompareSet<AddressIndex> operation)
{
    return detail::is_valid(operation.address) && operation.word_select < detail::WordsPerLine && operation.compare < detail::CompareSetValueLimit &&
           operation.set < detail::CompareSetValueLimit;
}

/** @brief Return whether a masked-write descriptor can be encoded without truncation. */
template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
constexpr bool is_valid(const MaskedWrite<AddressIndex, DataIndex> operation)
{
    return detail::is_valid(operation.address) && detail::is_valid(operation.data) && operation.data.index % detail::GprsPerQuad == 0u;
}

/** @brief Return whether a counter identity has a valid layout and address GPR. */
template <std::uint8_t WidthBits, std::uint32_t AddressIndex>
constexpr bool is_valid(const Counter<WidthBits, AddressIndex> counter)
{
    return WidthBits >= 1u && WidthBits <= detail::CounterWidthMax && detail::is_valid(counter.address) && counter.word < detail::WordsPerLine;
}

/** @brief Return whether a FIFO identity has a valid capacity and address GPR. */
template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
constexpr bool is_valid(const Fifo<CapacityLog2, AddressIndex> fifo)
{
    return CapacityLog2 <= detail::FifoCapacityLog2Max && detail::is_valid(fifo.control_address);
}

/** @brief Return whether a lock identity has a valid word and address GPR. */
template <std::uint32_t AddressIndex>
constexpr bool is_valid(const Lock<AddressIndex> lock)
{
    return detail::is_valid(lock.address) && lock.word < detail::WordsPerLine;
}

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t FetchIncrement<AddressIndex, DataIndex>::operation() const
{
    return detail::checked_encode(*this);
}

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t FifoAcquire<AddressIndex, DataIndex>::operation() const
{
    return detail::checked_encode(*this);
}

template <std::uint32_t AddressIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t CompareSet<AddressIndex>::operation() const
{
    return detail::checked_encode(*this);
}

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t MaskedWrite<AddressIndex, DataIndex>::operation() const
{
    return detail::checked_encode(*this);
}

template <std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void set_address(const hal::Gpr<AddressIndex> address, const std::uint32_t l1_byte_address)
{
    LLK_ASSERT(detail::is_valid(address), "Atomic address GPR index must be in [0, 63]");
    LLK_ASSERT(l1_byte_address % 16u == 0u, "Atomic L1 byte address must be 16-byte aligned");
    const std::uint32_t line_address = L1_ADDRESS(l1_byte_address) + 1u;
    detail::issue(TT_OP_SETDMAREG(0 /*Payload_SigSelSize*/, line_address & 0xffffu, 0 /*SetSignalsMode*/, address.index * 2u));
    detail::issue(TT_OP_SETDMAREG(0 /*Payload_SigSelSize*/, line_address >> 16, 0 /*SetSignalsMode*/, address.index * 2u + 1u));
}

namespace detail
{
template <std::uint8_t CapacityLog2, FifoPointer Pointer, std::uint32_t AddressIndex, std::uint32_t ResultIndex>
inline constexpr __attribute__((always_inline)) FifoAcquire<AddressIndex, ResultIndex> fifo_update_descriptor(
    const hal::Gpr<AddressIndex> address, const hal::Gpr<ResultIndex> result, const std::uint8_t increment_log2)
{
    return {address, result, Pointer, CapacityLog2, increment_log2, false /*probe_only*/};
}

template <std::uint8_t CapacityLog2, FifoPointer Pointer, std::uint8_t IncrementLog2, std::uint32_t AddressIndex, std::uint32_t ResultIndex>
inline constexpr __attribute__((always_inline)) FifoAcquire<AddressIndex, ResultIndex> fifo_update_descriptor(
    const hal::Gpr<AddressIndex> address, const hal::Gpr<ResultIndex> result)
{
    static_assert(IncrementLog2 <= IncrementLog2Max, "FIFO increment logarithm must be in [0, 15]");
    static_assert(IncrementLog2 <= CapacityLog2, "FIFO increment logarithm must not exceed the FIFO capacity logarithm");
    return fifo_update_descriptor<CapacityLog2, Pointer>(address, result, IncrementLog2);
}

// A single producer/consumer (or externally serialized endpoint) keeps a successful
// snapshot check valid until the update: the opposite endpoint can only add availability.
// MOP encodings cannot perform this CPU-side diagnostic on each execution.
// Evaluate only inside LLK_ASSERT so disabled assertions perform no synchronization or reads.
template <std::uint32_t AddressIndex, std::uint32_t ResultIndex>
inline __attribute__((always_inline)) bool is_valid_fifo_batch(const FifoAcquire<AddressIndex, ResultIndex> operation)
{
    if (operation.increment_log2 == 0u || !atomic::is_valid(operation))
    {
        return true;
    }

    // Complete earlier scalar work before reading the address through the RISC GPR window.
    sync::wait::stall<sync::StallTarget::Scalar, sync::StallCondition::ScalarIdle>();
    ckernel::tensix_sync();
    const std::uint32_t line_address = ckernel::regfile[operation.address.index];

    // Blackhole's noncoherent L0 must be flushed before observing externally updated L1.
    asm volatile("fence" ::: "memory");
    const volatile std::uint32_t* const control = reinterpret_cast<const volatile std::uint32_t*>(line_address << 4);
    const std::uint32_t read                    = control[0];
    const std::uint32_t write                   = control[1];
    const std::uint32_t capacity                = 1u << operation.capacity_log2;
    const std::uint32_t pointer_mask            = (capacity << 1) - 1u;
    const std::uint32_t used                    = (write - read) & pointer_mask;
    const std::uint32_t increment               = 1u << operation.increment_log2;
    return ((read | write) & ~pointer_mask) == 0u && used <= capacity &&
           (operation.pointer == FifoPointer::Read ? increment <= used : increment <= capacity - used);
}

template <std::uint32_t AddressIndex, std::uint32_t ResultIndex>
inline __attribute__((always_inline)) void issue_fifo_update(const FifoAcquire<AddressIndex, ResultIndex> descriptor)
{
    const std::uint32_t encoding = descriptor.operation();
    LLK_ASSERT(
        is_valid_fifo_batch(descriptor),
        descriptor.pointer == FifoPointer::Read ? "FIFO pop requires valid counters and enough occupied slots"
                                                : "FIFO push requires valid counters and enough free slots");
    issue(encoding);
}
} // namespace detail

template <std::uint8_t WidthBits, std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void Counter<WidthBits, AddressIndex>::set_address(const std::uint32_t l1_byte_address) const
{
    atomic::set_address(address, l1_byte_address);
}

template <std::uint8_t WidthBits, std::uint32_t AddressIndex>
template <std::uint32_t DataIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t Counter<WidthBits, AddressIndex>::Operations::fetch_add(const hal::Gpr<DataIndex> data) const
{
    return FetchIncrement<AddressIndex, DataIndex> {address, data, word, WidthBits}.operation();
}

template <std::uint8_t WidthBits, std::uint32_t AddressIndex>
template <std::uint32_t DataIndex>
inline __attribute__((always_inline)) void Counter<WidthBits, AddressIndex>::fetch_add(const hal::Gpr<DataIndex> data) const
{
    detail::issue(operation().fetch_add(data));
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void Fifo<CapacityLog2, AddressIndex>::set_address(const std::uint32_t l1_byte_address) const
{
    atomic::set_address(control_address, l1_byte_address);
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <std::uint8_t IncrementLog2, std::uint32_t ResultIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t Fifo<CapacityLog2, AddressIndex>::Operations::pop(const hal::Gpr<ResultIndex> result) const
{
    return detail::fifo_update_descriptor<CapacityLog2, FifoPointer::Read, IncrementLog2>(control_address, result).operation();
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <std::uint32_t ResultIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t Fifo<CapacityLog2, AddressIndex>::Operations::pop(
    const hal::Gpr<ResultIndex> result, const std::uint8_t increment_log2) const
{
    return detail::fifo_update_descriptor<CapacityLog2, FifoPointer::Read>(control_address, result, increment_log2).operation();
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <std::uint8_t IncrementLog2, std::uint32_t ResultIndex>
inline __attribute__((always_inline)) void Fifo<CapacityLog2, AddressIndex>::pop(const hal::Gpr<ResultIndex> result) const
{
    detail::issue_fifo_update(detail::fifo_update_descriptor<CapacityLog2, FifoPointer::Read, IncrementLog2>(control_address, result));
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <std::uint32_t ResultIndex>
inline __attribute__((always_inline)) void Fifo<CapacityLog2, AddressIndex>::pop(const hal::Gpr<ResultIndex> result, const std::uint8_t increment_log2) const
{
    detail::issue_fifo_update(detail::fifo_update_descriptor<CapacityLog2, FifoPointer::Read>(control_address, result, increment_log2));
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <std::uint8_t IncrementLog2, std::uint32_t ResultIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t Fifo<CapacityLog2, AddressIndex>::Operations::push(const hal::Gpr<ResultIndex> result) const
{
    return detail::fifo_update_descriptor<CapacityLog2, FifoPointer::Write, IncrementLog2>(control_address, result).operation();
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <std::uint32_t ResultIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t Fifo<CapacityLog2, AddressIndex>::Operations::push(
    const hal::Gpr<ResultIndex> result, const std::uint8_t increment_log2) const
{
    return detail::fifo_update_descriptor<CapacityLog2, FifoPointer::Write>(control_address, result, increment_log2).operation();
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <std::uint8_t IncrementLog2, std::uint32_t ResultIndex>
inline __attribute__((always_inline)) void Fifo<CapacityLog2, AddressIndex>::push(const hal::Gpr<ResultIndex> result) const
{
    detail::issue_fifo_update(detail::fifo_update_descriptor<CapacityLog2, FifoPointer::Write, IncrementLog2>(control_address, result));
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <std::uint32_t ResultIndex>
inline __attribute__((always_inline)) void Fifo<CapacityLog2, AddressIndex>::push(const hal::Gpr<ResultIndex> result, const std::uint8_t increment_log2) const
{
    detail::issue_fifo_update(detail::fifo_update_descriptor<CapacityLog2, FifoPointer::Write>(control_address, result, increment_log2));
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <FifoState State, std::uint32_t ResultIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t Fifo<CapacityLog2, AddressIndex>::Operations::wait_while(const hal::Gpr<ResultIndex> result) const
{
    return FifoAcquire<AddressIndex, ResultIndex> {
        control_address, result, detail::pointer_for<State>(), CapacityLog2, 0 /*increment_log2*/, true /*probe_only*/}
        .operation();
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <FifoState State, std::uint32_t ResultIndex>
inline __attribute__((always_inline)) void Fifo<CapacityLog2, AddressIndex>::wait_while(const hal::Gpr<ResultIndex> result) const
{
    detail::issue(operation().template wait_while<State>(result));
}

template <std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void Lock<AddressIndex>::set_address(const std::uint32_t l1_byte_address) const
{
    atomic::set_address(address, l1_byte_address);
}

template <std::uint32_t AddressIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t Lock<AddressIndex>::Operations::acquire() const
{
    return compare_set<detail::LockFree, detail::LockHeld>();
}

template <std::uint32_t AddressIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t Lock<AddressIndex>::Operations::release() const
{
    return compare_set<detail::LockHeld, detail::LockFree>();
}

template <std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void Lock<AddressIndex>::acquire() const
{
    detail::issue(operation().acquire());
}

template <std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void Lock<AddressIndex>::release() const
{
    detail::issue(operation().release());
}

template <std::uint32_t AddressIndex>
template <std::uint8_t Compare, std::uint8_t Set>
inline constexpr __attribute__((always_inline)) std::uint32_t Lock<AddressIndex>::Operations::compare_set() const
{
    static_assert(Compare < detail::CompareSetValueLimit, "ATCAS compare value must fit in four bits");
    static_assert(Set < detail::CompareSetValueLimit, "ATCAS replacement value must fit in four bits");
    return compare_set(Compare, Set);
}

template <std::uint32_t AddressIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t Lock<AddressIndex>::Operations::compare_set(
    const std::uint8_t compare, const std::uint8_t set) const
{
    return CompareSet<AddressIndex> {address, compare, set, word}.operation();
}

template <std::uint32_t AddressIndex>
template <std::uint8_t Compare, std::uint8_t Set>
inline __attribute__((always_inline)) void Lock<AddressIndex>::compare_set() const
{
    detail::issue(operation().template compare_set<Compare, Set>());
}

template <std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void Lock<AddressIndex>::compare_set(const std::uint8_t compare, const std::uint8_t set) const
{
    detail::issue(operation().compare_set(compare, set));
}

/**
 * @brief Issue a compile-time posted masked line store as one immediate ATSWAP.
 *
 * @tparam GranuleMask: Eight 16-bit line-granule write enables.
 * @tparam AddressIndex: Index of the GPR prepared by atomic::set_address with an L1 byte address.
 * @tparam DataIndex: Four-aligned index of the first of four consecutive source GPRs.
 * @note Drain ScalarIdle while blocking the dependent consumer before relying on the
 *       updated memory.
 */
template <std::uint8_t GranuleMask, std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline __attribute__((always_inline)) void store_masked(const hal::Gpr<AddressIndex>, const hal::Gpr<DataIndex>)
{
    static_assert(AddressIndex != hal::detail::DynamicGprIndex, "immediate masked write requires hal::gpr<Index>() for its address");
    static_assert(DataIndex != hal::detail::DynamicGprIndex, "immediate masked write requires hal::gpr<Index>() for its data quad");
    constexpr MaskedWrite<AddressIndex, DataIndex> descriptor {hal::Gpr<AddressIndex> {}, hal::Gpr<DataIndex> {}, GranuleMask};
    static_assert(is_valid(descriptor), "invalid atomic descriptor");
    constexpr std::uint32_t encoding = descriptor.operation();
    TTI_INSN(encoding);
}

/**
 * @brief Encode and issue a value-selected posted masked line store.
 *
 * @tparam AddressIndex: Address GPR type index.
 * @tparam DataIndex: Four-aligned source-quad base GPR type index.
 * @param address: GPR prepared by atomic::set_address with an L1 byte address.
 * @param data: First of four consecutive source GPRs.
 * @param granule_mask: Eight 16-bit line-granule write enables.
 * @note Drain ScalarIdle while blocking the dependent consumer before relying on the
 *       updated memory.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline __attribute__((always_inline)) void store_masked(const hal::Gpr<AddressIndex> address, const hal::Gpr<DataIndex> data, const std::uint8_t granule_mask)
{
    const MaskedWrite<AddressIndex, DataIndex> descriptor {address, data, granule_mask};
    TT_INSN(descriptor.operation());
}

} // namespace hal::atomic
