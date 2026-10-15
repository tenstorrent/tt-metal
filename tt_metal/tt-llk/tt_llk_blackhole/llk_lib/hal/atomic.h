// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"
#include "utils/gpr.h"

namespace hal::atomic
{

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
    std::uint8_t counter_width = 32;

    /**
     * @brief Return the encoded ATINCGET instruction without issuing it.
     *
     * @note Enable LLK assertions to diagnose invalid fields. Disabled assertions leave the
     *       raw TT_OP_ATINCGET encoding path without runtime overhead.
     */
    constexpr std::uint32_t operation() const;
};

/**
 * @brief Describe a blocking FIFO pointer update or state wait (ATINCGETPTR).
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
     * @note Enable LLK assertions to diagnose invalid fields. Disabled assertions leave the
     *       raw TT_OP_ATINCGETPTR encoding path without runtime overhead.
     */
    constexpr std::uint32_t operation() const;
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
    std::uint8_t word_select = 0;
    std::uint8_t compare;
    std::uint8_t set;

    /**
     * @brief Return the encoded ATCAS instruction without issuing it.
     *
     * @note Enable LLK assertions to diagnose invalid fields. Disabled assertions leave the
     *       raw TT_OP_ATCAS encoding path without runtime overhead.
     */
    constexpr std::uint32_t operation() const;
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
     * @note Enable LLK assertions to diagnose invalid fields. Disabled assertions leave the
     *       raw TT_OP_ATSWAP encoding path without runtime overhead.
     */
    constexpr std::uint32_t operation() const;
};

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
FetchIncrement(hal::Gpr<AddressIndex>, hal::Gpr<DataIndex>, std::uint8_t, std::uint8_t) -> FetchIncrement<AddressIndex, DataIndex>;

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
FifoAcquire(hal::Gpr<AddressIndex>, hal::Gpr<DataIndex>, FifoPointer, std::uint8_t, std::uint8_t, bool) -> FifoAcquire<AddressIndex, DataIndex>;

template <std::uint32_t AddressIndex>
CompareSet(hal::Gpr<AddressIndex>, std::uint8_t, std::uint8_t, std::uint8_t) -> CompareSet<AddressIndex>;

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
MaskedWrite(hal::Gpr<AddressIndex>, hal::Gpr<DataIndex>, std::uint8_t) -> MaskedWrite<AddressIndex, DataIndex>;

/**
 * @brief Identify a wrapping counter in one 32-bit word of a 16-byte L1 line.
 *
 * @tparam WidthBits: Counter width in bits, in [1, 32].
 * @tparam AddressIndex: Address GPR type index.
 */
template <std::uint8_t WidthBits, std::uint32_t AddressIndex>
struct Counter
{
    hal::Gpr<AddressIndex> address;
    std::uint8_t word = 0;

    /**
     * @brief Build the ATINCGET descriptor for this counter without issuing it.
     *
     * @tparam DataIndex: Addend and result GPR type index.
     * @param data: GPR supplying the addend and receiving the pre-increment value.
     */
    template <std::uint32_t DataIndex>
    constexpr FetchIncrement<AddressIndex, DataIndex> fetch_add(hal::Gpr<DataIndex> data) const;
};

/**
 * @brief Identify a hardware-managed FIFOControl structure in one 16-byte L1 line.
 *
 * @note Construct with fifo<CapacityLog2>(hal::gpr<Index>()) to deduce the address GPR index.
 *
 * @tparam CapacityLog2: Base-two logarithm of FIFO capacity, in [0, 15].
 * @tparam AddressIndex: FIFOControl address GPR type index.
 */
template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
struct Fifo
{
    hal::Gpr<AddressIndex> control_address;

    /**
     * @brief Build a blocking read-counter advance descriptor without issuing it.
     *
     * @tparam IncrementLog2: Base-two logarithm of the pointer increment, in [0, 15]; defaults to zero for one slot.
     * @tparam ResultIndex: Pointer-result GPR type index.
     * @param result: GPR receiving the pre-increment read pointer.
     */
    template <std::uint8_t IncrementLog2 = 0, std::uint32_t ResultIndex>
    constexpr FifoAcquire<AddressIndex, ResultIndex> pop(hal::Gpr<ResultIndex> result) const;

    /**
     * @brief Build a runtime-increment blocking read-counter advance descriptor without issuing it.
     *
     * @tparam ResultIndex: Pointer-result GPR type index.
     * @param result: GPR receiving the pre-increment read pointer.
     * @param increment_log2: Base-two logarithm of the pointer increment, in [0, 15].
     */
    template <std::uint32_t ResultIndex>
    constexpr FifoAcquire<AddressIndex, ResultIndex> pop(hal::Gpr<ResultIndex> result, std::uint8_t increment_log2) const;

    /**
     * @brief Build a blocking write-counter advance descriptor without issuing it.
     *
     * @tparam IncrementLog2: Base-two logarithm of the pointer increment, in [0, 15]; defaults to zero for one slot.
     * @tparam ResultIndex: Pointer-result GPR type index.
     * @param result: GPR receiving the pre-increment write pointer.
     */
    template <std::uint8_t IncrementLog2 = 0, std::uint32_t ResultIndex>
    constexpr FifoAcquire<AddressIndex, ResultIndex> push(hal::Gpr<ResultIndex> result) const;

    /**
     * @brief Build a runtime-increment blocking write-counter advance descriptor without issuing it.
     *
     * @tparam ResultIndex: Pointer-result GPR type index.
     * @param result: GPR receiving the pre-increment write pointer.
     * @param increment_log2: Base-two logarithm of the pointer increment, in [0, 15].
     */
    template <std::uint32_t ResultIndex>
    constexpr FifoAcquire<AddressIndex, ResultIndex> push(hal::Gpr<ResultIndex> result, std::uint8_t increment_log2) const;

    /**
     * @brief Build a blocking FIFO-state wait descriptor without issuing it.
     *
     * @tparam State: Condition that blocks the operation, values = <Empty/Full>.
     * @tparam ResultIndex: Pointer-result GPR type index.
     * @param result: GPR receiving the current read pointer for Empty or write pointer for Full.
     */
    template <FifoState State, std::uint32_t ResultIndex>
    constexpr FifoAcquire<AddressIndex, ResultIndex> wait_while(hal::Gpr<ResultIndex> result) const;
};

/**
 * @brief Identify an existing FIFOControl structure, deducing the address GPR index from its operand.
 *
 * @tparam CapacityLog2: Base-two logarithm of FIFO capacity, in [0, 15].
 * @tparam AddressIndex: Deduced FIFOControl address GPR type index.
 * @param control_address: GPR containing the FIFOControl line address; no memory or GPR is initialized.
 */
template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
inline constexpr Fifo<CapacityLog2, AddressIndex> fifo(const hal::Gpr<AddressIndex> control_address)
{
    return Fifo<CapacityLog2, AddressIndex> {control_address};
}

/**
 * @brief Identify a lock or flag in one 32-bit word of a 16-byte L1 line.
 *
 * @tparam AddressIndex: Address GPR type index.
 */
template <std::uint32_t AddressIndex>
struct Lock
{
    hal::Gpr<AddressIndex> address;
    std::uint8_t word = 0;

    /** @brief Build the blocking {0 -> 1} ATCAS descriptor without issuing it. */
    constexpr CompareSet<AddressIndex> acquire() const;

    /** @brief Build the blocking {1 -> 0} ATCAS descriptor without issuing it. */
    constexpr CompareSet<AddressIndex> release() const;

    /**
     * @brief Build a compile-time-value ATCAS descriptor without issuing it.
     *
     * @tparam Compare: Zero-extended whole-word compare value, in [0, 15].
     * @tparam Set: Zero-extended whole-word replacement value, in [0, 15].
     */
    template <std::uint8_t Compare, std::uint8_t Set>
    constexpr CompareSet<AddressIndex> compare_set() const;

    /**
     * @brief Build a runtime-value ATCAS descriptor without issuing it.
     *
     * @param compare: Zero-extended whole-word compare value, in [0, 15].
     * @param set: Zero-extended whole-word replacement value, in [0, 15].
     */
    constexpr CompareSet<AddressIndex> compare_set(std::uint8_t compare, std::uint8_t set) const;
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

#ifdef ENABLE_LLK_ASSERT
template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline __attribute__((always_inline)) void assert_valid(const FetchIncrement<AddressIndex, DataIndex> operation)
{
    LLK_ASSERT(is_valid(operation.address), "ATINCGET address GPR index must be in [0, 63]");
    LLK_ASSERT(is_valid(operation.data), "ATINCGET data GPR index must be in [0, 63]");
    LLK_ASSERT(operation.word_select < 4u, "ATINCGET word selection must be in [0, 3]");
    LLK_ASSERT(operation.counter_width >= 1u && operation.counter_width <= 32u, "ATINCGET counter width must be in [1, 32]");
}

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline __attribute__((always_inline)) void assert_valid(const FifoAcquire<AddressIndex, DataIndex> operation)
{
    LLK_ASSERT(is_valid(operation.address), "ATINCGETPTR address GPR index must be in [0, 63]");
    LLK_ASSERT(is_valid(operation.data), "ATINCGETPTR result GPR index must be in [0, 63]");
    LLK_ASSERT(is_valid(operation.pointer), "ATINCGETPTR pointer must be Read or Write");
    LLK_ASSERT(operation.capacity_log2 <= 15u, "ATINCGETPTR FIFO capacity logarithm must be in [0, 15]");
    LLK_ASSERT(operation.increment_log2 <= 15u, "ATINCGETPTR increment logarithm must be in [0, 15]");
    LLK_ASSERT(!operation.probe_only || operation.increment_log2 == 0u, "ATINCGETPTR readiness probes cannot increment the pointer");
}

template <std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void assert_valid(const CompareSet<AddressIndex> operation)
{
    LLK_ASSERT(is_valid(operation.address), "ATCAS address GPR index must be in [0, 63]");
    LLK_ASSERT(operation.word_select < 4u, "ATCAS word selection must be in [0, 3]");
    LLK_ASSERT(operation.compare < 16u, "ATCAS compare value must fit in four bits");
    LLK_ASSERT(operation.set < 16u, "ATCAS replacement value must fit in four bits");
}

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline __attribute__((always_inline)) void assert_valid(const MaskedWrite<AddressIndex, DataIndex> operation)
{
    LLK_ASSERT(is_valid(operation.address), "ATSWAP address GPR index must be in [0, 63]");
    LLK_ASSERT(is_valid(operation.data), "ATSWAP data GPR index must be in [0, 63]");
    LLK_ASSERT(operation.data.index % 4u == 0u, "ATSWAP data GPR must be four-aligned");
}
#endif

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
        (operation.capacity_log2 + 1u) & 0xfu,
        static_cast<std::uint32_t>(operation.pointer),
        operation.data.index,
        operation.address.index);
}

template <std::uint32_t AddressIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t encode(const CompareSet<AddressIndex> operation)
{
    return TT_OP_ATCAS(L1MemoryEncoding, operation.set, operation.compare, operation.word_select, 0, operation.address.index);
}

template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline constexpr __attribute__((always_inline)) std::uint32_t encode(const MaskedWrite<AddressIndex, DataIndex> operation)
{
    return TT_OP_ATSWAP(L1MemoryEncoding, operation.granule_mask, operation.data.index, operation.address.index);
}

template <typename Descriptor>
inline constexpr __attribute__((always_inline)) std::uint32_t checked_encode(const Descriptor &descriptor)
{
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        assert_valid(descriptor);
    }
#endif
    return encode(descriptor);
}
} // namespace detail

/** @brief Return whether a fetch-increment descriptor can be encoded without truncation. */
template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
constexpr bool is_valid(const FetchIncrement<AddressIndex, DataIndex> operation)
{
    return detail::is_valid(operation.address) && detail::is_valid(operation.data) && operation.word_select < 4u && operation.counter_width >= 1u &&
           operation.counter_width <= 32u;
}

/** @brief Return whether a FIFO descriptor can be encoded without truncation. */
template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
constexpr bool is_valid(const FifoAcquire<AddressIndex, DataIndex> operation)
{
    return detail::is_valid(operation.address) && detail::is_valid(operation.data) && detail::is_valid(operation.pointer) && operation.capacity_log2 <= 15u &&
           operation.increment_log2 <= 15u && (!operation.probe_only || operation.increment_log2 == 0u);
}

/** @brief Return whether a compare-set descriptor can be encoded without truncation. */
template <std::uint32_t AddressIndex>
constexpr bool is_valid(const CompareSet<AddressIndex> operation)
{
    return detail::is_valid(operation.address) && operation.word_select < 4u && operation.compare < 16u && operation.set < 16u;
}

/** @brief Return whether a masked-write descriptor can be encoded without truncation. */
template <std::uint32_t AddressIndex, std::uint32_t DataIndex>
constexpr bool is_valid(const MaskedWrite<AddressIndex, DataIndex> operation)
{
    return detail::is_valid(operation.address) && detail::is_valid(operation.data) && operation.data.index % 4u == 0u;
}

/** @brief Return whether a counter identity has a valid layout and address GPR. */
template <std::uint8_t WidthBits, std::uint32_t AddressIndex>
constexpr bool is_valid(const Counter<WidthBits, AddressIndex> counter)
{
    return WidthBits >= 1u && WidthBits <= 32u && detail::is_valid(counter.address) && counter.word < 4u;
}

/** @brief Return whether a FIFO identity has a valid capacity and address GPR. */
template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
constexpr bool is_valid(const Fifo<CapacityLog2, AddressIndex> fifo)
{
    return CapacityLog2 <= 15u && detail::is_valid(fifo.control_address);
}

/** @brief Return whether a lock identity has a valid word and address GPR. */
template <std::uint32_t AddressIndex>
constexpr bool is_valid(const Lock<AddressIndex> lock)
{
    return detail::is_valid(lock.address) && lock.word < 4u;
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

template <std::uint8_t WidthBits, std::uint32_t AddressIndex>
template <std::uint32_t DataIndex>
inline constexpr __attribute__((always_inline)) FetchIncrement<AddressIndex, DataIndex> Counter<WidthBits, AddressIndex>::fetch_add(
    const hal::Gpr<DataIndex> data) const
{
    return FetchIncrement<AddressIndex, DataIndex> {address, data, word, WidthBits};
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <std::uint8_t IncrementLog2, std::uint32_t ResultIndex>
inline constexpr __attribute__((always_inline)) FifoAcquire<AddressIndex, ResultIndex> Fifo<CapacityLog2, AddressIndex>::pop(
    const hal::Gpr<ResultIndex> result) const
{
    static_assert(IncrementLog2 <= 15u, "FIFO increment logarithm must be in [0, 15]");
    return pop(result, IncrementLog2);
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <std::uint32_t ResultIndex>
inline constexpr __attribute__((always_inline)) FifoAcquire<AddressIndex, ResultIndex> Fifo<CapacityLog2, AddressIndex>::pop(
    const hal::Gpr<ResultIndex> result, const std::uint8_t increment_log2) const
{
    return FifoAcquire<AddressIndex, ResultIndex> {control_address, result, FifoPointer::Read, CapacityLog2, increment_log2, false};
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <std::uint8_t IncrementLog2, std::uint32_t ResultIndex>
inline constexpr __attribute__((always_inline)) FifoAcquire<AddressIndex, ResultIndex> Fifo<CapacityLog2, AddressIndex>::push(
    const hal::Gpr<ResultIndex> result) const
{
    static_assert(IncrementLog2 <= 15u, "FIFO increment logarithm must be in [0, 15]");
    return push(result, IncrementLog2);
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <std::uint32_t ResultIndex>
inline constexpr __attribute__((always_inline)) FifoAcquire<AddressIndex, ResultIndex> Fifo<CapacityLog2, AddressIndex>::push(
    const hal::Gpr<ResultIndex> result, const std::uint8_t increment_log2) const
{
    return FifoAcquire<AddressIndex, ResultIndex> {control_address, result, FifoPointer::Write, CapacityLog2, increment_log2, false};
}

template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex>
template <FifoState State, std::uint32_t ResultIndex>
inline constexpr __attribute__((always_inline)) FifoAcquire<AddressIndex, ResultIndex> Fifo<CapacityLog2, AddressIndex>::wait_while(
    const hal::Gpr<ResultIndex> result) const
{
    return FifoAcquire<AddressIndex, ResultIndex> {control_address, result, detail::pointer_for<State>(), CapacityLog2, 0, true};
}

template <std::uint32_t AddressIndex>
inline constexpr __attribute__((always_inline)) CompareSet<AddressIndex> Lock<AddressIndex>::acquire() const
{
    return compare_set<0, 1>();
}

template <std::uint32_t AddressIndex>
inline constexpr __attribute__((always_inline)) CompareSet<AddressIndex> Lock<AddressIndex>::release() const
{
    return compare_set<1, 0>();
}

template <std::uint32_t AddressIndex>
template <std::uint8_t Compare, std::uint8_t Set>
inline constexpr __attribute__((always_inline)) CompareSet<AddressIndex> Lock<AddressIndex>::compare_set() const
{
    static_assert(Compare < 16u, "ATCAS compare value must fit in four bits");
    static_assert(Set < 16u, "ATCAS set value must fit in four bits");
    return compare_set(Compare, Set);
}

template <std::uint32_t AddressIndex>
inline constexpr __attribute__((always_inline)) CompareSet<AddressIndex> Lock<AddressIndex>::compare_set(
    const std::uint8_t compare, const std::uint8_t set) const
{
    return CompareSet<AddressIndex> {address, word, compare, set};
}

/**
 * @brief Issue a compile-time counter fetch-and-add as one immediate ATINCGET.
 *
 * @tparam CounterIdentity: Complete constexpr @ref Counter supplied inside angle brackets.
 * @tparam DataIndex: Index of the GPR supplying the addend and receiving the pre-increment value.
 * @note Drain ScalarIdle while blocking the dependent consumer before consuming or
 *       reusing the data GPR.
 */
template <auto CounterIdentity, std::uint32_t DataIndex>
inline __attribute__((always_inline)) void fetch_add(const hal::Gpr<DataIndex>)
{
    static_assert(DataIndex != hal::detail::DynamicGprIndex, "immediate atomic fetch-add requires hal::gpr<Index>()");
    static_assert(is_valid(CounterIdentity), "invalid atomic counter identity");
    constexpr auto descriptor = CounterIdentity.fetch_add(hal::Gpr<DataIndex> {});
    static_assert(is_valid(descriptor), "invalid atomic descriptor");
    constexpr std::uint32_t encoding = descriptor.operation();
    TTI_INSN(encoding);
}

/**
 * @brief Encode and issue a value-selected counter fetch-and-add.
 *
 * @tparam WidthBits: Counter width in bits, in [1, 32].
 * @tparam AddressIndex: Address GPR type index.
 * @tparam DataIndex: Addend and result GPR type index.
 * @param counter: Counter identity supplying the line, word, and wrap width.
 * @param data: GPR supplying the addend and receiving the pre-increment value.
 * @note Drain ScalarIdle while blocking the dependent consumer before consuming or
 *       reusing the data GPR.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
template <std::uint8_t WidthBits, std::uint32_t AddressIndex, std::uint32_t DataIndex>
inline __attribute__((always_inline)) void fetch_add(const Counter<WidthBits, AddressIndex> counter, const hal::Gpr<DataIndex> data)
{
    TT_INSN(counter.fetch_add(data).operation());
}

/**
 * @brief Issue a compile-time blocking FIFO-state wait as one immediate ATINCGETPTR.
 *
 * @tparam State: Condition that blocks the operation, values = <Empty/Full>.
 * @tparam FifoIdentity: Complete constexpr @ref Fifo supplied inside angle brackets.
 * @tparam ResultIndex: Index of the GPR receiving the current read pointer for Empty or write pointer for Full.
 */
template <FifoState State, auto FifoIdentity, std::uint32_t ResultIndex>
inline __attribute__((always_inline)) void wait_while(const hal::Gpr<ResultIndex>)
{
    static_assert(ResultIndex != hal::detail::DynamicGprIndex, "immediate FIFO wait requires hal::gpr<Index>()");
    static_assert(is_valid(FifoIdentity), "invalid atomic FIFO identity");
    constexpr auto descriptor = FifoIdentity.template wait_while<State>(hal::Gpr<ResultIndex> {});
    static_assert(is_valid(descriptor), "invalid atomic descriptor");
    constexpr std::uint32_t encoding = descriptor.operation();
    TTI_INSN(encoding);
}

/**
 * @brief Encode and issue a value-selected blocking FIFO-state wait.
 *
 * @tparam State: Condition that blocks the operation, values = <Empty/Full>.
 * @tparam CapacityLog2: Base-two logarithm of FIFO capacity, in [0, 15].
 * @tparam AddressIndex: FIFOControl address GPR type index.
 * @tparam ResultIndex: Pointer-result GPR type index.
 * @param fifo: FIFO identity supplying the control line and capacity.
 * @param result: GPR receiving the current read pointer for Empty or write pointer for Full.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
template <FifoState State, std::uint8_t CapacityLog2, std::uint32_t AddressIndex, std::uint32_t ResultIndex>
inline __attribute__((always_inline)) void wait_while(const Fifo<CapacityLog2, AddressIndex> fifo, const hal::Gpr<ResultIndex> result)
{
    TT_INSN(fifo.template wait_while<State>(result).operation());
}

/**
 * @brief Pop FIFO slots from the read-counter bookkeeping as one immediate ATINCGETPTR.
 *
 * Hardware checks only empty/nonempty; callers using an increment greater than one must
 * establish that the complete batch is available.
 *
 * @tparam FifoIdentity: Complete constexpr @ref Fifo supplied inside angle brackets.
 * @tparam IncrementLog2: Base-two logarithm of the read-pointer increment, in [0, 15]; defaults to zero for one slot.
 * @tparam ResultIndex: Index of the GPR receiving the pre-increment read pointer.
 * @note Move the payload separately; this operation updates FIFO bookkeeping only.
 */
template <auto FifoIdentity, std::uint8_t IncrementLog2 = 0, std::uint32_t ResultIndex>
inline __attribute__((always_inline)) void pop(const hal::Gpr<ResultIndex>)
{
    static_assert(ResultIndex != hal::detail::DynamicGprIndex, "immediate FIFO pop requires hal::gpr<Index>()");
    static_assert(is_valid(FifoIdentity), "invalid atomic FIFO identity");
    constexpr auto descriptor = FifoIdentity.template pop<IncrementLog2>(hal::Gpr<ResultIndex> {});
    static_assert(is_valid(descriptor), "invalid atomic descriptor");
    constexpr std::uint32_t encoding = descriptor.operation();
    TTI_INSN(encoding);
}

/**
 * @brief Pop FIFO slots from value-selected read-counter bookkeeping.
 *
 * Hardware checks only empty/nonempty; callers using an increment greater than one must
 * establish that the complete batch is available.
 *
 * @tparam CapacityLog2: Base-two logarithm of FIFO capacity, in [0, 15].
 * @tparam AddressIndex: FIFOControl address GPR type index.
 * @tparam ResultIndex: Pointer-result GPR type index.
 * @param fifo: FIFO identity supplying the control line and capacity.
 * @param result: GPR receiving the pre-increment read pointer.
 * @param increment_log2: Base-two logarithm of the read-pointer increment, in [0, 15]; defaults to zero for one slot.
 * @note Move the payload separately; this operation updates FIFO bookkeeping only.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex, std::uint32_t ResultIndex>
inline __attribute__((always_inline)) void pop(
    const Fifo<CapacityLog2, AddressIndex> fifo, const hal::Gpr<ResultIndex> result, const std::uint8_t increment_log2 = 0)
{
    TT_INSN(fifo.pop(result, increment_log2).operation());
}

/**
 * @brief Push FIFO slots into the write-counter bookkeeping as one immediate ATINCGETPTR.
 *
 * Hardware checks only full/nonfull; callers using an increment greater than one must
 * establish that the complete batch fits.
 *
 * @tparam FifoIdentity: Complete constexpr @ref Fifo supplied inside angle brackets.
 * @tparam IncrementLog2: Base-two logarithm of the write-pointer increment, in [0, 15]; defaults to zero for one slot.
 * @tparam ResultIndex: Index of the GPR receiving the pre-increment write pointer.
 * @note Move the payload separately; this operation updates FIFO bookkeeping only.
 */
template <auto FifoIdentity, std::uint8_t IncrementLog2 = 0, std::uint32_t ResultIndex>
inline __attribute__((always_inline)) void push(const hal::Gpr<ResultIndex>)
{
    static_assert(ResultIndex != hal::detail::DynamicGprIndex, "immediate FIFO push requires hal::gpr<Index>()");
    static_assert(is_valid(FifoIdentity), "invalid atomic FIFO identity");
    constexpr auto descriptor = FifoIdentity.template push<IncrementLog2>(hal::Gpr<ResultIndex> {});
    static_assert(is_valid(descriptor), "invalid atomic descriptor");
    constexpr std::uint32_t encoding = descriptor.operation();
    TTI_INSN(encoding);
}

/**
 * @brief Push FIFO slots into value-selected write-counter bookkeeping.
 *
 * Hardware checks only full/nonfull; callers using an increment greater than one must
 * establish that the complete batch fits.
 *
 * @tparam CapacityLog2: Base-two logarithm of FIFO capacity, in [0, 15].
 * @tparam AddressIndex: FIFOControl address GPR type index.
 * @tparam ResultIndex: Pointer-result GPR type index.
 * @param fifo: FIFO identity supplying the control line and capacity.
 * @param result: GPR receiving the pre-increment write pointer.
 * @param increment_log2: Base-two logarithm of the write-pointer increment, in [0, 15]; defaults to zero for one slot.
 * @note Move the payload separately; this operation updates FIFO bookkeeping only.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
template <std::uint8_t CapacityLog2, std::uint32_t AddressIndex, std::uint32_t ResultIndex>
inline __attribute__((always_inline)) void push(
    const Fifo<CapacityLog2, AddressIndex> fifo, const hal::Gpr<ResultIndex> result, const std::uint8_t increment_log2 = 0)
{
    TT_INSN(fifo.push(result, increment_log2).operation());
}

/**
 * @brief Issue a compile-time generic lock compare-and-set as one immediate ATCAS.
 *
 * @tparam LockIdentity: Complete constexpr @ref Lock supplied inside angle brackets.
 * @tparam Compare: Zero-extended whole-word compare value, in [0, 15].
 * @tparam Set: Zero-extended whole-word replacement value, in [0, 15].
 */
template <auto LockIdentity, std::uint8_t Compare, std::uint8_t Set>
inline __attribute__((always_inline)) void compare_set()
{
    static_assert(is_valid(LockIdentity), "invalid atomic lock identity");
    constexpr auto descriptor = LockIdentity.template compare_set<Compare, Set>();
    static_assert(is_valid(descriptor), "invalid atomic descriptor");
    constexpr std::uint32_t encoding = descriptor.operation();
    TTI_INSN(encoding);
}

/**
 * @brief Encode and issue a value-selected blocking lock compare-and-set.
 *
 * @tparam AddressIndex: Address GPR type index.
 * @param lock: Lock identity supplying the line and word.
 * @param compare: Zero-extended whole-word compare value, in [0, 15].
 * @param set: Zero-extended whole-word replacement value, in [0, 15].
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
template <std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void compare_set(const Lock<AddressIndex> lock, const std::uint8_t compare, const std::uint8_t set)
{
    TT_INSN(lock.compare_set(compare, set).operation());
}

/**
 * @brief Issue a compile-time blocking lock acquire ({0 -> 1}) as one immediate ATCAS.
 *
 * @tparam LockIdentity: Complete constexpr @ref Lock supplied inside angle brackets.
 */
template <auto LockIdentity>
inline __attribute__((always_inline)) void acquire()
{
    compare_set<LockIdentity, 0, 1>();
}

/**
 * @brief Encode and issue a value-selected blocking lock acquire ({0 -> 1}).
 *
 * @tparam AddressIndex: Address GPR type index.
 * @param lock: Lock identity supplying the line and word.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
template <std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void acquire(const Lock<AddressIndex> lock)
{
    compare_set(lock, 0, 1);
}

/**
 * @brief Issue a compile-time blocking lock release ({1 -> 0}) as one immediate ATCAS.
 *
 * @tparam LockIdentity: Complete constexpr @ref Lock supplied inside angle brackets.
 */
template <auto LockIdentity>
inline __attribute__((always_inline)) void release()
{
    compare_set<LockIdentity, 1, 0>();
}

/**
 * @brief Encode and issue a value-selected blocking lock release ({1 -> 0}).
 *
 * @tparam AddressIndex: Address GPR type index.
 * @param lock: Lock identity supplying the line and word.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
template <std::uint32_t AddressIndex>
inline __attribute__((always_inline)) void release(const Lock<AddressIndex> lock)
{
    compare_set(lock, 1, 0);
}

/**
 * @brief Issue a compile-time posted masked line store as one immediate ATSWAP.
 *
 * @tparam GranuleMask: Eight 16-bit line-granule write enables.
 * @tparam AddressIndex: Index of the GPR containing the 16-byte L1 line address.
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
 * @param address: GPR containing the 16-byte L1 line address.
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
