// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

namespace hal::fpu
{

/**
 * @brief Select the arithmetic an @ref Elementwise operation performs on each SrcA/SrcB datum pair.
 *
 * Subtract computes SrcA - SrcB. Multiply always accumulates onto Dest regardless of the
 * selected @ref DestWriteMode; zero the target rows first for overwrite semantics.
 */
enum class ElementwiseOperation : std::uint8_t
{
    Add,
    Subtract,
    Multiply
};

/**
 * @brief Select the reduction a @ref Pool operation performs.
 *
 * Sum (GAPOOL) is a four-row matrix multiply: Dest(4x16) += SrcB(4x16) x SrcA(16x16), with
 * SrcB read from the top half of the eight-row-aligned SrcB block and all four rows of the
 * Dest block accumulated. A row of scaled column sums is a software convention: fill one
 * SrcB row with the scaler (1/N for an average) and read the matching Dest row.
 * Maximum (GMPOOL) scales the sixteen-row SrcA block by one SrcB scaler row, reduces every
 * column to its maximum, and combines the 1x16 result with the existing Dest row.
 */
enum class PoolFunction : std::uint8_t
{
    Sum,
    Maximum
};

/**
 * @brief Select how SrcB datums are replicated before an elementwise operation consumes them.
 *
 * Column replicates each consumed row's column-0 datum across all sixteen columns. Row
 * replicates the counter-selected SrcB row — free of the usual eight-row alignment, so the
 * SrcB counter picks an exact row — onto all eight consumed rows, and Scalar replicates that
 * selected row's column-0 datum onto the whole block.
 */
enum class SrcBBroadcast : std::uint8_t
{
    None   = 0x0,
    Column = 0x1,
    Row    = 0x2,
    Scalar = 0x3
};

/**
 * @brief Select MVMUL SrcB row replication.
 *
 * Row replicates the counter-selected SrcB row — free of the usual eight-row alignment, so
 * the SrcB counter picks an exact row — and accumulates the product into the even rows of a
 * seven-row Dest block, leaving the odd rows unchanged. No LLK call site sets it today.
 */
enum class SrcBRowBroadcast : std::uint8_t
{
    None = 0x0,
    Row  = 0x1
};

/**
 * @brief Select whether results overwrite Dest or accumulate onto the existing Dest values.
 *
 * Honored by Add and Subtract only; Multiply always accumulates.
 */
enum class DestWriteMode : std::uint8_t
{
    Overwrite  = 0x0,
    Accumulate = 0x1
};

/**
 * @brief Select whether a maximum pool also records each column's winning row index.
 *
 * Indices land in the low bits of the pooled Dest datum, so they survive only 32-bit Dest
 * writes; a 16-bit Dest discards them. Only SrcA rows 0-7 can record an index, and the
 * recorded value is permuted by the involutory transform {0, 3, 6, 1, 4, 7, 2, 5}, so
 * applying the same table decodes it. Each issue also advances a four-bit phase counter in
 * the datum, distinguishing chained pools onto the same Dest row.
 */
enum class IndexTracking : std::uint8_t
{
    Disabled = 0x0,
    Enabled  = 0x1
};

/**
 * @brief Select which source banks the operation releases back to the unpackers when it retires.
 *
 * Releasing a bank flips the math thread to the paired bank and clears the released bank's
 * data-valid flag so the producing unpacker can refill it. The per-thread CLR_DVALID disable
 * configuration downgrades a release to a bank flip that keeps data-valid set. There is no
 * implicit default; select a release at every call site.
 */
enum class SourceRelease : std::uint8_t
{
    Unset = 0xFF,
    None  = 0x0,
    SrcA  = 0x1,
    SrcB  = 0x2,
    Both  = 0x3
};

/**
 * @brief Describe one elementwise operation (ELWADD/ELWSUB/ELWMUL) over an 8x16 datum block.
 *
 * Each issue consumes eight SrcA rows and eight SrcB rows (fewer under broadcast) and writes
 * eight Dest rows. Data formats, fidelity phases, and post-issue counter stepping remain in
 * ALU configuration and the selected ADDR_MOD register. Dest rows resolve as dest_row_offset
 * plus the Dest counter and the configured Dest base offsets, aligned down to eight rows.
 */
struct Elementwise
{
    ElementwiseOperation operation;
    SrcBBroadcast broadcast       = SrcBBroadcast::None;
    DestWriteMode dest_write      = DestWriteMode::Overwrite;
    std::uint8_t address_modifier = 0;
    std::uint16_t dest_row_offset = 0;
    SourceRelease release         = SourceRelease::Unset;
};

/**
 * @brief Describe one DOTPV dot product: Dest(8x16) += SrcB(8x16) x SrcA(16x16).
 *
 * Functionally MVMUL without row broadcast: it always accumulates onto Dest. The instruction
 * word reserves the elementwise broadcast and Dest-write bit positions, but the ISA gives
 * them no field, so this interface encodes them as zero; the raw macros remain the
 * raw compatibility path.
 *
 * @note DOTPV is the restricted legacy-compatible form of MVMUL; prefer @ref MatrixMultiply.
 */
struct DotProduct
{
    std::uint8_t address_modifier = 0;
    std::uint16_t dest_row_offset = 0;
    SourceRelease release         = SourceRelease::Unset;
};

/**
 * @brief Describe one MVMUL matrix multiply: Dest(8x16) += SrcB(8x16) x SrcA(16x16).
 *
 * Always accumulates onto Dest; zero the target rows first for overwrite semantics.
 * Fidelity phases and post-issue counter stepping remain in the selected ADDR_MOD register.
 */
struct MatrixMultiply
{
    SrcBRowBroadcast broadcast    = SrcBRowBroadcast::None;
    std::uint8_t address_modifier = 0;
    std::uint16_t dest_row_offset = 0;
    SourceRelease release         = SourceRelease::Unset;
};

/**
 * @brief Describe one pooling reduction (GAPOOL/GMPOOL) over the sixteen-row SrcA block.
 *
 * The Dest target is the four-row-aligned block selected by dest_row_offset, the Dest
 * counter, and the configured Dest base offsets: Sum accumulates all four of its rows,
 * Maximum combines its first row and zeroes (or phase-stamps) the other three. Per-function
 * SrcB consumption and result layout are described on @ref PoolFunction. The ISA defines no
 * pool-shape field; the bit the instruction metadata calls a single-row selector is encoded
 * at its only characterized value, and the raw macros remain the low-level compatibility path.
 *
 * @note Maximum reads uninitialized Dest rows as negative infinity; zero the target row first.
 * @note Index tracking applies only to Maximum with a 32-bit Dest; see @ref IndexTracking.
 */
struct Pool
{
    PoolFunction function;
    IndexTracking indices         = IndexTracking::Disabled;
    std::uint8_t address_modifier = 0;
    std::uint16_t dest_row_offset = 0;
    SourceRelease release         = SourceRelease::Unset;
};

namespace detail
{
constexpr bool is_valid(const ElementwiseOperation operation)
{
    return operation == ElementwiseOperation::Add || operation == ElementwiseOperation::Subtract || operation == ElementwiseOperation::Multiply;
}

constexpr bool is_valid(const PoolFunction function)
{
    return function == PoolFunction::Sum || function == PoolFunction::Maximum;
}

constexpr bool is_valid(const SrcBBroadcast broadcast)
{
    return broadcast == SrcBBroadcast::None || broadcast == SrcBBroadcast::Column || broadcast == SrcBBroadcast::Row || broadcast == SrcBBroadcast::Scalar;
}

constexpr bool is_valid(const SrcBRowBroadcast broadcast)
{
    return broadcast == SrcBRowBroadcast::None || broadcast == SrcBRowBroadcast::Row;
}

constexpr bool is_valid(const DestWriteMode mode)
{
    return mode == DestWriteMode::Overwrite || mode == DestWriteMode::Accumulate;
}

constexpr bool is_valid(const IndexTracking indices)
{
    return indices == IndexTracking::Disabled || indices == IndexTracking::Enabled;
}

constexpr bool is_valid(const SourceRelease release)
{
    return release == SourceRelease::None || release == SourceRelease::SrcA || release == SourceRelease::SrcB || release == SourceRelease::Both;
}

constexpr bool has_valid_address_modifier(const std::uint8_t address_modifier)
{
    return ckernel::is_valid(address_modifier, 3);
}

constexpr bool has_valid_dest_row_offset(const std::uint16_t dest_row_offset)
{
    return ckernel::is_valid(dest_row_offset, 10);
}

constexpr bool is_valid(const Elementwise operation)
{
    return is_valid(operation.operation) && is_valid(operation.broadcast) && is_valid(operation.dest_write) &&
           has_valid_address_modifier(operation.address_modifier) && has_valid_dest_row_offset(operation.dest_row_offset) && is_valid(operation.release);
}

constexpr bool is_valid(const DotProduct operation)
{
    return has_valid_address_modifier(operation.address_modifier) && has_valid_dest_row_offset(operation.dest_row_offset) && is_valid(operation.release);
}

constexpr bool is_valid(const MatrixMultiply operation)
{
    return is_valid(operation.broadcast) && has_valid_address_modifier(operation.address_modifier) && has_valid_dest_row_offset(operation.dest_row_offset) &&
           is_valid(operation.release);
}

constexpr bool has_valid_index_tracking(const Pool operation)
{
    return operation.indices == IndexTracking::Disabled || operation.function == PoolFunction::Maximum;
}

constexpr bool is_valid(const Pool operation)
{
    return is_valid(operation.function) && is_valid(operation.indices) && has_valid_address_modifier(operation.address_modifier) &&
           has_valid_dest_row_offset(operation.dest_row_offset) && is_valid(operation.release) && has_valid_index_tracking(operation);
}

#ifdef ENABLE_LLK_ASSERT
inline __attribute__((always_inline)) void assert_valid(const Elementwise operation)
{
    LLK_ASSERT(is_valid(operation.operation), "FPU elementwise operation must be Add, Subtract, or Multiply");
    LLK_ASSERT(is_valid(operation.broadcast), "FPU elementwise broadcast must be None, Column, Row, or Scalar");
    LLK_ASSERT(is_valid(operation.dest_write), "FPU elementwise Dest write mode must be Overwrite or Accumulate");
    LLK_ASSERT(has_valid_address_modifier(operation.address_modifier), "FPU elementwise address modifier must be in [0, 7]");
    LLK_ASSERT(has_valid_dest_row_offset(operation.dest_row_offset), "FPU elementwise Dest row offset must fit in ten bits");
    LLK_ASSERT(is_valid(operation.release), "FPU elementwise source release must be None, SrcA, SrcB, or Both");
}

inline __attribute__((always_inline)) void assert_valid(const DotProduct operation)
{
    LLK_ASSERT(has_valid_address_modifier(operation.address_modifier), "FPU dot-product address modifier must be in [0, 7]");
    LLK_ASSERT(has_valid_dest_row_offset(operation.dest_row_offset), "FPU dot-product Dest row offset must fit in ten bits");
    LLK_ASSERT(is_valid(operation.release), "FPU dot-product source release must be None, SrcA, SrcB, or Both");
}

inline __attribute__((always_inline)) void assert_valid(const MatrixMultiply operation)
{
    LLK_ASSERT(is_valid(operation.broadcast), "FPU matrix-multiply broadcast must be None or Row");
    LLK_ASSERT(has_valid_address_modifier(operation.address_modifier), "FPU matrix-multiply address modifier must be in [0, 7]");
    LLK_ASSERT(has_valid_dest_row_offset(operation.dest_row_offset), "FPU matrix-multiply Dest row offset must fit in ten bits");
    LLK_ASSERT(is_valid(operation.release), "FPU matrix-multiply source release must be None, SrcA, SrcB, or Both");
}

inline __attribute__((always_inline)) void assert_valid(const Pool operation)
{
    LLK_ASSERT(is_valid(operation.function), "FPU pool function must be Sum or Maximum");
    LLK_ASSERT(is_valid(operation.indices), "FPU pool index tracking must be Disabled or Enabled");
    LLK_ASSERT(has_valid_address_modifier(operation.address_modifier), "FPU pool address modifier must be in [0, 7]");
    LLK_ASSERT(has_valid_dest_row_offset(operation.dest_row_offset), "FPU pool Dest row offset must fit in ten bits");
    LLK_ASSERT(is_valid(operation.release), "FPU pool source release must be None, SrcA, SrcB, or Both");
    LLK_ASSERT(has_valid_index_tracking(operation), "FPU pool index tracking requires Maximum");
}
#endif

constexpr std::uint32_t value(const SrcBBroadcast broadcast)
{
    return static_cast<std::uint32_t>(broadcast);
}

constexpr std::uint32_t value(const SrcBRowBroadcast broadcast)
{
    return static_cast<std::uint32_t>(broadcast);
}

constexpr std::uint32_t value(const DestWriteMode mode)
{
    return static_cast<std::uint32_t>(mode);
}

constexpr std::uint32_t value(const IndexTracking indices)
{
    return static_cast<std::uint32_t>(indices);
}

constexpr std::uint32_t value(const SourceRelease release)
{
    return static_cast<std::uint32_t>(release);
}

constexpr std::uint32_t get_operation(const Elementwise operation)
{
    switch (operation.operation)
    {
        case ElementwiseOperation::Add:
            return TT_OP_ELWADD(
                value(operation.release), value(operation.dest_write), value(operation.broadcast), operation.address_modifier, operation.dest_row_offset);
        case ElementwiseOperation::Subtract:
            return TT_OP_ELWSUB(
                value(operation.release), value(operation.dest_write), value(operation.broadcast), operation.address_modifier, operation.dest_row_offset);
        default:
            return TT_OP_ELWMUL(
                value(operation.release), value(operation.dest_write), value(operation.broadcast), operation.address_modifier, operation.dest_row_offset);
    }
}

constexpr std::uint32_t get_operation(const DotProduct operation)
{
    // The accumulate and broadcast bit positions have no DOTPV field; encode them as zero.
    return TT_OP_DOTPV(value(operation.release), 0, 0, operation.address_modifier, operation.dest_row_offset);
}

constexpr std::uint32_t get_operation(const MatrixMultiply operation)
{
    return TT_OP_MVMUL(value(operation.release), value(operation.broadcast), operation.address_modifier, operation.dest_row_offset);
}

constexpr std::uint32_t get_operation(const Pool operation)
{
    // Bit 19 has no ISA field; 1 is the value every characterized pool call site encodes.
    return operation.function == PoolFunction::Sum
               ? TT_OP_GAPOOL(value(operation.release), 1, operation.address_modifier, value(operation.indices), operation.dest_row_offset)
               : TT_OP_GMPOOL(value(operation.release), 1, operation.address_modifier, value(operation.indices), operation.dest_row_offset);
}
} // namespace detail

/** @brief Return whether an elementwise descriptor can be encoded without truncation. */
constexpr bool is_valid(const Elementwise operation)
{
    return detail::is_valid(operation);
}

/** @brief Return whether a dot-product descriptor can be encoded without truncation. */
constexpr bool is_valid(const DotProduct operation)
{
    return detail::is_valid(operation);
}

/** @brief Return whether a matrix-multiply descriptor can be encoded without truncation. */
constexpr bool is_valid(const MatrixMultiply operation)
{
    return detail::is_valid(operation);
}

/** @brief Return whether a pool descriptor can be encoded without truncation. */
constexpr bool is_valid(const Pool operation)
{
    return detail::is_valid(operation);
}

/**
 * @brief Encode a compile-time elementwise operation without issuing it.
 *
 * @tparam Operation: Complete elementwise description.
 * @note Use the result where another expander accepts an encoded operation, such as a MOP
 *       or replay configuration.
 */
template <Elementwise Operation>
constexpr std::uint32_t get_operation()
{
    static_assert(is_valid(Operation), "invalid elementwise descriptor");

    return detail::get_operation(Operation);
}

/**
 * @brief Encode a runtime-selected elementwise operation without issuing it.
 *
 * @param operation: Complete elementwise description.
 * @note Use the result where another expander accepts an encoded operation, such as a MOP
 *       or replay configuration.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline constexpr __attribute__((always_inline)) std::uint32_t get_operation(const Elementwise operation)
{
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(operation);
    }
#endif

    return detail::get_operation(operation);
}

/**
 * @brief Issue a compile-time elementwise operation as one immediate instruction.
 *
 * @tparam Operation: Complete elementwise description.
 */
template <Elementwise Operation>
inline __attribute__((always_inline)) void run()
{
    constexpr std::uint32_t operation = get_operation<Operation>();
    INSTRUCTION_WORD(operation);
}

/**
 * @brief Encode and issue a runtime-selected elementwise operation.
 *
 * @param operation: Complete elementwise description.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline __attribute__((always_inline)) void run(const Elementwise operation)
{
    ckernel::instrn_buffer[0] = get_operation(operation);
}

/**
 * @brief Encode a compile-time DOTPV dot product without issuing it.
 *
 * @tparam Operation: Complete dot-product description.
 * @note Prefer @ref MatrixMultiply; DOTPV remains for legacy compatibility.
 */
template <DotProduct Operation>
constexpr std::uint32_t get_operation()
{
    static_assert(is_valid(Operation), "invalid dot-product descriptor");

    return detail::get_operation(Operation);
}

/**
 * @brief Encode a runtime-selected DOTPV dot product without issuing it.
 *
 * @param operation: Complete dot-product description.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline constexpr __attribute__((always_inline)) std::uint32_t get_operation(const DotProduct operation)
{
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(operation);
    }
#endif

    return detail::get_operation(operation);
}

/**
 * @brief Issue a compile-time DOTPV dot product as one immediate instruction.
 *
 * @tparam Operation: Complete dot-product description.
 */
template <DotProduct Operation>
inline __attribute__((always_inline)) void run()
{
    constexpr std::uint32_t operation = get_operation<Operation>();
    INSTRUCTION_WORD(operation);
}

/**
 * @brief Encode and issue a runtime-selected DOTPV dot product.
 *
 * @param operation: Complete dot-product description.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline __attribute__((always_inline)) void run(const DotProduct operation)
{
    ckernel::instrn_buffer[0] = get_operation(operation);
}

/**
 * @brief Encode a compile-time MVMUL matrix multiply without issuing it.
 *
 * @tparam Operation: Complete matrix-multiply description.
 * @note Use the result where another expander accepts an encoded operation, such as a MOP
 *       or replay configuration.
 */
template <MatrixMultiply Operation>
constexpr std::uint32_t get_operation()
{
    static_assert(is_valid(Operation), "invalid matrix-multiply descriptor");

    return detail::get_operation(Operation);
}

/**
 * @brief Encode a runtime-selected MVMUL matrix multiply without issuing it.
 *
 * @param operation: Complete matrix-multiply description.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline constexpr __attribute__((always_inline)) std::uint32_t get_operation(const MatrixMultiply operation)
{
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(operation);
    }
#endif

    return detail::get_operation(operation);
}

/**
 * @brief Issue a compile-time MVMUL matrix multiply as one immediate instruction.
 *
 * @tparam Operation: Complete matrix-multiply description.
 */
template <MatrixMultiply Operation>
inline __attribute__((always_inline)) void run()
{
    constexpr std::uint32_t operation = get_operation<Operation>();
    INSTRUCTION_WORD(operation);
}

/**
 * @brief Encode and issue a runtime-selected MVMUL matrix multiply.
 *
 * @param operation: Complete matrix-multiply description.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline __attribute__((always_inline)) void run(const MatrixMultiply operation)
{
    ckernel::instrn_buffer[0] = get_operation(operation);
}

/**
 * @brief Encode a compile-time pool operation without issuing it.
 *
 * @tparam Operation: Complete pool description.
 * @note Use the result where another expander accepts an encoded operation, such as a MOP
 *       or replay configuration.
 */
template <Pool Operation>
constexpr std::uint32_t get_operation()
{
    static_assert(is_valid(Operation), "invalid pool descriptor");

    return detail::get_operation(Operation);
}

/**
 * @brief Encode a runtime-selected pool operation without issuing it.
 *
 * @param operation: Complete pool description.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline constexpr __attribute__((always_inline)) std::uint32_t get_operation(const Pool operation)
{
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(operation);
    }
#endif

    return detail::get_operation(operation);
}

/**
 * @brief Issue a compile-time pool operation as one immediate instruction.
 *
 * @tparam Operation: Complete pool description.
 */
template <Pool Operation>
inline __attribute__((always_inline)) void run()
{
    constexpr std::uint32_t operation = get_operation<Operation>();
    INSTRUCTION_WORD(operation);
}

/**
 * @brief Encode and issue a runtime-selected pool operation.
 *
 * @param operation: Complete pool description.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline __attribute__((always_inline)) void run(const Pool operation)
{
    ckernel::instrn_buffer[0] = get_operation(operation);
}

} // namespace hal::fpu
