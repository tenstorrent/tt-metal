// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cfg/detail/mmio_read.h"
#include "ckernel.h"
#include "utils/helpers.h"
#include "utils/operation.h"
#include "utils/validate.h"

namespace hal::fpu
{

/**
 * @brief Select the arithmetic an @ref Eltwise operation performs on each SrcA/SrcB datum pair.
 *
 * Subtract computes SrcA - SrcB. Multiply always accumulates onto Dest regardless of the
 * selected @ref DestWriteMode; zero the target rows first for overwrite semantics.
 */
enum class EltwiseOperation : std::uint8_t
{
    Add,
    Subtract,
    Multiply
};

/**
 * @brief Select the reduction a @ref Pool operation performs.
 *
 * Sum (GAPOOL) is a four-row matmul: Dest(4x16) += SrcB(4x16) x SrcA(16x16), with
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
 * @brief Select how SrcB datums are replicated before an eltwise operation consumes them.
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
 * seven-row Dest block, leaving the odd rows unchanged.
 *
 * @note With Row, a Dst row with bit 1 or 2 set is UnsupportedFunctionality: the Dst valid bits are
 *       then updated for the wrong rows. The encoder rejects such a dest_row_offset; keep the Dst
 *       counter and base offsets aligned as well.
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
 * @brief Describe one eltwise operation (ELWADD/ELWSUB/ELWMUL) over an 8x16 datum block.
 *
 * Each issue consumes eight SrcA rows and eight SrcB rows (fewer under broadcast) and writes
 * eight Dest rows. Data formats, fidelity phases, and post-issue counter stepping remain in
 * ALU configuration and the selected ADDR_MOD register. Dest rows resolve as dest_row_offset
 * plus the Dest counter and the configured Dest base offsets, aligned down to eight rows.
 */
class Eltwise
{
public:
    EltwiseOperation operation;
    SrcBBroadcast broadcast       = SrcBBroadcast::None;
    DestWriteMode dest_write      = DestWriteMode::Overwrite;
    std::uint8_t address_modifier = 0;
    std::uint16_t dest_row_offset = 0;
    SourceRelease release         = SourceRelease::Unset;
};

/**
 * @brief Describe one MVMUL matmul: Dest(8x16) += SrcB(8x16) x SrcA(16x16).
 *
 * Always accumulates onto Dest; zero the target rows first for overwrite semantics.
 * Fidelity phases and post-issue counter stepping remain in the selected ADDR_MOD register.
 */
class Matmul
{
public:
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
class Pool
{
public:
    PoolFunction function;
    IndexTracking indices         = IndexTracking::Disabled;
    std::uint8_t address_modifier = 0;
    std::uint16_t dest_row_offset = 0;
    SourceRelease release         = SourceRelease::Unset;
};

namespace detail
{
constexpr bool is_valid(const EltwiseOperation operation)
{
    return operation == EltwiseOperation::Add || operation == EltwiseOperation::Subtract || operation == EltwiseOperation::Multiply;
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
    return hal::detail::is_valid(address_modifier, 3);
}

constexpr bool has_valid_dest_row_offset(const std::uint16_t dest_row_offset)
{
    return hal::detail::is_valid(dest_row_offset, 10);
}

constexpr bool is_valid(const Eltwise operation)
{
    return is_valid(operation.operation) && is_valid(operation.broadcast) && is_valid(operation.dest_write) &&
           has_valid_address_modifier(operation.address_modifier) && has_valid_dest_row_offset(operation.dest_row_offset) && is_valid(operation.release);
}

// Row broadcast with Dst row bit 1 or 2 set updates the wrong Dst valid bits.
constexpr bool has_valid_broadcast_row(const Matmul operation)
{
    return operation.broadcast != SrcBRowBroadcast::Row || (operation.dest_row_offset & 0b110u) == 0;
}

constexpr bool is_valid(const Matmul operation)
{
    return is_valid(operation.broadcast) && has_valid_address_modifier(operation.address_modifier) && has_valid_dest_row_offset(operation.dest_row_offset) &&
           is_valid(operation.release) && has_valid_broadcast_row(operation);
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

inline constexpr __attribute__((always_inline)) void validate(const Eltwise operation)
{
    hal::detail::require(is_valid(operation.operation), "FPU eltwise operation must be Add, Subtract, or Multiply");
    hal::detail::require(is_valid(operation.broadcast), "FPU eltwise broadcast must be None, Column, Row, or Scalar");
    hal::detail::require(is_valid(operation.dest_write), "FPU eltwise Dest write mode must be Overwrite or Accumulate");
    hal::detail::require(has_valid_address_modifier(operation.address_modifier), "FPU eltwise address modifier must be in [0, 7]");
    hal::detail::require(has_valid_dest_row_offset(operation.dest_row_offset), "FPU eltwise Dest row offset must fit in ten bits");
    hal::detail::require(is_valid(operation.release), "FPU eltwise source release must be None, SrcA, SrcB, or Both");
}

inline constexpr __attribute__((always_inline)) void validate(const Matmul operation)
{
    hal::detail::require(is_valid(operation.broadcast), "FPU matmul broadcast must be None or Row");
    hal::detail::require(has_valid_address_modifier(operation.address_modifier), "FPU matmul address modifier must be in [0, 7]");
    hal::detail::require(has_valid_dest_row_offset(operation.dest_row_offset), "FPU matmul Dest row offset must fit in ten bits");
    hal::detail::require(is_valid(operation.release), "FPU matmul source release must be None, SrcA, SrcB, or Both");
    hal::detail::require(has_valid_broadcast_row(operation), "FPU matmul row broadcast needs a Dest row offset with bits 1 and 2 clear");
}

inline constexpr __attribute__((always_inline)) void validate(const Pool operation)
{
    hal::detail::require(is_valid(operation.function), "FPU pool function must be Sum or Maximum");
    hal::detail::require(is_valid(operation.indices), "FPU pool index tracking must be Disabled or Enabled");
    hal::detail::require(has_valid_address_modifier(operation.address_modifier), "FPU pool address modifier must be in [0, 7]");
    hal::detail::require(has_valid_dest_row_offset(operation.dest_row_offset), "FPU pool Dest row offset must fit in ten bits");
    hal::detail::require(is_valid(operation.release), "FPU pool source release must be None, SrcA, SrcB, or Both");
    hal::detail::require(has_valid_index_tracking(operation), "FPU pool index tracking requires Maximum");
}

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

constexpr std::uint32_t eltwise_operation(const Eltwise operation)
{
    switch (operation.operation)
    {
        case EltwiseOperation::Add:
            return TT_OP_ELWADD(
                value(operation.release) /*clear_dvalid*/,
                value(operation.dest_write) /*dest_accum_en*/,
                value(operation.broadcast) /*instr_mod19*/,
                operation.address_modifier /*addr_mode*/,
                operation.dest_row_offset /*dst*/);
        case EltwiseOperation::Subtract:
            return TT_OP_ELWSUB(
                value(operation.release) /*clear_dvalid*/,
                value(operation.dest_write) /*dest_accum_en*/,
                value(operation.broadcast) /*instr_mod19*/,
                operation.address_modifier /*addr_mode*/,
                operation.dest_row_offset /*dst*/);
        default:
            return TT_OP_ELWMUL(
                value(operation.release) /*clear_dvalid*/,
                value(operation.dest_write) /*dest_accum_en*/,
                value(operation.broadcast) /*instr_mod19*/,
                operation.address_modifier /*addr_mode*/,
                operation.dest_row_offset /*dst*/);
    }
}

constexpr std::uint32_t matmul_operation(const Matmul operation)
{
    return TT_OP_MVMUL(
        value(operation.release) /*clear_dvalid*/,
        value(operation.broadcast) /*instr_mod19*/,
        operation.address_modifier /*addr_mode*/,
        operation.dest_row_offset /*dst*/);
}

constexpr std::uint32_t pool_operation(const Pool operation)
{
    // Bit 19 has no ISA field; 1 is the value every characterized pool call site encodes.
    return operation.function == PoolFunction::Sum ? TT_OP_GAPOOL(
                                                         value(operation.release) /*clear_dvalid*/,
                                                         1 /*instr_mod19*/,
                                                         operation.address_modifier /*addr_mode*/,
                                                         value(operation.indices) /*max_pool_index_en*/,
                                                         operation.dest_row_offset /*dst*/)
                                                   : TT_OP_GMPOOL(
                                                         value(operation.release) /*clear_dvalid*/,
                                                         1 /*instr_mod19*/,
                                                         operation.address_modifier /*addr_mode*/,
                                                         value(operation.indices) /*max_pool_index_en*/,
                                                         operation.dest_row_offset /*dst*/);
}
} // namespace detail

/** @brief Return whether eltwise options can be encoded without truncation. */
constexpr bool is_valid(const Eltwise operation)
{
    return detail::is_valid(operation);
}

/** @brief Return whether matmul options can be encoded and are supported. */
constexpr bool is_valid(const Matmul operation)
{
    return detail::is_valid(operation);
}

/** @brief Return whether pool options can be encoded without truncation. */
constexpr bool is_valid(const Pool operation)
{
    return detail::is_valid(operation);
}

/**
 * @brief Issue or encode the operations of the Matrix Unit (FPU).
 *
 * Template forms check their options at compile time and issue one immediate instruction; runtime
 * forms check them with LLK_ASSERT.
 *
 * @code
 * MatrixUnit.eltwise<Eltwise {.operation = EltwiseOperation::Add, .release = SourceRelease::Both}>();
 * MatrixUnit.matmul({.address_modifier = addr_mod, .release = SourceRelease::None});
 *
 * constexpr std::uint32_t mvmul = MatrixUnit.operation().matmul<Matmul {.release = SourceRelease::SrcA}>();
 * @endcode
 */
class Fpu
{
public:
    /** @brief Encode Matrix Unit operations without issuing them, for MOP or replay use. */
    class Operations
    {
    public:
        /**
         * @brief Encode an eltwise operation (ELWADD, ELWSUB or ELWMUL).
         *
         * @tparam Options: Eltwise options.
         */
        template <Eltwise Options>
        [[nodiscard]] constexpr hal::Operation eltwise() const
        {
            static_assert(is_valid(Options), "invalid eltwise options");
            return hal::detail::make_operation(detail::eltwise_operation(Options));
        }

        /**
         * @brief Encode a runtime-selected eltwise operation (ELWADD, ELWSUB or ELWMUL).
         *
         * @param options: Eltwise options.
         * @note Invalid options fail constant evaluation; at runtime, enable LLK assertions to diagnose them.
         */
        [[nodiscard]] inline constexpr __attribute__((always_inline)) hal::Operation eltwise(const Eltwise options) const
        {
            detail::validate(options);
            return hal::detail::make_operation(detail::eltwise_operation(options));
        }

        /**
         * @brief Encode a matmul (MVMUL).
         *
         * @tparam Options: Matrix-multiply options.
         */
        template <Matmul Options>
        [[nodiscard]] constexpr hal::Operation matmul() const
        {
            static_assert(is_valid(Options), "invalid matmul options");
            return hal::detail::make_operation(detail::matmul_operation(Options));
        }

        /**
         * @brief Encode a runtime-selected matmul (MVMUL).
         *
         * @param options: Matrix-multiply options.
         * @note Invalid options fail constant evaluation; at runtime, enable LLK assertions to diagnose them.
         */
        [[nodiscard]] inline constexpr __attribute__((always_inline)) hal::Operation matmul(const Matmul options) const
        {
            detail::validate(options);
            return hal::detail::make_operation(detail::matmul_operation(options));
        }

        /**
         * @brief Encode a pooling reduction (GAPOOL or GMPOOL).
         *
         * @tparam Options: Pool options.
         */
        template <Pool Options>
        [[nodiscard]] constexpr hal::Operation pool() const
        {
            static_assert(is_valid(Options), "invalid pool options");
            return hal::detail::make_operation(detail::pool_operation(Options));
        }

        /**
         * @brief Encode a runtime-selected pooling reduction (GAPOOL or GMPOOL).
         *
         * @param options: Pool options.
         * @note Invalid options fail constant evaluation; at runtime, enable LLK assertions to diagnose them.
         */
        [[nodiscard]] inline constexpr __attribute__((always_inline)) hal::Operation pool(const Pool options) const
        {
            detail::validate(options);
            return hal::detail::make_operation(detail::pool_operation(options));
        }
    };

    /** @brief Return an encoding view of the Matrix Unit. */
    [[nodiscard]] constexpr Operations operation() const
    {
        return {};
    }

    /** @brief Issue an eltwise operation (ELWADD, ELWSUB or ELWMUL). */
    template <Eltwise Options>
    inline __attribute__((always_inline)) void eltwise() const
    {
        hal::detail::run_immediate<Operations {}.eltwise<Options>()>();
    }

    /**
     * @brief Issue a runtime-selected eltwise operation (ELWADD, ELWSUB or ELWMUL).
     *
     * @note Enable LLK assertions to diagnose invalid options; disabled assertions add no runtime check.
     */
    inline __attribute__((always_inline)) void eltwise(const Eltwise options) const
    {
        hal::run(operation().eltwise(options));
    }

    /** @brief Issue a matmul (MVMUL). */
    template <Matmul Options>
    inline __attribute__((always_inline)) void matmul() const
    {
        hal::detail::run_immediate<Operations {}.matmul<Options>()>();
    }

    /**
     * @brief Issue a runtime-selected matmul (MVMUL).
     *
     * @note Enable LLK assertions to diagnose invalid options; disabled assertions add no runtime check.
     */
    inline __attribute__((always_inline)) void matmul(const Matmul options) const
    {
        hal::run(operation().matmul(options));
    }

    /** @brief Issue a pooling reduction (GAPOOL or GMPOOL). */
    template <Pool Options>
    inline __attribute__((always_inline)) void pool() const
    {
        hal::detail::run_immediate<Operations {}.pool<Options>()>();
    }

    /**
     * @brief Issue a runtime-selected pooling reduction (GAPOOL or GMPOOL).
     *
     * @note Enable LLK assertions to diagnose invalid options; disabled assertions add no runtime check.
     */
    inline __attribute__((always_inline)) void pool(const Pool options) const
    {
        hal::run(operation().pool(options));
    }

    /**
     * @brief Read the sticky special-value flags (such as NaN or infinity) the Matrix Unit has raised since the last clear.
     *
     * One RISC load of RISCV_DEBUG_REG_FPU_STICKY_BITS; it is not ordered with Tensix instructions still in flight.
     */
    [[nodiscard]] inline __attribute__((always_inline)) std::uint32_t special_value_flags() const
    {
        return hal::cfg::detail::reg_read(RISCV_DEBUG_REG_FPU_STICKY_BITS);
    }

    /** @brief Clear the Matrix Unit's sticky special-value flags with one RISC store. */
    inline __attribute__((always_inline)) void clear_special_value_flags() const
    {
        hal::cfg::detail::reg_write(RISCV_DEBUG_REG_FPU_STICKY_BITS, 0);
    }
};

/** @brief The Matrix Unit (FPU). */
inline constexpr Fpu MatrixUnit {};

} // namespace hal::fpu
