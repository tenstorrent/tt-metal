// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "access_types.h"
#include "ckernel.h" // RDCFG, SETC16
#include "detail/gpr_operand.h"
#include "detail/mmio_read.h"
#include "detail/state_bank.h"
#include "detail/write_backend.h"
#include "detail/write_operands.h"
#include "registers.h"

namespace hal::cfg
{

// Public operand factories and hardware-access entry points.
// Read from factories to reads, then from basic writes to grouped writes and batches.
// Operand types and implementation helpers live under cfg/detail/.

// -------------------------------------------------------------------------------------------------
// Operand factories
//
// Construct field assignments and GPR operands without accessing hardware.
// Runtime and constant variants are adjacent; addresses and masks remain compile-time metadata.
// -------------------------------------------------------------------------------------------------

/**
 * @brief Associate a runtime value with a generated CFG field.
 */
template <const Field& F, Sec S>
inline constexpr FieldAssignment<F, S> set(const std::uint32_t value)
{
    return {value};
}

/**
 * @brief Associate a compile-time value with a generated CFG field.
 */
template <const Field& F, Sec S, std::uint32_t Value>
inline constexpr ConstantFieldAssignment<F, S, Value> set()
{
    return {};
}

/**
 * @brief Construct a compile-time-indexed Tensix GPR operand.
 */
template <std::uint32_t Index, GprTransferSize Size = GprTransferSize::Bits32, WrcfgCompletion Completion = WrcfgCompletion::Wait>
inline constexpr auto gpr()
{
    return detail::with_cfg_policy<Size, Completion>(hal::gpr<Index>());
}

/**
 * @brief Construct a runtime-indexed Tensix GPR operand.
 */
template <GprTransferSize Size = GprTransferSize::Bits32, WrcfgCompletion Completion = WrcfgCompletion::Wait>
inline constexpr auto gpr(const std::uint32_t index)
{
    return detail::with_cfg_policy<Size, Completion>(hal::gpr(index));
}

/**
 * @brief Bind a GPR source to a complete state-CFG word destination.
 *
 * The returned operation can be combined with @ref set assignments in one
 * heterogeneous @ref write call. It acts as an ordering barrier between
 * automatically grouped assignment runs.
 *
 * @tparam F: Field anchoring the destination CFG word.
 * @tparam S: Repeated descriptor section; compilation fails when it is outside F.count.
 * @tparam GprIndex: Compile-time GPR index or the runtime-index sentinel.
 * @tparam Size: Transfer width, values = <Bits32/Bits128>.
 * @tparam Completion: WRCFG completion policy.
 * @param source: GPR source and transfer policy.
 */
template <const Field& F, Sec S, std::uint32_t GprIndex, GprTransferSize Size, WrcfgCompletion Completion>
inline constexpr auto from_gpr(const detail::GprOperand<GprIndex, Size, Completion> source)
{
    return GprWrite<F, S, detail::GprOperand<GprIndex, Size, Completion>> {source};
}

/**
 * @brief Bind a default 32-bit GPR source to a state-CFG destination.
 */
template <const Field& F, Sec S, std::uint32_t GprIndex>
inline constexpr auto from_gpr(const hal::Gpr<GprIndex> source)
{
    return from_gpr<F, S>(detail::with_cfg_policy<GprTransferSize::Bits32, WrcfgCompletion::Wait>(source));
}

// -------------------------------------------------------------------------------------------------
// Reads: MMIO values and Tensix GPR transfers
//
// MMIO reads return a value to the RISC. RDCFG transfers a complete word into a Tensix GPR.
// -------------------------------------------------------------------------------------------------

/**
 * @brief Read and extract a state- or thread-CFG field through RISC MMIO.
 *
 * @tparam B  must be @ref Access::MMIO.
 * @tparam F  reference to a generated `static constexpr Field`.
 * @tparam S: Section (@ref Sec); compilation fails when it is outside F.count.
 * @tparam Target thread-CFG bank. Current selects the issuing TRISC; an
 *         explicit T0/T1/T2 target is required from BRISC. Leave this at
 *         Current for state CFG.
 *
 * @return The field value, shifted down to bit zero.
 */
template <Access B, const Field& F, Sec S, ThreadTarget Target = ThreadTarget::Current>
inline __attribute__((always_inline)) std::uint32_t read()
{
    static_assert(B == Access::MMIO, "value-returning cfg::read() requires Access::MMIO");
    static_assert(F.width <= 32, "field wider than 32b cannot be read through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(F.shamt(S) + F.width <= F.word_size, "field crosses a CFG word boundary");

    if constexpr (F.scope == RegisterScope::Thread)
    {
        return (detail::read_thread_word_mmio<Target, F.addr32(S)>() & F.mask(S)) >> F.shamt(S);
    }
    else
    {
        static_assert(Target == ThreadTarget::Current, "ThreadTarget applies only to thread CFG reads");
        return (detail::read_state_word_mmio(F.addr32(S)) & F.mask(S)) >> F.shamt(S);
    }
}

/**
 * @brief Read a complete state- or thread-CFG word through RISC MMIO.
 *
 * @p F is a typed address anchor and @p WordOffset selects a word relative to
 * the word containing that field. Unlike @ref read, no mask or shift is
 * applied.
 */
template <Access B, const Field& F, Sec S, std::uint32_t WordOffset = 0, ThreadTarget Target = ThreadTarget::Current>
inline __attribute__((always_inline)) std::uint32_t read_word()
{
    static_assert(B == Access::MMIO, "value-returning cfg::read_word() requires Access::MMIO");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    if constexpr (F.scope == RegisterScope::Thread)
    {
        static_assert(F.addr32(S) + WordOffset < detail::ThreadCfgWordCount, "thread CFG word offset crosses the selected thread bank");
        return detail::read_thread_word_mmio<Target, F.addr32(S) + WordOffset>() & 0xffffu;
    }
    else
    {
        static_assert(Target == ThreadTarget::Current, "ThreadTarget applies only to thread CFG reads");
        return detail::read_state_word_mmio(F.addr32(S) + WordOffset);
    }
}

/**
 * @brief Read a complete CFG word through an already-resolved RISC MMIO bank.
 *
 * Reuses the caller's bank selection when reading several words.
 */
template <Access B, const Field& F, Sec S, std::uint32_t WordOffset = 0>
inline __attribute__((always_inline)) std::uint32_t read_word(const volatile std::uint32_t* tt_reg_ptr cfg)
{
    static_assert(B == Access::MMIO, "an already-resolved CFG pointer is valid only for the Access::MMIO backend");
    static_assert(F.scope == RegisterScope::State, "Access::MMIO targets the state CFG");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    return cfg[F.addr32(S) + WordOffset];
}

/**
 * @brief Issue RDCFG through the common read() entry point.
 *
 * RDCFG returns a complete 32-bit CFG word from the bank selected by the
 * current thread's CFG_STATE_ID. Use `F.mask(S)` and `F.shamt(S)` when
 * subsequent GPR operations need only the selected field.
 */
template <Access A, const Field& F, Sec S, std::uint32_t GprIndex, GprTransferSize Size, WrcfgCompletion Completion>
inline __attribute__((always_inline)) void read(detail::GprOperand<GprIndex, Size, Completion>)
{
    static_assert(A == Access::TensixCfgUnit, "RDCFG requires Access::TensixCfgUnit");
    static_assert(GprIndex != detail::DynamicGprIndex, "RDCFG requires a compile-time GPR index: use gpr<Index>()");
    static_assert(Size == GprTransferSize::Bits32, "RDCFG supports 32-bit reads only");
    static_assert(F.scope == RegisterScope::State, "RDCFG cannot read thread CFG (SETC16) fields");
    static_assert(F.width <= 32, "field wider than 32b cannot be selected through a single CFG word");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(F.shamt(S) + F.width <= 32, "field crosses a CFG word boundary");

    TTI_RDCFG(GprIndex, F.addr32(S));
}

/**
 * @brief Issue a default 32-bit RDCFG using a common GPR operand.
 *
 * @tparam A: Access path, values = <TensixCfgUnit>.
 * @tparam F: Field identifying the source CFG word.
 * @tparam S: Repeated descriptor section; compilation fails when it is outside F.count.
 * @tparam GprIndex: Compile-time destination GPR index.
 * @param destination: Common GPR operand receiving the CFG word.
 */
template <Access A, const Field& F, Sec S, std::uint32_t GprIndex>
inline __attribute__((always_inline)) void read(const hal::Gpr<GprIndex> destination)
{
    read<A, F, S>(detail::with_cfg_policy<GprTransferSize::Bits32, WrcfgCompletion::Wait>(destination));
}

// -------------------------------------------------------------------------------------------------
// Writes: one field, runtime or constant value
//
// The runtime-value overload accepts a RISC value; the constant-value overload emits an immediate
// Tensix instruction. Both resolve the field address and bit position at compile time.
// -------------------------------------------------------------------------------------------------

/**
 * @brief Write a runtime value into a CFG field through the chosen access path.
 *
 * The field address, mask, and shift are compile-time constants. Only @p value
 * is supplied at runtime; the backend uses MMIO or TT_* instructions.
 *
 * @tparam A  @ref Access — RISC MMIO or Tensix instruction.
 * @tparam F  reference to a generated `static constexpr Field` (e.g. Reg::Field).
 * @tparam S: Section (@ref Sec); compilation fails when it is outside F.count.
 *
 * @note SETC16 (Access::TensixCfgUnit on the Thread scope) writes a whole 16-bit thread
 *       word and has no per-field RMW; for a word packing several fields,
 *       compose the value and write it whole (as addr_mod_t does).
 */
template <Access A, const Field& F, Sec S>
inline __attribute__((always_inline)) void write(const std::uint32_t value)
{
    static_assert(
        A == Access::MMIO || A == Access::TensixCfgUnit,
        "value-backed cfg::write requires Access::MMIO or Access::TensixCfgUnit; Access::TensixScalarUnit requires a GPR operand");
    static_assert(F.width <= 32, "field wider than 32b cannot be written through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    constexpr std::uint32_t cfg_word_addr = F.addr32(S);
    constexpr std::uint32_t max_value     = F.width == 32 ? 0xffffffffu : ((std::uint32_t {1} << F.width) - 1u);
    LLK_ASSERT(value <= max_value, "value exceeds field width");

    if constexpr (A == Access::MMIO)
    {
        static_assert(F.scope == RegisterScope::State, "RISC writes target state CFG; use Access::TensixCfgUnit for thread CFG");
        if constexpr (F.width == 32)
        {
            detail::state_cfg_bank()[cfg_word_addr] = value; // whole 32-bit word
        }
        else
        {
            detail::rmw_state_word_mmio(cfg_word_addr, F.shamt(S), F.mask(S), value); // read-modify-write
        }
    }
    else // Access::TensixCfgUnit
    {
        if constexpr (F.scope == RegisterScope::Thread)
        {
            TT_SETC16(cfg_word_addr, (value << F.shamt(S)) & 0xffffu);
        }
        else
        {
            detail::write_runtime_rmwcib<cfg_word_addr, F.shamt(S), F.mask(S)>(value);
        }
    }
}

/**
 * @brief Write a compile-time constant using immediate Tensix instructions.
 *
 * Both address and data are embedded in the instruction stream via .ttinsn
 * (TTI_*), avoiding runtime opcode construction. The hardware write still
 * executes at runtime.
 *
 * @tparam A      must be @ref Access::TensixCfgUnit (RISC MMIO is a runtime store).
 * @tparam F      reference to a generated `static constexpr Field`.
 * @tparam S: Section (@ref Sec); compilation fails when it is outside F.count.
 * @tparam Value  compile-time value to place in the field.
 *
 * @note Only the config-word bytes the field actually covers are emitted
 *       (RMWCIB per non-zero mask byte), pruned at compile time.
 */
template <Access A, const Field& F, Sec S, std::uint32_t Value>
inline __attribute__((always_inline)) void write()
{
    static_assert(A == Access::TensixCfgUnit, "compile-time instruction emission requires Access::TensixCfgUnit");
    static_assert(F.width <= 32, "field wider than 32b cannot be written through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(Value <= ((std::uint64_t {1} << F.width) - 1u), "value exceeds field width");

    detail::write_constant_word<F.scope, F.addr32(S), F.mask(S), Value << F.shamt(S)>();
}

// -------------------------------------------------------------------------------------------------
// Writes: grouped field assignments and mixed operation sequences
//
// Combine set() operands by physical CFG word. A group may contain runtime values, constants, or both.
// Each from_gpr() operation separates assignment runs and emits its transfer in source order.
// -------------------------------------------------------------------------------------------------

/**
 * @brief Group field assignments and emit ordered GPR transfers.
 *
 * Assignments with the same register scope and resolved address are composed
 * even when they are not adjacent. Distinct words are emitted in the order
 * their addresses first appear. A Tensix group made entirely from constant
 * assignments uses TTI instructions; a group containing a runtime value uses
 * TT instructions. Use separate calls when hardware programming order matters.
 * Grouping never crosses a @ref from_gpr operation; its transfer and completion
 * policy are emitted between the surrounding assignment runs.
 *
 * @code
 * write<Access::TensixCfgUnit>(
 *     set<AluFormatSpecReg0::SrcA, Sec::S0>(src_a),
 *     set<AluFormatSpecReg1::SrcB, Sec::S0>(src_b),
 *     set<AluAccCtrl::Fp32_enabled, Sec::S0>(fp32));
 * @endcode
 *
 * @tparam A: Access path, values = <MMIO/TensixCfgUnit>; from_gpr requires TensixCfgUnit.
 * @tparam First: First operation type returned by @ref set or @ref from_gpr.
 * @tparam Rest: Remaining operation types.
 * @param first: First write operation.
 * @param rest: Remaining write operations.
 */
template <
    Access A,
    typename First,
    typename... Rest,
    std::enable_if_t<detail::is_write_operation_v<First> && (detail::is_write_operation_v<Rest> && ...), int> = 0>
inline __attribute__((always_inline)) void write(const First& first, const Rest&... rest)
{
    if constexpr (detail::is_field_assignment_v<First> && (detail::is_field_assignment_v<Rest> && ...))
    {
        detail::write_assignments<A>(first, rest...);
    }
    else
    {
        detail::write_mixed_operations<A>(first, rest...);
    }
}

// -------------------------------------------------------------------------------------------------
// Writes: direct GPR transfers
//
// GPR transfers replace complete words through the CFG or scalar unit.
// -------------------------------------------------------------------------------------------------

/**
 * @brief Move one or four GPR words to state CFG through the selected Tensix unit.
 *
 * @tparam A: Access path, values = <TensixCfgUnit/TensixScalarUnit>.
 * @tparam F: Field identifying the first destination CFG word.
 * @tparam S: Repeated descriptor section; compilation fails when it is outside F.count.
 * @tparam GprIndex: Compile-time GPR index or the runtime-index sentinel.
 * @tparam Size: Transfer width, values = <Bits32/Bits128>.
 * @tparam Completion: WRCFG completion policy used by TensixCfgUnit.
 * @param source: GPR operand supplying one or four complete words.
 * @note TensixCfgUnit emits WRCFG and its requested completion NOP. TensixScalarUnit
 *       emits REG2FLOP without a completion NOP and accepts only THCON destinations.
 */
template <Access A, const Field& F, Sec S, std::uint32_t GprIndex, GprTransferSize Size, WrcfgCompletion Completion>
inline __attribute__((always_inline)) void write(const detail::GprOperand<GprIndex, Size, Completion> source)
{
    detail::write_gpr<A>(from_gpr<F, S>(source));
}

/**
 * @brief Issue a default 32-bit CFG or THCON write using a common GPR operand.
 *
 * @tparam A: Access path, values = <TensixCfgUnit/TensixScalarUnit>.
 * @tparam F: Field identifying the destination CFG word.
 * @tparam S: Repeated descriptor section; compilation fails when it is outside F.count.
 * @tparam GprIndex: Compile-time or runtime source GPR index.
 * @param source: Common GPR operand supplying one complete word.
 */
template <Access A, const Field& F, Sec S, std::uint32_t GprIndex>
inline __attribute__((always_inline)) void write(const hal::Gpr<GprIndex> source)
{
    write<A, F, S>(detail::with_cfg_policy<GprTransferSize::Bits32, WrcfgCompletion::Wait>(source));
}

// -------------------------------------------------------------------------------------------------
// Writes: complete word arrays through MMIO
//
// Store consecutive prepacked words without applying a field mask or shift.
// The field descriptor supplies the starting address; WriteBatch::replace handles individual words.
// -------------------------------------------------------------------------------------------------

/**
 * @brief Write a fixed-size array of consecutive prepacked CFG words.
 */
template <Access A, const Field& F, Sec S, std::uint32_t Count, std::size_t ArrayCount>
inline __attribute__((always_inline)) void write(const std::uint32_t (&values)[ArrayCount])
{
    static_assert(A == Access::MMIO, "array writes require Access::MMIO");
    detail::write_array_mmio<F, S, Count>(detail::state_cfg_bank(), values);
}

// -------------------------------------------------------------------------------------------------
// Writes: batches with ordinary C++ control flow
//
// WriteBatch provides a writer for loops and conditionals. MMIO batches resolve the state bank once.
// Each writer call emits its own writes; calls are not accumulated into one combined update.
// -------------------------------------------------------------------------------------------------

/**
 * @brief Writer passed to the lambda overload of cfg::write().
 *
 * A RISC batch resolves the active state bank once. The callable may emit
 * field assignments, individual fields, prepacked words, or arrays, and may
 * use ordinary control flow to iterate over runtime data.
 */
template <Access A>
class WriteBatch
{
public:
    // Resolve the active bank once for MMIO; Tensix batches need no bank pointer.
    inline __attribute__((always_inline)) WriteBatch()
    {
        static_assert(A != Access::TensixScalarUnit, "Access::TensixScalarUnit supports only GPR-backed cfg::write");
        if constexpr (A == Access::MMIO)
        {
            cfg_ = detail::state_cfg_bank();
        }
    }

    // Group set() operands while preserving from_gpr() transfer boundaries.
    template <typename First, typename... Rest, std::enable_if_t<detail::is_write_operation_v<First> && (detail::is_write_operation_v<Rest> && ...), int> = 0>
    inline __attribute__((always_inline)) void operator()(const First& first, const Rest&... rest) const
    {
        if constexpr (detail::is_field_assignment_v<First> && (detail::is_field_assignment_v<Rest> && ...))
        {
            detail::write_assignments_in_bank<A>(cfg_, first, rest...);
        }
        else
        {
            detail::write_mixed_operations<A>(first, rest...);
        }
    }

    // Convenience form for one runtime field assignment.
    template <const Field& F, Sec S>
    inline __attribute__((always_inline)) void field(const std::uint32_t value) const
    {
        (*this)(set<F, S>(value));
    }

    /**
     * @brief Replace one prepacked state-CFG word through the resolved MMIO bank.
     *
     * The anchor supplies only the address; its mask and shift are ignored.
     *
     * @tparam Anchor: Field descriptor anchoring the destination word.
     * @tparam S: Section containing the destination word.
     * @tparam WordOffset: Additional 32-bit word offset from the anchor.
     * @param value: Complete prepacked 32-bit word to store.
     */
    template <const Field& Anchor, Sec S, std::uint32_t WordOffset = 0>
    inline __attribute__((always_inline)) void replace(const std::uint32_t value) const
    {
        static_assert(A == Access::MMIO, "whole-word replacement requires Access::MMIO");
        static_assert(Anchor.scope == RegisterScope::State, "Access::MMIO targets the state CFG");
        static_assert(static_cast<std::uint32_t>(S) < Anchor.count, "section index out of range for this register");

        cfg_[Anchor.addr32(S) + WordOffset] = value;
    }

    // Replace consecutive prepacked words through the batch's MMIO bank.
    template <const Field& F, Sec S, std::uint32_t Count, std::size_t ArrayCount>
    inline __attribute__((always_inline)) void words(const std::uint32_t (&values)[ArrayCount]) const
    {
        static_assert(A == Access::MMIO, "array writes require Access::MMIO");
        detail::write_array_mmio<F, S, Count>(cfg_, values);
    }

private:
    volatile std::uint32_t* cfg_ = nullptr;
};

/**
 * @brief Perform a group of CFG writes with ordinary C++ control flow.
 *
 * @code
 * write<Access::MMIO>([&](auto& out) {
 *     out.template field<PrngSeed::Seed_Val, Sec::S0>(seed);
 *     out.template replace<PackCounters::pack_per_xy_plane, Sec::S0>(packed_counters);
 * });
 * @endcode
 */
template <Access A, typename Configure, std::enable_if_t<std::is_invocable_v<Configure, WriteBatch<A>&>, int> = 0>
inline __attribute__((always_inline)) void write(Configure&& configure)
{
    WriteBatch<A> batch;
    configure(batch);
}

} // namespace hal::cfg
