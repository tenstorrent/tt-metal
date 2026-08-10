// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "access_types.h"
#include "ckernel.h" // cfg_read, get_cfg_pointer, cfg_rmw, RDCFG, SETC16
#include "composition.h"
#include "detail/gpr_operand.h"
#include "detail/mmio_read.h"
#include "detail/write_backend.h"
#include "registers.h"

namespace hal::cfg
{

// Public hardware-access entry points. Field/word construction lives in
// composition.h; implementation helpers live under cfg/detail/.

// -------------------------------------------------------------------------------------------------
// GPR operands and transfer policy
// -------------------------------------------------------------------------------------------------

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

// -------------------------------------------------------------------------------------------------
// RISC reads
// -------------------------------------------------------------------------------------------------

/**
 * @brief Read and extract a state- or thread-CFG field through RISC MMIO.
 *
 * @tparam B  must be @ref Access::MMIO.
 * @tparam F  reference to a generated `static constexpr Field`.
 * @tparam S  section (@ref Sec), defaults to S0.
 * @tparam Target thread-CFG bank. Current selects the issuing TRISC; an
 *         explicit T0/T1/T2 target is required from BRISC. Leave this at
 *         Current for state CFG.
 *
 * @return The field value, shifted down to bit zero.
 */
template <Access B, const Field& F, Sec S = Sec::S0, ThreadTarget Target = ThreadTarget::Current>
inline __attribute__((always_inline)) std::uint32_t read()
{
    static_assert(B == Access::MMIO, "value-returning cfg::read() requires Access::MMIO");
    static_assert(F.width <= 32, "field wider than 32b cannot be read through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(F.shamt(S) + F.width <= F.wbits, "field crosses a CFG word boundary");

    if constexpr (F.file == RegisterFile::Thread)
    {
        return (detail::read_thread_word_mmio<Target, F.addr32(S)>() & F.mask(S)) >> F.shamt(S);
    }
    else
    {
        static_assert(Target == ThreadTarget::Current, "ThreadTarget applies only to thread CFG reads");
        return (ckernel::cfg_read(F.addr32(S)) & F.mask(S)) >> F.shamt(S);
    }
}

/**
 * @brief Read a complete state- or thread-CFG word through RISC MMIO.
 *
 * @p F is a typed address anchor and @p WordOffset selects a word relative to
 * the word containing that field. Unlike @ref read, no mask or shift is
 * applied.
 */
template <Access B, const Field& F, std::uint32_t WordOffset = 0, Sec S = Sec::S0, ThreadTarget Target = ThreadTarget::Current>
inline __attribute__((always_inline)) std::uint32_t read_word()
{
    static_assert(B == Access::MMIO, "value-returning cfg::read_word() requires Access::MMIO");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    if constexpr (F.file == RegisterFile::Thread)
    {
        static_assert(F.addr32(S) + WordOffset < detail::ThreadCfgWordCount, "thread CFG word offset crosses the selected thread bank");
        return detail::read_thread_word_mmio<Target, F.addr32(S) + WordOffset>() & 0xffffu;
    }
    else
    {
        static_assert(Target == ThreadTarget::Current, "ThreadTarget applies only to thread CFG reads");
        return ckernel::cfg_read(F.addr32(S) + WordOffset);
    }
}

/**
 * @brief Read a complete CFG word through an already-resolved RISC MMIO bank.
 *
 * This is the read-side counterpart of the pointer-taking `write()` overload
 * and avoids re-reading CFG_STATE_ID when several words use the same bank.
 */
template <Access B, const Field& F, std::uint32_t WordOffset = 0, Sec S = Sec::S0>
inline __attribute__((always_inline)) std::uint32_t read_word(const volatile std::uint32_t* tt_reg_ptr cfg)
{
    static_assert(B == Access::MMIO, "an already-resolved CFG pointer is valid only for the Access::MMIO backend");
    static_assert(F.file == RegisterFile::State, "Access::MMIO targets the state CFG");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    return cfg[F.addr32(S) + WordOffset];
}

// -------------------------------------------------------------------------------------------------
// Tensix GPR transfers
// -------------------------------------------------------------------------------------------------

/**
 * @brief Issue RDCFG through the common read() entry point.
 *
 * RDCFG returns a complete 32-bit CFG word from the bank selected by the
 * current thread's CFG_STATE_ID. Use `F.mask(S)` and `F.shamt(S)` when
 * subsequent GPR operations need only the selected field.
 */
template <Access A, const Field& F, Sec S = Sec::S0, std::uint32_t GprIndex, GprTransferSize Size, WrcfgCompletion Completion>
inline __attribute__((always_inline)) void read(detail::GprOperand<GprIndex, Size, Completion>)
{
    static_assert(A == Access::TensixCfgUnit, "RDCFG requires Access::TensixCfgUnit");
    static_assert(GprIndex != detail::DynamicGprIndex, "RDCFG requires a compile-time GPR index: use gpr<Index>()");
    static_assert(Size == GprTransferSize::Bits32, "RDCFG supports 32-bit reads only");
    static_assert(F.file == RegisterFile::State, "RDCFG cannot read thread CFG (SETC16) fields");
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
 * @tparam S: Repeated descriptor section.
 * @tparam GprIndex: Compile-time destination GPR index.
 * @param destination: Common GPR operand receiving the CFG word.
 */
template <Access A, const Field& F, Sec S = Sec::S0, std::uint32_t GprIndex>
inline __attribute__((always_inline)) void read(const hal::Gpr<GprIndex> destination)
{
    read<A, F, S>(detail::with_cfg_policy<GprTransferSize::Bits32, WrcfgCompletion::Wait>(destination));
}

/**
 * @brief Move one or four GPR words to state CFG through the selected Tensix unit.
 *
 * @tparam A: Access path, values = <TensixCfgUnit/TensixScalarUnit>.
 * @tparam F: Field identifying the first destination CFG word.
 * @tparam S: Repeated descriptor section.
 * @tparam GprIndex: Compile-time GPR index or the runtime-index sentinel.
 * @tparam Size: Transfer width, values = <Bits32/Bits128>.
 * @tparam Completion: WRCFG completion policy used by TensixCfgUnit.
 * @param source: GPR operand supplying one or four complete words.
 * @note TensixCfgUnit emits WRCFG and its requested completion NOP. TensixScalarUnit
 *       emits REG2FLOP without a completion NOP and accepts only THCON destinations.
 */
template <Access A, const Field& F, Sec S = Sec::S0, std::uint32_t GprIndex, GprTransferSize Size, WrcfgCompletion Completion>
inline __attribute__((always_inline)) void write(const detail::GprOperand<GprIndex, Size, Completion> source)
{
    static_assert(
        A == Access::TensixCfgUnit || A == Access::TensixScalarUnit, "GPR-backed cfg::write requires Access::TensixCfgUnit or Access::TensixScalarUnit");
    static_assert(F.file == RegisterFile::State, "GPR-backed cfg::write cannot write thread CFG (SETC16) fields");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(F.shamt(S) == 0, "GPR cfg::write must start at the beginning of a CFG word");
    if constexpr (Size == GprTransferSize::Bits128)
    {
        static_assert((F.addr32(S) & 0x3u) == 0u, "128-bit GPR cfg::write destination must be four-word aligned");
    }

    if constexpr (A == Access::TensixScalarUnit)
    {
        constexpr std::uint32_t address = F.addr32(S);
        static_assert(
            address >= THCON_CFGREG_BASE_ADDR32 && address < GLOBAL_CFGREG_BASE_ADDR32, "Access::TensixScalarUnit supports THCON CFG destinations only");
        if constexpr (Size == GprTransferSize::Bits128)
        {
            static_assert(address + 3u < GLOBAL_CFGREG_BASE_ADDR32, "128-bit REG2FLOP transfer crosses the THCON CFG range");
        }

        constexpr std::uint32_t size_sel   = Size == GprTransferSize::Bits128 ? 0u : 1u;
        constexpr std::uint32_t flop_index = address - THCON_CFGREG_BASE_ADDR32;
        if constexpr (GprIndex == detail::DynamicGprIndex)
        {
            LLK_ASSERT(source.index < 64u, "REG2FLOP GPR index must be in [0, 63]");
            if constexpr (Size == GprTransferSize::Bits128)
            {
                LLK_ASSERT((source.index & 0x3u) == 0u, "128-bit REG2FLOP source GPR must be four-word aligned");
            }
            TT_REG2FLOP(size_sel, 0, 0, 0, flop_index, source.index);
        }
        else
        {
            static_assert(GprIndex < 64u, "REG2FLOP GPR index must be in [0, 63]");
            if constexpr (Size == GprTransferSize::Bits128)
            {
                static_assert((GprIndex & 0x3u) == 0u, "128-bit REG2FLOP source GPR must be four-word aligned");
            }
            TTI_REG2FLOP(size_sel, 0, 0, 0, flop_index, GprIndex);
        }
    }
    else
    {
        if constexpr (GprIndex == detail::DynamicGprIndex)
        {
            TT_WRCFG(source.index, Size == GprTransferSize::Bits128, F.addr32(S));
        }
        else
        {
            TTI_WRCFG(GprIndex, Size == GprTransferSize::Bits128, F.addr32(S));
        }
        if constexpr (Completion == WrcfgCompletion::Wait)
        {
            TTI_NOP;
        }
    }
}

/**
 * @brief Issue a default 32-bit CFG or THCON write using a common GPR operand.
 *
 * @tparam A: Access path, values = <TensixCfgUnit/TensixScalarUnit>.
 * @tparam F: Field identifying the destination CFG word.
 * @tparam S: Repeated descriptor section.
 * @tparam GprIndex: Compile-time or runtime source GPR index.
 * @param source: Common GPR operand supplying one complete word.
 */
template <Access A, const Field& F, Sec S = Sec::S0, std::uint32_t GprIndex>
inline __attribute__((always_inline)) void write(const hal::Gpr<GprIndex> source)
{
    write<A, F, S>(detail::with_cfg_policy<GprTransferSize::Bits32, WrcfgCompletion::Wait>(source));
}

// -------------------------------------------------------------------------------------------------
// Public write API
// -------------------------------------------------------------------------------------------------

/**
 * @brief Write one or more composed CFG words.
 *
 * One ConfigWord is one logical word update. Several ConfigWords provide a
 * transaction-style interface, but hardware still needs one update for every
 * distinct physical address.
 */
template <Access A, typename First, typename... Rest, std::enable_if_t<detail::is_config_word_v<First>, int> = 0>
inline __attribute__((always_inline)) void write(const First& first, const Rest&... rest)
{
    detail::write_words<A>(first, rest...);
}

/**
 * @brief Write composed CFG words through an already-resolved RISC MMIO bank.
 *
 * This overload avoids re-reading CFG_STATE_ID when a hot loop performs
 * several writes to the same bank. The pointer must come from
 * `ckernel::get_cfg_pointer()`.
 */
template <Access A, typename First, typename... Rest, std::enable_if_t<detail::is_config_word_v<First>, int> = 0>
inline __attribute__((always_inline)) void write(volatile std::uint32_t* tt_reg_ptr cfg, const First& first, const Rest&... rest)
{
    static_assert(A == Access::MMIO, "an already-resolved CFG pointer requires Access::MMIO");
    detail::write_words_mmio(cfg, first, rest...);
}

/**
 * @brief Writer passed to the lambda overload of cfg::write().
 *
 * A RISC batch resolves the active state bank once. The callable may emit
 * composed words, field assignments, individual fields, or arrays, and may
 * use ordinary control flow to iterate over runtime data.
 */
template <Access A>
class WriteBatch
{
public:
    inline __attribute__((always_inline)) WriteBatch()
    {
        static_assert(A != Access::TensixScalarUnit, "Access::TensixScalarUnit supports only GPR-backed cfg::write");
        if constexpr (A == Access::MMIO)
        {
            cfg_ = ckernel::get_cfg_pointer();
        }
    }

    inline __attribute__((always_inline)) explicit WriteBatch(volatile std::uint32_t* tt_reg_ptr cfg) : cfg_(cfg)
    {
        static_assert(A == Access::MMIO, "an already-resolved CFG pointer requires Access::MMIO");
    }

    template <typename First, typename... Rest, std::enable_if_t<detail::is_config_word_v<First>, int> = 0>
    inline __attribute__((always_inline)) void operator()(const First& first, const Rest&... rest) const
    {
        if constexpr (A == Access::MMIO)
        {
            detail::write_words_mmio(cfg_, first, rest...);
        }
        else
        {
            detail::write_words<A>(first, rest...);
        }
    }

    template <typename First, typename... Rest, std::enable_if_t<detail::is_field_assignment_v<First> && (detail::is_field_assignment_v<Rest> && ...), int> = 0>
    inline __attribute__((always_inline)) void operator()(const First& first, const Rest&... rest) const
    {
        (*this)(word(first, rest...));
    }

    template <const Field& F, Sec S = Sec::S0>
    inline __attribute__((always_inline)) void field(const std::uint32_t value) const
    {
        (*this)(set<F, S>(value));
    }

    template <const Field& F, std::uint32_t Count, Sec S = Sec::S0, std::size_t ArrayCount>
    inline __attribute__((always_inline)) void words(const std::uint32_t (&values)[ArrayCount]) const
    {
        static_assert(A == Access::MMIO, "array writes require Access::MMIO");
        detail::write_array_mmio<F, Count, S>(cfg_, values);
    }

private:
    volatile std::uint32_t* cfg_ = nullptr;
};

/**
 * @brief Perform a group of CFG writes with ordinary C++ control flow.
 *
 * @code
 * write<Access::MMIO>([&](auto& out) {
 *     out.template field<PrngSeed::Seed_Val>(seed);
 *     out(word<Thcon[Reg1].Row_start_section_size>(packed));
 * });
 * @endcode
 */
template <Access A, typename Configure, std::enable_if_t<std::is_invocable_v<Configure, WriteBatch<A>&>, int> = 0>
inline __attribute__((always_inline)) void write(Configure&& configure)
{
    WriteBatch<A> batch;
    configure(batch);
}

/**
 * @brief Perform a group of RISC MMIO writes through an already-resolved CFG bank.
 *
 * @tparam A Access path, values = <MMIO>.
 * @tparam Configure Callable accepting a @ref WriteBatch.
 * @param cfg Active CFG bank returned by `ckernel::get_cfg_pointer()`.
 * @param configure Callable receiving a @ref WriteBatch.
 */
template <Access A, typename Configure, std::enable_if_t<std::is_invocable_v<Configure, WriteBatch<A>&>, int> = 0>
inline __attribute__((always_inline)) void write(volatile std::uint32_t* tt_reg_ptr cfg, Configure&& configure)
{
    static_assert(A == Access::MMIO, "an already-resolved CFG pointer requires Access::MMIO");
    WriteBatch<A> batch(cfg);
    configure(batch);
}

/**
 * @brief Write a fixed-size array of consecutive prepacked CFG words.
 */
template <Access A, const Field& F, std::uint32_t Count, Sec S = Sec::S0, std::size_t ArrayCount>
inline __attribute__((always_inline)) void write(const std::uint32_t (&values)[ArrayCount])
{
    static_assert(A == Access::MMIO, "array writes require Access::MMIO");
    detail::write_array_mmio<F, Count, S>(ckernel::get_cfg_pointer(), values);
}

/**
 * @brief Combine and write fields from one physical CFG word.
 *
 * @code
 * write<Access::TensixCfgUnit>(
 *     set<AluFormatSpecReg0::SrcA>(src_a),
 *     set<AluFormatSpecReg1::SrcB>(src_b),
 *     set<AluAccCtrl::Fp32_enabled>(fp32));
 * @endcode
 */
template <
    Access A,
    typename First,
    typename... Rest,
    std::enable_if_t<detail::is_field_assignment_v<First> && (detail::is_field_assignment_v<Rest> && ...), int> = 0>
inline __attribute__((always_inline)) void write(const First& first, const Rest&... rest)
{
    if constexpr (detail::is_constant_field_assignment_v<First> && (detail::is_constant_field_assignment_v<Rest> && ...))
    {
        detail::write_constant_assignments<A, First, Rest...>();
    }
    else
    {
        detail::write_word<A>(word(first, rest...));
    }
}

/**
 * @brief Write @p value into a CFG field through the chosen access path.
 *
 * @tparam A  @ref Access — RISC MMIO or Tensix instruction.
 * @tparam F  reference to a generated `static constexpr Field` (e.g. Reg::Field).
 * @tparam S  section (@ref Sec), defaults to S0.
 *
 * @note SETC16 (Access::TensixCfgUnit on the Thread file) writes a whole 16-bit thread
 *       word and has no per-field RMW; for a word packing several fields,
 *       compose the value and write it whole (as addr_mod_t does).
 */
template <Access A, const Field& F, Sec S = Sec::S0>
inline __attribute__((always_inline)) void write(const std::uint32_t value)
{
    static_assert(
        A == Access::MMIO || A == Access::TensixCfgUnit,
        "value-backed cfg::write requires Access::MMIO or Access::TensixCfgUnit; Access::TensixScalarUnit requires a GPR operand");
    static_assert(F.width <= 32, "field wider than 32b cannot be written through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");

    const std::uint32_t a = F.addr32(S);

    if constexpr (A == Access::MMIO)
    {
        static_assert(F.file == RegisterFile::State, "RISC writes target state CFG; use Access::TensixCfgUnit for thread CFG");
        if constexpr (F.width >= 32)
        {
            ckernel::get_cfg_pointer()[a] = value; // whole 32-bit word
        }
        else
        {
            ckernel::cfg_rmw(a, F.shamt(S), F.mask(S), value); // read-modify-write
        }
    }
    else // Access::TensixCfgUnit
    {
        if constexpr (F.file == RegisterFile::Thread)
        {
            TT_SETC16(a, value << F.shamt(S));
        }
        else
        {
            detail::cfg_reg_rmw_tensix<F.addr32(S), F.shamt(S), F.mask(S)>(value);
        }
    }
}

/**
 * @brief Fully compile-time write: the whole instruction word (address AND
 *        data) is embedded into the instruction stream via .ttinsn (TTI_*),
 *        so nothing is composed or pushed at runtime.
 *
 * @tparam A      must be @ref Access::TensixCfgUnit (RISC MMIO is a runtime store).
 * @tparam F      reference to a generated `static constexpr Field`.
 * @tparam Value  compile-time value to place in the field.
 * @tparam S      section, defaults to S0.
 *
 * @note Only the config-word bytes the field actually covers are emitted
 *       (RMWCIB per non-zero mask byte), pruned at compile time.
 */
template <Access A, const Field& F, std::uint32_t Value, Sec S = Sec::S0>
inline __attribute__((always_inline)) void write()
{
    static_assert(A == Access::TensixCfgUnit, "compile-time instruction emission requires Access::TensixCfgUnit");
    static_assert(F.width <= 32, "field wider than 32b cannot be written through a single value");
    static_assert(static_cast<std::uint32_t>(S) < F.count, "section index out of range for this register");
    static_assert(Value <= ((std::uint64_t {1} << F.width) - 1u), "value exceeds field width");

    if constexpr (F.file == RegisterFile::Thread)
    {
        TTI_SETC16(F.addr32(S), (Value << F.shamt(S)) & 0xffffu);
    }
    else
    {
        constexpr std::uint32_t wr = Value << F.shamt(S);
        constexpr std::uint32_t m  = F.mask(S);
        detail::write_constant_word<F.file, F.addr32(S), m, wr>();
    }
}

} // namespace hal::cfg
